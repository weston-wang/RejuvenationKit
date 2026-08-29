"""Provider-neutral resource snapshots and reproducible query provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Self, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicFeatureType,
    validate_feature_domain_compatibility,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNRESOLVED_RELEASES = frozenset({"latest", "current", "default"})
_SECRET_KEY_COMPACT_FORMS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "bearer",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "token",
    }
)


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible payload after deterministic canonicalization."""
    normalized = _canonicalize(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class FeatureDomain(BaseModel):
    """Species, entity, identifier, and optional assembly compatibility boundary."""

    model_config = ConfigDict(frozen=True)

    species_taxon_id: int = Field(gt=0)
    feature_type: GenomicFeatureType
    namespace: FeatureNamespace
    genome_assembly: str | None = None

    @model_validator(mode="after")
    def validate_assembly(self) -> Self:
        """Enforce the shared entity, identifier, and assembly boundary."""
        validate_feature_domain_compatibility(
            feature_type=self.feature_type,
            namespace=self.namespace,
            genome_assembly=self.genome_assembly,
        )
        return self

    @property
    def domain_hash(self) -> str:
        """Return a stable fingerprint for the complete feature domain."""
        return canonical_sha256(_feature_domain_payload(self))


class ResourceSnapshot(BaseModel):
    """One immutable retrieval of a named external biological resource."""

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    resource_release: str = Field(min_length=1)
    retrieved_at: datetime
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_uri: str | None = None
    license_id: str | None = None
    citation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        """Require a resolved release, aware retrieval time, and unique citations."""
        _require_clean_text(self.provider_id, "provider_id")
        _require_clean_text(self.resource_id, "resource_id")
        _require_resolved_release(self.resource_release, "resource_release")
        _require_aware_datetime(self.retrieved_at, "retrieved_at")
        if self.source_uri is not None:
            _require_clean_text(self.source_uri, "source_uri")
            _reject_secret_uri(self.source_uri)
        if self.license_id is not None:
            _require_clean_text(self.license_id, "license_id")
        _require_unique_clean_strings(self.citation_ids, "citation_ids")
        return self

    @property
    def snapshot_id(self) -> str:
        """Return the core-computed hash of this exact resource snapshot."""
        return canonical_sha256(_resource_snapshot_payload(self))


class FeatureCollection(BaseModel):
    """A versioned, domain-specific set of unique feature identifiers."""

    model_config = ConfigDict(frozen=True)

    collection_id: str = Field(min_length=1)
    domain: FeatureDomain
    feature_ids: tuple[str, ...]
    source_snapshot_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_collection(self) -> Self:
        """Require a nonempty set without duplicate or ambiguous identifiers."""
        validated_domain = FeatureDomain.model_validate(self.domain.model_dump(mode="python"))
        object.__setattr__(self, "domain", validated_domain)
        _require_clean_text(self.collection_id, "collection_id")
        _require_clean_text(self.source_snapshot_id, "source_snapshot_id")
        if not self.feature_ids:
            raise ValueError("feature_ids must be nonempty")
        _require_unique_clean_strings(self.feature_ids, "feature_ids")
        return self

    @property
    def content_hash(self) -> str:
        """Hash collection content independently of feature input order."""
        return build_feature_collection_hash(
            collection_id=self.collection_id,
            domain=self.domain,
            feature_ids=self.feature_ids,
            source_snapshot_id=self.source_snapshot_id,
        )


class QueryProvenance(BaseModel):
    """Complete, secret-free provenance for one external-provider result."""

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    resources: tuple[ResourceSnapshot, ...]
    domain: FeatureDomain
    retrieved_at: datetime
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_hash: str = ""
    response_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query_parameters: Mapping[str, object] = Field(default_factory=dict)
    software_versions: Mapping[str, str] = Field(default_factory=dict)
    complete: bool = True
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_and_hash_query(self) -> Self:
        """Validate query identity and install the canonical core-computed hash."""
        validated_resources = tuple(
            ResourceSnapshot.model_validate(item.model_dump(mode="python"))
            for item in self.resources
        )
        validated_domain = FeatureDomain.model_validate(self.domain.model_dump(mode="python"))
        object.__setattr__(self, "resources", validated_resources)
        object.__setattr__(self, "domain", validated_domain)
        _require_clean_text(self.provider_id, "provider_id")
        _require_resolved_release(self.provider_version, "provider_version")
        _require_aware_datetime(self.retrieved_at, "retrieved_at")
        if not self.resources:
            raise ValueError("resources must be nonempty")
        snapshot_ids = tuple(item.snapshot_id for item in self.resources)
        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise ValueError("resources must contain unique snapshots")
        _reject_secret_parameter_keys(self.query_parameters)
        for key, value in self.software_versions.items():
            _require_clean_text(key, "software_versions key")
            _require_clean_text(value, f"software version for {key}")
        _require_unique_clean_strings(self.warnings, "warnings")
        frozen_parameters = _cast_mapping(_deep_freeze(self.query_parameters))
        frozen_versions = _cast_string_mapping(_deep_freeze(self.software_versions))
        object.__setattr__(self, "query_parameters", frozen_parameters)
        object.__setattr__(self, "software_versions", frozen_versions)
        expected = build_query_hash(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            resources=self.resources,
            domain=self.domain,
            input_hash=self.input_hash,
            query_parameters=self.query_parameters,
            software_versions=self.software_versions,
        )
        if self.query_hash:
            if _SHA256_PATTERN.fullmatch(self.query_hash) is None:
                raise ValueError("query_hash must be a lowercase SHA-256 digest")
            if self.query_hash != expected:
                raise ValueError("query_hash does not match the canonical query payload")
        object.__setattr__(self, "query_hash", expected)
        return self

    @field_serializer("query_parameters", "software_versions")
    def serialize_frozen_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        """Serialize recursively frozen mappings as ordinary JSON-compatible data."""
        return cast(dict[str, object], _deep_thaw(value))


def build_feature_collection_hash(
    *,
    collection_id: str,
    domain: FeatureDomain,
    feature_ids: Sequence[str],
    source_snapshot_id: str,
) -> str:
    """Build an order-independent hash for one feature collection."""
    return canonical_sha256(
        {
            "collection_id": collection_id,
            "domain": _feature_domain_payload(domain),
            "feature_ids": sorted(feature_ids),
            "source_snapshot_id": source_snapshot_id,
        }
    )


def build_query_hash(
    *,
    provider_id: str,
    provider_version: str,
    resources: Sequence[ResourceSnapshot],
    domain: FeatureDomain,
    input_hash: str,
    query_parameters: Mapping[str, object],
    software_versions: Mapping[str, str],
) -> str:
    """Build a resource-order-independent hash of scientific query identity."""
    return canonical_sha256(
        {
            "provider_id": provider_id,
            "provider_version": provider_version,
            "resources": sorted(item.snapshot_id for item in resources),
            "domain": _feature_domain_payload(domain),
            "input_hash": input_hash,
            "query_parameters": query_parameters,
            "software_versions": software_versions,
        }
    )


def _feature_domain_payload(domain: FeatureDomain) -> dict[str, object]:
    return {
        "species_taxon_id": domain.species_taxon_id,
        "feature_type": domain.feature_type.value,
        "namespace": domain.namespace.value,
        "genome_assembly": domain.genome_assembly,
    }


def _resource_snapshot_payload(snapshot: ResourceSnapshot) -> dict[str, object]:
    return {
        "provider_id": snapshot.provider_id,
        "resource_id": snapshot.resource_id,
        "resource_release": snapshot.resource_release,
        "retrieved_at": snapshot.retrieved_at,
        "response_sha256": snapshot.response_sha256,
        "source_uri": snapshot.source_uri,
        "license_id": snapshot.license_id,
        "citation_ids": sorted(snapshot.citation_ids),
    }


def _canonicalize(payload: object) -> object:
    if isinstance(payload, BaseModel):
        return _canonicalize(payload.model_dump(mode="python"))
    if isinstance(payload, datetime):
        _require_aware_datetime(payload, "canonical datetime")
        return payload.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(payload, Enum):
        return _canonicalize(payload.value)
    if isinstance(payload, Mapping):
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            normalized[key] = _canonicalize(value)
        return normalized
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_canonicalize(value) for value in payload]
    if payload is None or isinstance(payload, (str, bool, int)):
        return payload
    if isinstance(payload, float):
        if not isfinite(payload):
            raise ValueError("canonical payload floats must be finite")
        return payload
    raise TypeError(f"unsupported canonical payload type: {type(payload).__name__}")


def _deep_freeze(value: object) -> object:
    """Copy JSON-like values into recursively immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    """Copy recursively immutable query values into JSON-serializable containers."""
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_deep_thaw(item) for item in value]
    return value


def _cast_mapping(value: object) -> Mapping[str, object]:
    """Narrow a deeply frozen query-parameter mapping for strict typing."""
    if not isinstance(value, Mapping):
        raise TypeError("query parameters must be a mapping")
    return value


def _cast_string_mapping(value: object) -> Mapping[str, str]:
    """Narrow a deeply frozen software-version mapping for strict typing."""
    if not isinstance(value, Mapping):
        raise TypeError("software versions must be a mapping")
    if any(not isinstance(item, str) for item in value.values()):
        raise TypeError("software version values must be strings")
    return cast(Mapping[str, str], value)


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_resolved_release(value: str, field_name: str) -> None:
    _require_clean_text(value, field_name)
    if value.casefold() in _UNRESOLVED_RELEASES:
        raise ValueError(f"{field_name} must identify a resolved release, not {value!r}")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_unique_clean_strings(values: Sequence[str], field_name: str) -> None:
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")


def _reject_secret_parameter_keys(parameters: Mapping[str, object]) -> None:
    def walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                if not isinstance(raw_key, str):
                    raise TypeError("query parameter keys must be strings")
                _reject_secret_name(raw_key, ".".join((*path, raw_key)))
                walk(nested, (*path, raw_key))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested in enumerate(value):
                walk(nested, (*path, str(index)))

    walk(parameters, ())


def _reject_secret_uri(uri: str) -> None:
    """Reject credentials and secret-like query keys from persisted resource URIs."""
    parsed = urlsplit(uri)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_uri must not contain user credentials")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        _reject_secret_name(key, f"source_uri query parameter {key!r}")


def _reject_secret_name(name: str, location: str) -> None:
    """Reject common credential names after punctuation-insensitive normalization."""
    lowered = name.casefold()
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    segments = frozenset(filter(None, re.split(r"[^a-z0-9]+", lowered)))
    secret_compounds = (
        "accesskey",
        "accesstoken",
        "apikey",
        "apitoken",
        "authtoken",
        "clientsecret",
        "privatekey",
    )
    secret_segments = {
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
    }
    if (
        compact in _SECRET_KEY_COMPACT_FORMS
        or any(marker in compact for marker in secret_compounds)
        or bool(segments.intersection(secret_segments))
    ):
        raise ValueError(f"persisted provenance contains a secret-like key at {location}")
