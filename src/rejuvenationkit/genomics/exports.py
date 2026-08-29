"""Offline archive envelopes for provider-specific biological-data exports.

The helpers in this module do not contact external services.  They bind bytes
already retrieved by a caller to the release, license, pagination, request,
connector, and software metadata needed to reproduce a later import.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NotRequired, Self, TypedDict, Unpack, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rejuvenationkit.genomics.resources import (
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)

_ARCHIVE_FORMAT = "rejuvenationkit-provider-export/v2"
_SUPPORTED_ARCHIVE_FORMATS = frozenset({_ARCHIVE_FORMAT, "rejuvenationkit-provider-export/v1"})
_MANIFEST_MEMBER = "manifest.json"
_RAW_MEMBER = "raw-response.bin"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPDX_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+()\- :]*(?:[A-Za-z0-9.+)\-])?$")
_UNRESOLVED = frozenset({"latest", "current", "default", "unknown"})
_SECRET_COMPOUNDS = (
    "accesskey",
    "accesstoken",
    "apikey",
    "apitoken",
    "authtoken",
    "clientsecret",
    "privatekey",
)
_SECRET_SEGMENTS = frozenset(
    {
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
    }
)


class ExternalExportProvider(StrEnum):
    """Upstream source families with explicit archive helpers."""

    GENE_ONTOLOGY = "gene-ontology"
    STRING = "string-db"
    ENSEMBL = "ensembl"
    NCBI_SRA = "ncbi-sra"


class ExportLicensePolicy(StrEnum):
    """Whether an upstream license is declared or explicitly unknown."""

    SPDX = "spdx"
    UNKNOWN_WITH_WARNING = "unknown_with_warning"


class ExportLicense(BaseModel):
    """An upstream SPDX declaration or a fail-visible unknown-license policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: ExportLicensePolicy
    spdx_expression: str | None = None
    warning: str | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        """Require exactly the evidence appropriate to the selected policy."""
        if self.policy is ExportLicensePolicy.SPDX:
            if self.spdx_expression is None:
                raise ValueError("the SPDX license policy requires spdx_expression")
            _require_clean_text(self.spdx_expression, "spdx_expression")
            if self.spdx_expression.casefold() in {"unknown", "noassertion"}:
                raise ValueError("unknown licenses require policy='unknown_with_warning'")
            if _SPDX_EXPRESSION_RE.fullmatch(self.spdx_expression) is None:
                raise ValueError("spdx_expression contains invalid SPDX-expression characters")
            if self.warning is not None:
                raise ValueError("the SPDX license policy cannot carry an unknown-license warning")
        else:
            if self.spdx_expression is not None:
                raise ValueError("an unknown upstream license cannot claim an SPDX expression")
            if self.warning is None:
                raise ValueError("an unknown upstream license requires an explicit warning")
            _require_clean_text(self.warning, "license warning")
        return self

    @classmethod
    def spdx(cls, expression: str) -> ExportLicense:
        """Declare an SPDX expression reported for the upstream resource."""
        return cls(policy=ExportLicensePolicy.SPDX, spdx_expression=expression)

    @classmethod
    def unknown(cls, warning: str) -> ExportLicense:
        """Declare that the upstream license is unknown and retain why."""
        return cls(policy=ExportLicensePolicy.UNKNOWN_WITH_WARNING, warning=warning)


class PaginationMode(StrEnum):
    """How a provider response was divided during acquisition."""

    NONE = "none"
    PAGE_NUMBER = "page_number"
    OFFSET = "offset"
    CURSOR = "cursor"
    PROVIDER_EXPORT = "provider_export"


class ExportPageReceipt(BaseModel):
    """Hash-only receipt for one page; cursor values themselves are not retained."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    request_marker_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    provider_reported_more: bool


class PaginationAudit(BaseModel):
    """Complete page accounting for an already acquired provider export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: PaginationMode
    pages: tuple[ExportPageReceipt, ...]
    complete: bool
    truncated: bool = False
    provider_reported_total_items: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_pagination(self) -> Self:
        """Require contiguous pages and internally consistent completion claims."""
        if not self.pages:
            raise ValueError("pagination audit must contain at least one page receipt")
        sequences = tuple(page.sequence for page in self.pages)
        expected = tuple(range(1, len(self.pages) + 1))
        if sequences != expected:
            raise ValueError("page receipt sequences must be contiguous and ordered from one")
        if self.mode is PaginationMode.NONE and len(self.pages) != 1:
            raise ValueError("pagination mode 'none' requires exactly one response receipt")
        if self.complete and self.truncated:
            raise ValueError("a truncated provider response cannot be complete")
        if self.complete and self.pages[-1].provider_reported_more:
            raise ValueError("a complete export cannot end while the provider reports more pages")
        if any(not page.provider_reported_more for page in self.pages[:-1]):
            raise ValueError("every non-final page must report that another page follows")
        marker_hashes = tuple(page.request_marker_sha256 for page in self.pages)
        if len(set(marker_hashes)) != len(marker_hashes):
            raise ValueError("page request-marker hashes must be unique")
        observed = sum(page.item_count for page in self.pages)
        if (
            self.complete
            and self.provider_reported_total_items is not None
            and observed != self.provider_reported_total_items
        ):
            raise ValueError("complete pagination item count does not match provider total")
        if (
            self.provider_reported_total_items is not None
            and observed > self.provider_reported_total_items
        ):
            raise ValueError("observed pagination items cannot exceed provider total")
        return self

    @property
    def item_count(self) -> int:
        """Return the number of archived provider items across all pages."""
        return sum(page.item_count for page in self.pages)


class ProviderExportEnvelope(BaseModel):
    """Integrity-bound metadata for raw bytes acquired outside RejuvenationKit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ExternalExportProvider
    resource_id: str = Field(min_length=1)
    resource_release: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    retrieved_at: datetime
    license: ExportLicense
    connector_id: str = Field(min_length=1)
    connector_version: str = Field(min_length=1)
    software_versions: Mapping[str, str] = Field(default_factory=dict)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_parameters: Mapping[str, object] = Field(default_factory=dict)
    request_hash: str = ""
    raw_media_type: str = Field(min_length=1)
    raw_byte_count: int = Field(ge=0)
    raw_bytes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pagination: PaginationAudit
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_and_bind(self) -> Self:
        """Validate metadata, reject secrets, freeze mappings, and bind the request."""
        validated_license = ExportLicense.model_validate(self.license.model_dump(mode="python"))
        validated_pagination = PaginationAudit.model_validate(
            self.pagination.model_dump(mode="python")
        )
        object.__setattr__(self, "license", validated_license)
        object.__setattr__(self, "pagination", validated_pagination)
        for field_name, value in (
            ("resource_id", self.resource_id),
            ("resource_release", self.resource_release),
            ("source_uri", self.source_uri),
            ("connector_id", self.connector_id),
            ("connector_version", self.connector_version),
            ("raw_media_type", self.raw_media_type),
        ):
            _require_clean_text(value, field_name)
        _require_resolved(self.resource_release, "resource_release")
        _require_resolved(self.connector_version, "connector_version")
        _require_aware_datetime(self.retrieved_at)
        _reject_secret_parameters(self.request_parameters)
        _reject_secret_uri(self.source_uri)
        for name, version in self.software_versions.items():
            _require_clean_text(name, "software name")
            _reject_secret_name(name, "software_versions")
            _require_resolved(version, f"software version for {name}")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("warnings must contain unique values")
        for warning in self.warnings:
            _require_clean_text(warning, "warning")
        required_warnings: set[str] = set()
        if self.license.policy is ExportLicensePolicy.UNKNOWN_WITH_WARNING:
            required_warnings.add("upstream_license_unknown")
        if not self.pagination.complete:
            required_warnings.add("provider_export_incomplete")
        if self.pagination.truncated:
            required_warnings.add("provider_export_truncated")
        missing_warnings = sorted(required_warnings.difference(self.warnings))
        if missing_warnings:
            raise ValueError(f"provider export is missing required warnings: {missing_warnings}")
        if self.pagination.mode is PaginationMode.NONE:
            receipt = self.pagination.pages[0]
            if (
                receipt.response_sha256 != self.raw_bytes_sha256
                or receipt.byte_count != self.raw_byte_count
            ):
                raise ValueError(
                    "a non-paginated receipt must identify the exact archived raw bytes"
                )
        elif sum(page.byte_count for page in self.pagination.pages) != self.raw_byte_count:
            raise ValueError(
                "paginated page byte counts must partition the archived concatenated bytes"
            )
        frozen_parameters = _freeze_mapping(self.request_parameters)
        frozen_versions = _freeze_string_mapping(self.software_versions)
        object.__setattr__(self, "request_parameters", frozen_parameters)
        object.__setattr__(self, "software_versions", frozen_versions)
        expected = _build_export_request_hash(
            provider=self.provider,
            resource_id=self.resource_id,
            resource_release=self.resource_release,
            input_hash=self.input_hash,
            request_parameters=self.request_parameters,
        )
        if self.request_hash:
            if _SHA256_RE.fullmatch(self.request_hash) is None:
                raise ValueError("request_hash must be a lowercase SHA-256 digest")
            if self.request_hash != expected:
                raise ValueError("request_hash does not match the canonical provider request")
        object.__setattr__(self, "request_hash", expected)
        return self

    @field_serializer("request_parameters", "software_versions")
    def serialize_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        """Serialize recursively frozen mappings as ordinary JSON-compatible data."""
        return cast(dict[str, object], _deep_thaw(value))

    @property
    def envelope_hash(self) -> str:
        """Return the deterministic identity of metadata and archived raw bytes."""
        return canonical_sha256(self.model_dump(mode="python"))

    def to_resource_snapshot(self) -> ResourceSnapshot:
        """Create the core snapshot corresponding to the archived raw bytes."""
        return ResourceSnapshot(
            provider_id=self.provider.value,
            resource_id=self.resource_id,
            resource_release=self.resource_release,
            retrieved_at=self.retrieved_at,
            response_sha256=self.raw_bytes_sha256,
            source_uri=self.source_uri,
            license_id=self.license.spdx_expression,
        )

    def to_query_provenance(self, domain: FeatureDomain) -> QueryProvenance:
        """Create query provenance without claiming a normalized result checksum."""
        software_versions = dict(self.software_versions)
        existing = software_versions.get(self.connector_id)
        if existing is not None and existing != self.connector_version:
            raise ValueError("connector version conflicts with software_versions")
        software_versions[self.connector_id] = self.connector_version
        return QueryProvenance(
            provider_id=self.connector_id,
            provider_version=self.connector_version,
            resources=(self.to_resource_snapshot(),),
            domain=domain,
            retrieved_at=self.retrieved_at,
            input_hash=self.input_hash,
            query_parameters=self.request_parameters,
            software_versions=software_versions,
            complete=self.pagination.complete,
            warnings=self.warnings,
        )


class ProviderExportMetadata(TypedDict):
    """Typed keyword metadata shared by the provider-specific builders."""

    resource_id: str
    resource_release: str
    source_uri: str
    retrieved_at: datetime
    license: ExportLicense
    connector_id: str
    connector_version: str
    input_hash: str
    raw_media_type: str
    pagination: PaginationAudit
    software_versions: NotRequired[Mapping[str, str]]
    request_parameters: NotRequired[Mapping[str, object]]
    warnings: NotRequired[tuple[str, ...]]


def single_response_pagination(
    raw_bytes: bytes,
    *,
    item_count: int,
    request_marker: object = None,
) -> PaginationAudit:
    """Build a complete non-paginated audit for one saved response."""
    return PaginationAudit(
        mode=PaginationMode.NONE,
        pages=(
            ExportPageReceipt(
                sequence=1,
                request_marker_sha256=canonical_sha256(request_marker),
                response_sha256=sha256(raw_bytes).hexdigest(),
                byte_count=len(raw_bytes),
                item_count=item_count,
                provider_reported_more=False,
            ),
        ),
        complete=True,
        provider_reported_total_items=item_count,
    )


def build_go_export(
    raw_bytes: bytes, **metadata: Unpack[ProviderExportMetadata]
) -> ProviderExportEnvelope:
    """Bind one response or ordered page-byte concatenation to exact metadata."""
    return _build_provider_export(ExternalExportProvider.GENE_ONTOLOGY, raw_bytes, metadata)


def build_string_export(
    raw_bytes: bytes, **metadata: Unpack[ProviderExportMetadata]
) -> ProviderExportEnvelope:
    """Bind one response or ordered page-byte concatenation to exact metadata."""
    return _build_provider_export(ExternalExportProvider.STRING, raw_bytes, metadata)


def build_ensembl_export(
    raw_bytes: bytes, **metadata: Unpack[ProviderExportMetadata]
) -> ProviderExportEnvelope:
    """Bind one response or ordered page-byte concatenation to exact metadata."""
    return _build_provider_export(ExternalExportProvider.ENSEMBL, raw_bytes, metadata)


def build_sra_export(
    raw_bytes: bytes, **metadata: Unpack[ProviderExportMetadata]
) -> ProviderExportEnvelope:
    """Bind one response or ordered page-byte concatenation to exact metadata."""
    return _build_provider_export(ExternalExportProvider.NCBI_SRA, raw_bytes, metadata)


def save_export_archive(
    path: str | os.PathLike[str],
    envelope: ProviderExportEnvelope,
    raw_bytes: bytes,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save one validated manifest-and-raw-bytes ZIP archive offline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _verify_raw_bytes(envelope, raw_bytes)
    manifest_payload = {
        "archive_format": _ARCHIVE_FORMAT,
        "envelope": envelope.model_dump(mode="json"),
        "envelope_hash": envelope.envelope_hash,
    }
    manifest_bytes = json.dumps(
        manifest_payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED) as archive:
            _write_zip_member(archive, _MANIFEST_MEMBER, manifest_bytes)
            _write_zip_member(archive, _RAW_MEMBER, raw_bytes)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(f"export archive already exists: {destination}") from exc
            temporary.unlink()
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def load_export_archive(
    path: str | os.PathLike[str],
    *,
    maximum_raw_bytes: int | None = None,
) -> tuple[ProviderExportEnvelope, bytes]:
    """Load and fully verify a saved provider export without any network access."""
    if maximum_raw_bytes is not None and maximum_raw_bytes < 0:
        raise ValueError("maximum_raw_bytes must be nonnegative")
    source = Path(path)
    with zipfile.ZipFile(source, mode="r") as archive:
        if sorted(archive.namelist()) != sorted((_MANIFEST_MEMBER, _RAW_MEMBER)):
            raise ValueError("provider export archive must contain exactly manifest and raw bytes")
        raw_info = archive.getinfo(_RAW_MEMBER)
        if maximum_raw_bytes is not None and raw_info.file_size > maximum_raw_bytes:
            raise ValueError("archived raw response exceeds maximum_raw_bytes")
        manifest_bytes = archive.read(_MANIFEST_MEMBER)
        raw_bytes = archive.read(_RAW_MEMBER)
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider export manifest is not valid UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("archive_format") not in _SUPPORTED_ARCHIVE_FORMATS
    ):
        raise ValueError("provider export archive format is missing or unsupported")
    try:
        envelope_payload = payload["envelope"]
        stored_envelope_hash = payload["envelope_hash"]
    except KeyError as exc:
        raise ValueError("provider export manifest is missing required fields") from exc
    envelope = ProviderExportEnvelope.model_validate(envelope_payload)
    if stored_envelope_hash != envelope.envelope_hash:
        raise ValueError("provider export manifest envelope hash does not match")
    _verify_raw_bytes(envelope, raw_bytes)
    return envelope, raw_bytes


def _build_provider_export(
    provider: ExternalExportProvider,
    raw_bytes: bytes,
    metadata: Mapping[str, object],
) -> ProviderExportEnvelope:
    values = dict(metadata)
    supplied_provider = values.pop("provider", provider)
    if supplied_provider != provider:
        raise ValueError(f"provider helper requires provider={provider.value!r}")
    warnings = list(cast(Sequence[str], values.pop("warnings", ())))
    license_declaration = values.get("license")
    pagination = values.get("pagination")
    if isinstance(license_declaration, ExportLicense):
        if license_declaration.policy is ExportLicensePolicy.UNKNOWN_WITH_WARNING:
            _append_once(warnings, "upstream_license_unknown")
    if isinstance(pagination, PaginationAudit):
        if not pagination.complete:
            _append_once(warnings, "provider_export_incomplete")
        if pagination.truncated:
            _append_once(warnings, "provider_export_truncated")
    values["warnings"] = tuple(warnings)
    values["provider"] = provider
    values["raw_byte_count"] = len(raw_bytes)
    values["raw_bytes_sha256"] = sha256(raw_bytes).hexdigest()
    envelope = ProviderExportEnvelope.model_validate(values)
    _verify_raw_bytes(envelope, raw_bytes)
    return envelope


def _build_export_request_hash(
    *,
    provider: ExternalExportProvider,
    resource_id: str,
    resource_release: str,
    input_hash: str,
    request_parameters: Mapping[str, object],
) -> str:
    return canonical_sha256(
        {
            "provider": provider.value,
            "resource_id": resource_id,
            "resource_release": resource_release,
            "input_hash": input_hash,
            "request_parameters": request_parameters,
        }
    )


def _verify_raw_bytes(envelope: ProviderExportEnvelope, raw_bytes: bytes) -> None:
    if len(raw_bytes) != envelope.raw_byte_count:
        raise ValueError("raw byte count does not match the provider export envelope")
    if sha256(raw_bytes).hexdigest() != envelope.raw_bytes_sha256:
        raise ValueError("raw byte SHA-256 does not match the provider export envelope")
    if envelope.pagination.mode is PaginationMode.NONE:
        return
    offset = 0
    for page in envelope.pagination.pages:
        page_bytes = raw_bytes[offset : offset + page.byte_count]
        if len(page_bytes) != page.byte_count:
            raise ValueError("paginated page byte counts exceed archived raw bytes")
        if sha256(page_bytes).hexdigest() != page.response_sha256:
            raise ValueError(
                f"archived bytes do not match pagination receipt for page {page.sequence}"
            )
        offset += page.byte_count
    if offset != len(raw_bytes):
        raise ValueError("pagination receipts do not account for all archived raw bytes")


def _write_zip_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_resolved(value: str, field_name: str) -> None:
    _require_clean_text(value, field_name)
    if value.casefold() in _UNRESOLVED:
        raise ValueError(f"{field_name} must identify an exact resolved version")


def _require_aware_datetime(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")


def _reject_secret_name(name: str, location: str) -> None:
    lowered = name.casefold()
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    segments = frozenset(filter(None, re.split(r"[^a-z0-9]+", lowered)))
    if any(marker in compact for marker in _SECRET_COMPOUNDS) or segments & _SECRET_SEGMENTS:
        raise ValueError(f"{location} contains a secret-like key: {name}")


def _reject_secret_parameters(parameters: Mapping[str, object]) -> None:
    def walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                if not isinstance(raw_key, str):
                    raise TypeError("request parameter keys must be strings")
                _reject_secret_name(raw_key, ".".join((*path, raw_key)))
                walk(nested, (*path, raw_key))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested in enumerate(value):
                walk(nested, (*path, str(index)))

    walk(parameters, ())


def _reject_secret_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_uri must not contain user credentials")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        _reject_secret_name(key, "source_uri query")


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("request parameter keys must be strings")
            frozen[key] = _deep_freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_deep_thaw(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("request parameters must be a mapping")
    return frozen


def _freeze_string_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping) or any(
        not isinstance(item, str) for item in frozen.values()
    ):
        raise TypeError("software versions must map names to strings")
    return cast(Mapping[str, str], frozen)


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
