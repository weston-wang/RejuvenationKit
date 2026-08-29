"""Versioned, ambiguity-reporting cross-species signature translation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from enum import StrEnum
from math import isclose, isfinite
from typing import Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType
from rejuvenationkit.genomics.signatures import (
    GeneSignature,
    SignatureFeature,
)


class OrthologAmbiguityPolicy(StrEnum):
    """Handling of one-to-many source-to-target ortholog relationships."""

    ERROR = "error"
    DROP_AMBIGUOUS = "drop_ambiguous"
    SELECT_HIGHEST_CONFIDENCE = "select_highest_confidence"
    DISTRIBUTE_WEIGHT = "distribute_weight"


class OrthologRecord(BaseModel):
    """One versioned source-to-target ortholog assertion."""

    model_config = ConfigDict(frozen=True)

    source_feature_id: str = Field(min_length=1)
    target_feature_id: str = Field(min_length=1)
    confidence: float | None = None
    relationship: str = Field(default="ortholog", min_length=1)
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        """Reject invalid optional confidence and canonicalize evidence identifiers."""
        if self.confidence is not None and not isfinite(self.confidence):
            raise ValueError("ortholog confidence must be finite")
        for field_name, value in (
            ("source_feature_id", self.source_feature_id),
            ("target_feature_id", self.target_feature_id),
            ("relationship", self.relationship),
        ):
            if value != value.strip():
                raise ValueError(f"{field_name} cannot contain surrounding whitespace")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("ortholog evidence_ids must be unique")
        if any(not value or value != value.strip() for value in self.evidence_ids):
            raise ValueError("ortholog evidence_ids must be nonempty without padding")
        object.__setattr__(self, "evidence_ids", tuple(sorted(self.evidence_ids)))
        return self


class OrthologMap(BaseModel):
    """Local mapping bound to exact biological domains and provider provenance.

    Legacy species/namespace fields remain available, but are validated against
    the source and target :class:`FeatureDomain` objects. A manually authored map
    without an external snapshot may omit the source query and query provenance;
    translating a signature requires the fully provenance-bound form.
    """

    model_config = ConfigDict(frozen=True)

    source_species_taxon_id: int = Field(gt=0)
    target_species_taxon_id: int = Field(gt=0)
    source_namespace: FeatureNamespace
    target_namespace: FeatureNamespace
    source_domain: FeatureDomain
    target_domain: FeatureDomain
    resource_id: str = Field(min_length=1)
    resource_version: str = Field(min_length=1)
    records: tuple[OrthologRecord, ...]
    resource_snapshot: ResourceSnapshot | None = None
    source_query: FeatureCollection | None = None
    query_provenance: QueryProvenance | None = None
    confidence_definition: str | None = None
    confidence_scale: tuple[float, float] | None = None
    normalized_import_checksum: str = ""

    @model_validator(mode="before")
    @classmethod
    def populate_compatible_domains(cls, data: Any) -> Any:
        """Populate exact gene domains for legacy species/namespace constructors."""
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for prefix in ("source", "target"):
            domain_key = f"{prefix}_domain"
            species_key = f"{prefix}_species_taxon_id"
            namespace_key = f"{prefix}_namespace"
            raw_domain = values.get(domain_key)
            if raw_domain is None:
                if species_key in values and namespace_key in values:
                    values[domain_key] = {
                        "species_taxon_id": values[species_key],
                        "feature_type": GenomicFeatureType.GENE,
                        "namespace": values[namespace_key],
                    }
            else:
                domain = FeatureDomain.model_validate(raw_domain)
                values.setdefault(species_key, domain.species_taxon_id)
                values.setdefault(namespace_key, domain.namespace)
        return values

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        """Require exact domains, resource binding, and canonical imported content."""
        if self.source_species_taxon_id == self.target_species_taxon_id:
            raise ValueError("ortholog map source and target species must differ")
        if self.source_domain.feature_type is not GenomicFeatureType.GENE:
            raise ValueError("ortholog source domain must contain gene features")
        if self.target_domain.feature_type is not GenomicFeatureType.GENE:
            raise ValueError("ortholog target domain must contain gene features")
        if (
            self.source_domain.species_taxon_id != self.source_species_taxon_id
            or self.source_domain.namespace is not self.source_namespace
        ):
            raise ValueError("ortholog source domain must match legacy source fields")
        if (
            self.target_domain.species_taxon_id != self.target_species_taxon_id
            or self.target_domain.namespace is not self.target_namespace
        ):
            raise ValueError("ortholog target domain must match legacy target fields")
        if not self.records:
            raise ValueError("ortholog map must contain records")
        canonical_records = tuple(sorted(self.records, key=_ortholog_record_sort_key))
        pairs = [(item.source_feature_id, item.target_feature_id) for item in canonical_records]
        if len(set(pairs)) != len(pairs):
            raise ValueError("ortholog source-target pairs must be unique")
        object.__setattr__(self, "records", canonical_records)
        if (self.confidence_definition is None) != (self.confidence_scale is None):
            raise ValueError("confidence definition and scale must be supplied together")
        if self.confidence_definition is not None and (
            not self.confidence_definition.strip()
            or self.confidence_definition != self.confidence_definition.strip()
        ):
            raise ValueError("confidence_definition must be nonempty without padding")
        if self.confidence_scale is not None:
            lower, upper = self.confidence_scale
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError("confidence_scale must contain finite increasing bounds")
            out_of_scale = [
                item.confidence
                for item in canonical_records
                if item.confidence is not None and not lower <= item.confidence <= upper
            ]
            if out_of_scale:
                raise ValueError("ortholog confidence lies outside confidence_scale")
        if self.resource_snapshot is not None and (
            self.resource_snapshot.resource_id != self.resource_id
            or self.resource_snapshot.resource_release != self.resource_version
        ):
            raise ValueError("ortholog resource ID/version must match its resource snapshot")
        external_bindings = (
            self.resource_snapshot,
            self.source_query,
            self.query_provenance,
        )
        if any(item is None for item in external_bindings) and not all(
            item is None for item in external_bindings
        ):
            raise ValueError(
                "resource snapshot, source query, and query provenance must be supplied together"
            )
        if (
            self.resource_snapshot is not None
            and self.source_query is not None
            and self.query_provenance is not None
        ):
            if self.source_query.domain != self.source_domain:
                raise ValueError("ortholog source query must match the exact source domain")
            if self.query_provenance.domain != self.source_domain:
                raise ValueError("ortholog query provenance must match the exact source domain")
            if self.resource_snapshot not in self.query_provenance.resources:
                raise ValueError("ortholog snapshot must be present in query provenance resources")
            if self.query_provenance.input_hash != self.source_query.content_hash:
                raise ValueError("ortholog provenance input_hash must match the exact source query")
            unexpected_source_ids = sorted(
                {item.source_feature_id for item in canonical_records}.difference(
                    self.source_query.feature_ids
                )
            )
            if unexpected_source_ids:
                raise ValueError(
                    "ortholog response contains source features outside the exact query: "
                    f"{unexpected_source_ids}"
                )

        normalized_checksum = _normalized_ortholog_checksum(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            resource_id=self.resource_id,
            resource_version=self.resource_version,
            records=canonical_records,
            confidence_definition=self.confidence_definition,
            confidence_scale=self.confidence_scale,
        )
        if self.normalized_import_checksum and (
            self.normalized_import_checksum != normalized_checksum
        ):
            raise ValueError("normalized_import_checksum does not match ortholog content")
        object.__setattr__(self, "normalized_import_checksum", normalized_checksum)
        if self.query_provenance is not None:
            if self.query_provenance.response_checksum not in (None, normalized_checksum):
                raise ValueError(
                    "query provenance response checksum does not match normalized import"
                )
            if self.query_provenance.response_checksum is None:
                object.__setattr__(
                    self,
                    "query_provenance",
                    self.query_provenance.model_copy(
                        update={"response_checksum": normalized_checksum}
                    ),
                )
        return self

    @property
    def content_hash(self) -> str:
        """Return the canonical checksum of normalized ortholog assertions."""
        return self.normalized_import_checksum


class OrthologTargetCollision(BaseModel):
    """Multiple selected source features contributing to one target feature."""

    model_config = ConfigDict(frozen=True)

    target_feature_id: str = Field(min_length=1)
    source_feature_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """Require at least two unique, canonically ordered source identifiers."""
        if len(self.source_feature_ids) < 2:
            raise ValueError("target collision requires at least two source features")
        if len(set(self.source_feature_ids)) != len(self.source_feature_ids):
            raise ValueError("target collision source features must be unique")
        if self.source_feature_ids != tuple(sorted(self.source_feature_ids)):
            raise ValueError("target collision source features must be canonically ordered")
        return self


class OrthologTranslation(BaseModel):
    """Context-only translated signature and complete mapping audit.

    Ortholog database assertions are not subject-level efficacy estimates and this
    object intentionally exposes no conversion to ``EvidenceEstimate``.
    """

    model_config = ConfigDict(frozen=True)

    signature: GeneSignature
    source_signature_id: str
    mapping_resource_id: str
    mapping_resource_version: str
    mapping_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_domain: FeatureDomain
    target_domain: FeatureDomain
    policy: OrthologAmbiguityPolicy
    minimum_confidence: float | None = None
    confidence_definition: str | None = None
    confidence_scale: tuple[float, float] | None = None
    allowed_relationships: tuple[str, ...]
    mapped_source_feature_ids: tuple[str, ...]
    missing_source_feature_ids: tuple[str, ...]
    ambiguous_source_feature_ids: tuple[str, ...]
    dropped_source_feature_ids: tuple[str, ...]
    selected_pairs: tuple[tuple[str, str], ...]
    target_collisions: tuple[OrthologTargetCollision, ...]
    source_absolute_weight: float = Field(gt=0)
    retained_absolute_weight: float = Field(ge=0)
    retained_weight_fraction: float = Field(ge=0, le=1)
    translated_absolute_weight: float = Field(ge=0)
    translated_weight_fraction: float = Field(ge=0)
    post_aggregation_absolute_weight: float = Field(ge=0)
    post_aggregation_weight_fraction: float = Field(ge=0)
    warnings: tuple[str, ...] = ()
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"

    @model_validator(mode="after")
    def validate_translation_audit(self) -> Self:
        """Require internally consistent domains, provenance, collisions, and weights."""
        if (
            self.signature.species_taxon_id != self.target_domain.species_taxon_id
            or self.signature.namespace is not self.target_domain.namespace
        ):
            raise ValueError("translated signature must match the exact target domain")
        for marker, value in (
            ("ortholog_snapshot", self.mapping_snapshot_id),
            ("ortholog_query", self.mapping_query_hash),
            ("ortholog_content", self.mapping_content_hash),
        ):
            if f"{marker}={value}" not in self.signature.resource_id:
                raise ValueError("translated signature resource provenance is incomplete")
        expected_retained_fraction = self.retained_absolute_weight / self.source_absolute_weight
        expected_translated_fraction = self.translated_absolute_weight / self.source_absolute_weight
        if not isclose(
            self.retained_weight_fraction,
            expected_retained_fraction,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("retained weight fraction does not match absolute weights")
        if not isclose(
            self.translated_weight_fraction,
            expected_translated_fraction,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("translated weight fraction does not match absolute weights")
        if not isclose(
            self.post_aggregation_absolute_weight,
            self.translated_absolute_weight,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or not isclose(
            self.post_aggregation_weight_fraction,
            self.translated_weight_fraction,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("post-aggregation weight aliases must match translated weight")
        if self.translated_absolute_weight > self.retained_absolute_weight and not isclose(
            self.translated_absolute_weight,
            self.retained_absolute_weight,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("ortholog translation cannot amplify retained absolute weight")
        pair_sources: defaultdict[str, set[str]] = defaultdict(set)
        for source_id, target_id in self.selected_pairs:
            pair_sources[target_id].add(source_id)
        expected_collisions = tuple(
            OrthologTargetCollision(
                target_feature_id=target_id,
                source_feature_ids=tuple(sorted(source_ids)),
            )
            for target_id, source_ids in sorted(pair_sources.items())
            if len(source_ids) > 1
        )
        if self.target_collisions != expected_collisions:
            raise ValueError("target collision audit does not match selected ortholog pairs")
        if self.target_collisions and (
            "multiple_source_features_share_target_ortholog" not in self.warnings
        ):
            raise ValueError("target collisions must be reported in warnings")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("ortholog translation warnings must be unique")
        return self


def _ortholog_record_sort_key(
    record: OrthologRecord,
) -> tuple[str, str, str, float, tuple[str, ...]]:
    """Return a deterministic key that places unavailable confidence last."""
    confidence_key = record.confidence if record.confidence is not None else float("inf")
    return (
        record.source_feature_id,
        record.target_feature_id,
        record.relationship,
        confidence_key,
        record.evidence_ids,
    )


def _normalized_ortholog_checksum(
    *,
    source_domain: FeatureDomain,
    target_domain: FeatureDomain,
    resource_id: str,
    resource_version: str,
    records: tuple[OrthologRecord, ...],
    confidence_definition: str | None,
    confidence_scale: tuple[float, float] | None,
) -> str:
    """Hash canonical imported assertions independently of raw provider bytes."""
    return canonical_sha256(
        {
            "schema": "rejuvenationkit.ortholog-import/v1",
            "source_domain": source_domain.model_dump(mode="python"),
            "target_domain": target_domain.model_dump(mode="python"),
            "resource_id": resource_id,
            "resource_version": resource_version,
            "records": [item.model_dump(mode="python") for item in records],
            "confidence_definition": confidence_definition,
            "confidence_scale": confidence_scale,
        }
    )


def _resolve_ortholog_domain(
    *,
    domain: FeatureDomain | None,
    species_taxon_id: int | None,
    namespace: FeatureNamespace | None,
    label: str,
) -> FeatureDomain:
    """Resolve new domain inputs while safely accepting complete legacy inputs."""
    if domain is None:
        if species_taxon_id is None or namespace is None:
            raise ValueError(
                f"{label}_domain or both legacy {label} species/namespace are required"
            )
        return FeatureDomain(
            species_taxon_id=species_taxon_id,
            feature_type=GenomicFeatureType.GENE,
            namespace=namespace,
        )
    if domain.feature_type is not GenomicFeatureType.GENE:
        raise ValueError(f"ortholog {label} domain must contain gene features")
    if species_taxon_id is not None and species_taxon_id != domain.species_taxon_id:
        raise ValueError(f"legacy {label} species does not match {label}_domain")
    if namespace is not None and namespace is not domain.namespace:
        raise ValueError(f"legacy {label} namespace does not match {label}_domain")
    return domain


def _validate_confidence_operation(mapping: OrthologMap, *, operation: str) -> None:
    """Fail closed when confidence-dependent decisions lack declared semantics."""
    if mapping.confidence_definition is None or mapping.confidence_scale is None:
        raise ValueError(f"{operation} requires an explicit confidence definition and scale")


def read_ortholog_map(
    frame: pd.DataFrame,
    *,
    source_feature_id_column: str,
    target_feature_id_column: str,
    resource: ResourceSnapshot,
    source_query: FeatureCollection,
    provenance: QueryProvenance,
    source_domain: FeatureDomain | None = None,
    target_domain: FeatureDomain | None = None,
    source_species_taxon_id: int | None = None,
    target_species_taxon_id: int | None = None,
    source_namespace: FeatureNamespace | None = None,
    target_namespace: FeatureNamespace | None = None,
    confidence_column: str | None = None,
    confidence_definition: str | None = None,
    confidence_scale: tuple[float, float] | None = None,
    relationship_column: str | None = None,
    evidence_ids_column: str | None = None,
    evidence_separator: str = ";",
) -> OrthologMap:
    """Read an archived ortholog table with exact domains and query provenance.

    ``ResourceSnapshot.response_sha256`` remains the checksum of the raw provider
    response. The returned map separately computes a canonical checksum over the
    normalized imported assertions and binds it to query provenance.
    """
    resolved_source_domain = _resolve_ortholog_domain(
        domain=source_domain,
        species_taxon_id=source_species_taxon_id,
        namespace=source_namespace,
        label="source",
    )
    resolved_target_domain = _resolve_ortholog_domain(
        domain=target_domain,
        species_taxon_id=target_species_taxon_id,
        namespace=target_namespace,
        label="target",
    )
    if provenance.domain != resolved_source_domain:
        raise ValueError("ortholog query provenance must match the exact source domain")
    if resource not in provenance.resources:
        raise ValueError("ortholog resource must be present in query provenance resources")
    if source_query.domain != resolved_source_domain:
        raise ValueError("ortholog source query must match the exact source domain")
    if provenance.input_hash != source_query.content_hash:
        raise ValueError("ortholog provenance input_hash must match the exact source query")
    if confidence_column is not None and (
        confidence_definition is None or confidence_scale is None
    ):
        raise ValueError(
            "confidence_definition and confidence_scale are required with confidence_column"
        )
    if confidence_column is None and (
        confidence_definition is not None or confidence_scale is not None
    ):
        raise ValueError("confidence metadata cannot be supplied without confidence_column")
    required = {source_feature_id_column, target_feature_id_column}
    optional = {
        value
        for value in (confidence_column, relationship_column, evidence_ids_column)
        if value is not None
    }
    missing = sorted((required | optional).difference(frame.columns))
    if missing:
        raise ValueError(f"ortholog-map columns are absent: {missing}")
    if frame.empty:
        raise ValueError("ortholog-map table is empty")
    records: list[OrthologRecord] = []
    for _, row in frame.iterrows():
        source_id = _required_table_string(row, source_feature_id_column)
        target_id = _required_table_string(row, target_feature_id_column)
        relationship = (
            _required_table_string(row, relationship_column)
            if relationship_column is not None
            else "ortholog"
        )
        evidence_ids = (
            tuple(
                sorted(
                    {
                        item.strip()
                        for item in _required_table_string(row, evidence_ids_column).split(
                            evidence_separator
                        )
                        if item.strip()
                    }
                )
            )
            if evidence_ids_column is not None and not pd.isna(row[evidence_ids_column])
            else ()
        )
        records.append(
            OrthologRecord(
                source_feature_id=source_id,
                target_feature_id=target_id,
                confidence=(
                    None
                    if confidence_column is None or pd.isna(row[confidence_column])
                    else float(row[confidence_column])
                ),
                relationship=relationship,
                evidence_ids=evidence_ids,
            )
        )
    records.sort(key=lambda item: (item.source_feature_id, item.target_feature_id))
    return OrthologMap(
        source_species_taxon_id=resolved_source_domain.species_taxon_id,
        target_species_taxon_id=resolved_target_domain.species_taxon_id,
        source_namespace=resolved_source_domain.namespace,
        target_namespace=resolved_target_domain.namespace,
        source_domain=resolved_source_domain,
        target_domain=resolved_target_domain,
        resource_id=resource.resource_id,
        resource_version=resource.resource_release,
        records=tuple(records),
        resource_snapshot=resource,
        source_query=source_query,
        query_provenance=provenance,
        confidence_definition=confidence_definition,
        confidence_scale=confidence_scale,
    )


def _required_table_string(row: pd.Series, column: str) -> str:
    if pd.isna(row[column]):
        raise ValueError(f"ortholog-map value in {column!r} cannot be missing")
    value = str(row[column]).strip()
    if not value:
        raise ValueError(f"ortholog-map value in {column!r} cannot be empty")
    return value


def translate_signature(
    signature: GeneSignature,
    mapping: OrthologMap,
    *,
    policy: OrthologAmbiguityPolicy = OrthologAmbiguityPolicy.ERROR,
    minimum_confidence: float | None = None,
    minimum_retained_weight_fraction: float = 0.7,
    target_tissue: str | None = None,
    allowed_relationships: tuple[str, ...] = ("ortholog",),
) -> OrthologTranslation:
    """Translate feature weights as contextual, fully provenance-bound metadata."""
    if (
        mapping.resource_snapshot is None
        or mapping.source_query is None
        or mapping.query_provenance is None
    ):
        raise ValueError("ortholog translation requires snapshot-bound query provenance")
    if minimum_confidence is not None and not isfinite(minimum_confidence):
        raise ValueError("minimum_confidence must be finite")
    if minimum_confidence is not None:
        _validate_confidence_operation(mapping, operation="confidence filtering")
        if mapping.confidence_scale is None:  # pragma: no cover - guarded above
            raise RuntimeError("validated confidence scale is unexpectedly absent")
        lower, upper = mapping.confidence_scale
        if not lower <= minimum_confidence <= upper:
            raise ValueError("minimum_confidence must lie within confidence_scale")
    if not 0 < minimum_retained_weight_fraction <= 1:
        raise ValueError("minimum_retained_weight_fraction must lie in (0, 1]")
    if not allowed_relationships or len(set(allowed_relationships)) != len(allowed_relationships):
        raise ValueError("allowed_relationships must be nonempty and unique")
    if signature.species_taxon_id != mapping.source_species_taxon_id:
        raise ValueError("signature species does not match ortholog-map source species")
    if signature.namespace is not mapping.source_namespace:
        raise ValueError("signature namespace does not match ortholog-map source namespace")
    unqueried_signature_features = sorted(
        {item.feature_id for item in signature.features}.difference(
            mapping.source_query.feature_ids
        )
    )
    if unqueried_signature_features:
        raise ValueError(
            "signature contains source features outside the exact ortholog query: "
            f"{unqueried_signature_features}"
        )
    if target_tissue is not None and target_tissue != signature.tissue:
        raise ValueError(
            "target_tissue cannot relabel a cross-species signature; source tissue is preserved"
        )

    by_source: defaultdict[str, list[OrthologRecord]] = defaultdict(list)
    relationships_excluded = False
    confidence_filtered = False
    signature_feature_ids = {item.feature_id for item in signature.features}
    for record in mapping.records:
        if record.source_feature_id not in signature_feature_ids:
            continue
        if record.relationship not in allowed_relationships:
            relationships_excluded = True
            continue
        if minimum_confidence is not None:
            if record.confidence is None:
                raise ValueError(
                    "confidence filtering requires confidence for every candidate record"
                )
            if record.confidence < minimum_confidence:
                confidence_filtered = True
                continue
        by_source[record.source_feature_id].append(record)
    translated_weights: defaultdict[str, float] = defaultdict(float)
    mapped: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    dropped: list[str] = []
    selected_pairs: list[tuple[str, str]] = []
    retained_source_weight = 0.0
    for feature in signature.features:
        candidates = sorted(
            by_source.get(feature.feature_id, ()),
            key=lambda item: item.target_feature_id,
        )
        if not candidates:
            missing.append(feature.feature_id)
            dropped.append(feature.feature_id)
            continue
        if len(candidates) == 1:
            selected = candidates
        else:
            ambiguous.append(feature.feature_id)
            if policy is OrthologAmbiguityPolicy.ERROR:
                targets = [item.target_feature_id for item in candidates]
                raise ValueError(f"ambiguous ortholog mapping for {feature.feature_id}: {targets}")
            if policy is OrthologAmbiguityPolicy.DROP_AMBIGUOUS:
                dropped.append(feature.feature_id)
                continue
            if policy is OrthologAmbiguityPolicy.SELECT_HIGHEST_CONFIDENCE:
                _validate_confidence_operation(
                    mapping,
                    operation="highest-confidence resolution",
                )
                if any(item.confidence is None for item in candidates):
                    raise ValueError(
                        "highest-confidence resolution requires confidence for every candidate"
                    )
                highest = max(item.confidence for item in candidates if item.confidence is not None)
                highest_candidates = [item for item in candidates if item.confidence == highest]
                if len(highest_candidates) != 1:
                    targets = [item.target_feature_id for item in highest_candidates]
                    raise ValueError(
                        f"highest-confidence ortholog tie for {feature.feature_id}: {targets}"
                    )
                selected = highest_candidates
            else:
                selected = candidates
        mapped.append(feature.feature_id)
        retained_source_weight += abs(feature.weight)
        divided_weight = feature.weight / len(selected)
        for record in selected:
            translated_weights[record.target_feature_id] += divided_weight
            selected_pairs.append((feature.feature_id, record.target_feature_id))

    source_weight = sum(abs(item.weight) for item in signature.features)
    retained_fraction = retained_source_weight / source_weight
    if retained_fraction < minimum_retained_weight_fraction:
        raise ValueError(
            f"ortholog retained-weight fraction {retained_fraction:.3f} is below minimum "
            f"{minimum_retained_weight_fraction:.3f}"
        )
    translated_features = tuple(
        SignatureFeature(feature_id=feature_id, weight=weight)
        for feature_id, weight in sorted(translated_weights.items())
        if weight != 0
    )
    if not translated_features:
        raise ValueError("ortholog translation produced no nonzero target features")
    translated_absolute_weight = sum(abs(item.weight) for item in translated_features)
    translated_weight_fraction = translated_absolute_weight / source_weight
    target_sources: defaultdict[str, set[str]] = defaultdict(set)
    for source_id, target_id in selected_pairs:
        target_sources[target_id].add(source_id)
    target_collisions = tuple(
        OrthologTargetCollision(
            target_feature_id=target_id,
            source_feature_ids=tuple(sorted(source_ids)),
        )
        for target_id, source_ids in sorted(target_sources.items())
        if len(source_ids) > 1
    )
    warnings: list[str] = []
    if missing:
        warnings.append("ortholog_features_missing")
    if ambiguous:
        warnings.append("one_to_many_orthologs_present")
    if dropped:
        warnings.append("ortholog_features_dropped")
    if target_collisions:
        warnings.append("multiple_source_features_share_target_ortholog")
    if relationships_excluded:
        warnings.append("ortholog_relationships_excluded")
    if confidence_filtered:
        warnings.append("ortholog_records_below_confidence_threshold")
    if not mapping.query_provenance.complete:
        warnings.append("ortholog_query_incomplete")
    snapshot_id = mapping.resource_snapshot.snapshot_id
    query_hash = mapping.query_provenance.query_hash
    content_hash = mapping.content_hash
    translated = GeneSignature(
        signature_id=(
            f"{signature.signature_id}:{mapping.source_species_taxon_id}-to-"
            f"{mapping.target_species_taxon_id}"
        ),
        version=(f"{signature.version}+ortholog-{mapping.resource_id}-{mapping.resource_version}"),
        name=f"{signature.name} ({mapping.target_species_taxon_id} ortholog translation)",
        features=translated_features,
        namespace=mapping.target_namespace,
        species_taxon_id=mapping.target_species_taxon_id,
        tissue=signature.tissue,
        target_name=signature.target_name,
        target_unit=signature.target_unit,
        direction=signature.direction,
        resource_id=(
            f"{signature.resource_id};ortholog_snapshot={snapshot_id};"
            f"ortholog_query={query_hash};ortholog_content={content_hash}"
        ),
        allowed_scales=signature.allowed_scales,
    )
    return OrthologTranslation(
        signature=translated,
        source_signature_id=signature.signature_id,
        mapping_resource_id=mapping.resource_id,
        mapping_resource_version=mapping.resource_version,
        mapping_snapshot_id=snapshot_id,
        mapping_query_hash=query_hash,
        mapping_content_hash=content_hash,
        source_domain=mapping.source_domain,
        target_domain=mapping.target_domain,
        policy=policy,
        minimum_confidence=minimum_confidence,
        confidence_definition=mapping.confidence_definition,
        confidence_scale=mapping.confidence_scale,
        allowed_relationships=allowed_relationships,
        mapped_source_feature_ids=tuple(mapped),
        missing_source_feature_ids=tuple(missing),
        ambiguous_source_feature_ids=tuple(ambiguous),
        dropped_source_feature_ids=tuple(dropped),
        selected_pairs=tuple(selected_pairs),
        target_collisions=target_collisions,
        source_absolute_weight=source_weight,
        retained_absolute_weight=retained_source_weight,
        retained_weight_fraction=retained_fraction,
        translated_absolute_weight=translated_absolute_weight,
        translated_weight_fraction=translated_weight_fraction,
        post_aggregation_absolute_weight=translated_absolute_weight,
        post_aggregation_weight_fraction=translated_weight_fraction,
        warnings=tuple(warnings),
    )
