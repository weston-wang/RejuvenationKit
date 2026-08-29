"""Allele- and assembly-specific external variant annotations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Self, cast

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from rejuvenationkit.genomics.resources import (
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)
from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicFeatureType,
    GenomicMatrix,
    MatrixScale,
)


class VariantKey(BaseModel):
    """An exact split-allele key on one declared reference assembly.

    Allele case and surrounding whitespace are normalized. Equivalent indels are
    not minimized or left-aligned; callers must do that upstream against a
    versioned reference when equivalence matching is required.
    """

    model_config = ConfigDict(frozen=True)

    genome_assembly: str = Field(min_length=1)
    chromosome: str = Field(min_length=1)
    position_1_based: int = Field(gt=0)
    reference_allele: str = Field(min_length=1)
    alternate_allele: str = Field(min_length=1)

    @field_validator("genome_assembly", "chromosome", mode="before")
    @classmethod
    def normalize_coordinate_text(cls, value: object) -> str:
        """Normalize required coordinate labels without stringifying missing values."""
        return _required_scalar_text(value, "variant coordinate")

    @field_validator("position_1_based", mode="before")
    @classmethod
    def validate_position(cls, value: object) -> int:
        """Reject booleans, fractions, missing values, and lossy integer coercion."""
        return _positive_integral_value(value, "variant position")

    @field_validator("reference_allele", "alternate_allele", mode="before")
    @classmethod
    def normalize_allele(cls, value: object) -> str:
        """Normalize case and reject surrounding whitespace before strict matching."""
        return _required_scalar_text(value, "variant allele").upper()

    @model_validator(mode="after")
    def validate_alleles(self) -> Self:
        """Reject unsplit or identity alleles and normalize case."""
        if "," in self.alternate_allele:
            raise ValueError("variant keys must contain one split alternate allele")
        if self.reference_allele == self.alternate_allele:
            raise ValueError("reference and alternate alleles must differ")
        return self

    @property
    def canonical_id(self) -> str:
        """Return an assembly-aware identifier suitable for strict joins."""
        return (
            f"{self.genome_assembly}:{self.chromosome}:{self.position_1_based}:"
            f"{self.reference_allele}>{self.alternate_allele}"
        )


class VariantConsequence(BaseModel):
    """One transcript- or gene-specific consequence assertion."""

    model_config = ConfigDict(frozen=True)

    sequence_ontology_terms: tuple[str, ...]
    gene_id: str | None = None
    gene_namespace: FeatureNamespace | None = None
    transcript_id: str | None = None
    hgvs_c: str | None = None
    hgvs_p: str | None = None
    impact_label: str | None = None
    canonical_transcript: bool = False

    @model_validator(mode="after")
    def validate_terms_and_gene(self) -> Self:
        """Require unique terms and a namespace whenever a gene ID is present."""
        if not self.sequence_ontology_terms:
            raise ValueError("variant consequence must contain a Sequence Ontology term")
        if len(set(self.sequence_ontology_terms)) != len(self.sequence_ontology_terms):
            raise ValueError("Sequence Ontology terms must be unique")
        if (self.gene_id is None) != (self.gene_namespace is None):
            raise ValueError("gene_id and gene_namespace must be supplied together")
        return self


class VariantAnnotationRecord(BaseModel):
    """All descriptive annotations retained for one exact variant allele."""

    model_config = ConfigDict(frozen=True)

    variant: VariantKey
    consequences: tuple[VariantConsequence, ...]
    population_frequencies: Mapping[str, float] = Field(default_factory=dict)
    source_record_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        """Reject duplicates and invalid frequency claims."""
        if not self.consequences:
            raise ValueError("variant annotation record must contain consequences")
        consequence_keys = [item.model_dump_json() for item in self.consequences]
        if len(set(consequence_keys)) != len(consequence_keys):
            raise ValueError("variant consequences must be unique")
        if len(set(self.source_record_ids)) != len(self.source_record_ids):
            raise ValueError("variant source record identifiers must be unique")
        if any(
            not isfinite(value) or value < 0 or value > 1
            for value in self.population_frequencies.values()
        ):
            raise ValueError("population frequencies must be finite and lie in [0, 1]")
        for population in self.population_frequencies:
            _require_clean_text(population, "population frequency identifier")
        ordered_frequencies = MappingProxyType(dict(sorted(self.population_frequencies.items())))
        object.__setattr__(self, "population_frequencies", ordered_frequencies)
        object.__setattr__(
            self,
            "consequences",
            tuple(sorted(self.consequences, key=lambda item: item.model_dump_json())),
        )
        object.__setattr__(self, "source_record_ids", tuple(sorted(self.source_record_ids)))
        return self

    @field_serializer("population_frequencies")
    def serialize_frequencies(self, value: Mapping[str, float]) -> dict[str, float]:
        """Serialize the immutable frequency mapping deterministically."""
        return dict(value)


class VariantAnnotationRequest(BaseModel):
    """The exact allele set submitted for external annotation."""

    model_config = ConfigDict(frozen=True)

    species_taxon_id: int = Field(gt=0)
    variants: tuple[VariantKey, ...]
    resource: ResourceSnapshot
    normalization_policy: Literal["exact_split_alleles_no_equivalence_normalization"] = (
        "exact_split_alleles_no_equivalence_normalization"
    )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Require unique alleles on one assembly."""
        if not self.variants:
            raise ValueError("variant annotation request cannot be empty")
        identifiers = [item.canonical_id for item in self.variants]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("variant annotation request keys must be unique")
        if len({item.genome_assembly for item in self.variants}) != 1:
            raise ValueError("variant annotation request must use one genome assembly")
        return self

    @property
    def input_hash(self) -> str:
        """Hash the exact, order-independent allele query and species."""
        return canonical_sha256(
            {
                "species_taxon_id": self.species_taxon_id,
                "variant_ids": sorted(item.canonical_id for item in self.variants),
                "resource_snapshot_id": self.resource.snapshot_id,
                "normalization_policy": self.normalization_policy,
            }
        )


class VariantAnnotationBatch(BaseModel):
    """Frozen external variant annotations that are not efficacy evidence."""

    model_config = ConfigDict(frozen=True)

    request: VariantAnnotationRequest
    records: tuple[VariantAnnotationRecord, ...] = ()
    unmatched_variant_ids: tuple[str, ...] = ()
    resource: ResourceSnapshot
    provenance: QueryProvenance
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        """Require a single variant domain and a resource captured in provenance."""
        identifiers = [item.variant.canonical_id for item in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("variant annotation records must have unique keys")
        requested = {item.canonical_id for item in self.request.variants}
        matched = set(identifiers)
        unmatched = set(self.unmatched_variant_ids)
        if len(unmatched) != len(self.unmatched_variant_ids):
            raise ValueError("unmatched variant identifiers must be unique")
        if matched.intersection(unmatched) or matched.union(unmatched) != requested:
            raise ValueError("matched and unmatched variants must partition the exact request")
        request_assembly = self.request.variants[0].genome_assembly
        if any(item.variant.genome_assembly != request_assembly for item in self.records):
            raise ValueError("variant annotation records must match the request assembly")
        if self.provenance.domain.species_taxon_id != self.request.species_taxon_id:
            raise ValueError("variant provenance species does not match the annotation batch")
        if self.provenance.domain.feature_type is not GenomicFeatureType.VARIANT:
            raise ValueError("variant provenance must declare variant features")
        if self.provenance.domain.genome_assembly != request_assembly:
            raise ValueError("variant provenance assembly does not match annotation records")
        if self.resource != self.request.resource:
            raise ValueError("variant batch resource must match its request")
        if self.resource not in self.provenance.resources:
            raise ValueError("variant resource must be captured in query provenance")
        if self.provenance.input_hash != self.request.input_hash:
            raise ValueError("variant provenance input hash must match the exact allele request")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("variant annotation warnings must be unique")
        if (
            not self.provenance.complete
            and "variant_annotation_result_incomplete" not in self.warnings
        ):
            raise ValueError("incomplete variant results must report an incompleteness warning")
        if self.provenance.warnings != self.warnings:
            raise ValueError("variant annotation warnings must match query provenance")
        if identifiers != sorted(identifiers):
            raise ValueError("variant annotation records must be canonically ordered")
        normalized_hash = _variant_response_hash(
            request_hash=self.request.input_hash,
            records=self.records,
            unmatched_variant_ids=self.unmatched_variant_ids,
            complete=self.provenance.complete,
            warnings=self.warnings,
        )
        if self.provenance.response_checksum != normalized_hash:
            raise ValueError(
                "variant response checksum does not match normalized imported annotations"
            )
        return self


class VariantAnnotationMatch(BaseModel):
    """One exact matrix-feature to external-annotation match."""

    model_config = ConfigDict(frozen=True)

    matrix_feature_id: str = Field(min_length=1)
    annotation: VariantAnnotationRecord


class VariantAnnotationJoin(BaseModel):
    """Audit of an allele- and assembly-strict annotation join."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    matrix_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_response_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    matrix_feature_ids: tuple[str, ...]
    annotation_variant_ids: tuple[str, ...]
    matches: tuple[VariantAnnotationMatch, ...]
    unmatched_matrix_feature_ids: tuple[str, ...]
    unmatched_annotation_variant_ids: tuple[str, ...]
    join_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_join_contract(self) -> Self:
        """Reconstruct partitions and bind all normalized join content."""
        matches = tuple(
            VariantAnnotationMatch.model_validate(item.model_dump(mode="python"))
            for item in self.matches
        )
        match_keys = tuple(
            (item.matrix_feature_id, item.annotation.variant.canonical_id) for item in matches
        )
        if match_keys != tuple(sorted(match_keys)):
            raise ValueError("variant annotation matches must be canonically ordered")
        matched_matrix_ids = tuple(item.matrix_feature_id for item in matches)
        if len(set(matched_matrix_ids)) != len(matched_matrix_ids):
            raise ValueError("each matrix feature may have at most one annotation match")

        for values, label in (
            (self.matrix_feature_ids, "matrix feature identifiers"),
            (self.annotation_variant_ids, "annotation variant identifiers"),
            (self.unmatched_matrix_feature_ids, "unmatched matrix feature identifiers"),
            (
                self.unmatched_annotation_variant_ids,
                "unmatched annotation variant identifiers",
            ),
            (self.warnings, "variant annotation join warnings"),
        ):
            _require_unique_clean_strings(values, label)
        for values, label in (
            (self.matrix_feature_ids, "matrix feature identifiers"),
            (self.annotation_variant_ids, "annotation variant identifiers"),
            (self.unmatched_matrix_feature_ids, "unmatched matrix feature identifiers"),
            (
                self.unmatched_annotation_variant_ids,
                "unmatched annotation variant identifiers",
            ),
        ):
            if values != tuple(sorted(values)):
                raise ValueError(f"{label} must be canonically ordered")

        matrix_universe = set(self.matrix_feature_ids)
        matched_matrix = set(matched_matrix_ids)
        unmatched_matrix = set(self.unmatched_matrix_feature_ids)
        if (
            matched_matrix & unmatched_matrix
            or matched_matrix | unmatched_matrix != matrix_universe
        ):
            raise ValueError(
                "matched and unmatched matrix features must partition matrix_feature_ids"
            )
        annotation_universe = set(self.annotation_variant_ids)
        matched_annotations = {item.annotation.variant.canonical_id for item in matches}
        unmatched_annotations = set(self.unmatched_annotation_variant_ids)
        if (
            matched_annotations & unmatched_annotations
            or matched_annotations | unmatched_annotations != annotation_universe
        ):
            raise ValueError(
                "matched and unmatched annotations must partition annotation_variant_ids"
            )

        expected_warnings: list[str] = []
        if unmatched_matrix:
            expected_warnings.append("matrix_variants_without_annotations")
        if unmatched_annotations:
            expected_warnings.append("annotation_variants_absent_from_matrix")
        if self.warnings != tuple(expected_warnings):
            raise ValueError("variant annotation join warnings do not match unmatched partitions")
        expected_checksum = _variant_join_hash(
            matrix_artifact_hash=self.matrix_artifact_hash,
            annotation_query_hash=self.annotation_query_hash,
            annotation_response_checksum=self.annotation_response_checksum,
            matrix_feature_ids=self.matrix_feature_ids,
            annotation_variant_ids=self.annotation_variant_ids,
            matches=matches,
            unmatched_matrix_feature_ids=self.unmatched_matrix_feature_ids,
            unmatched_annotation_variant_ids=self.unmatched_annotation_variant_ids,
            warnings=self.warnings,
        )
        if self.join_checksum != expected_checksum:
            raise ValueError("variant annotation join checksum does not match normalized content")
        object.__setattr__(self, "matches", matches)
        return self


def read_variant_annotations(
    frame: pd.DataFrame,
    *,
    chromosome_column: str,
    position_column: str,
    reference_column: str,
    alternate_column: str,
    consequence_column: str,
    genome_assembly: str,
    species_taxon_id: int,
    requested_variants: tuple[VariantKey, ...],
    gene_namespace: FeatureNamespace | None,
    resource: ResourceSnapshot,
    provenance: QueryProvenance,
    gene_id_column: str | None = None,
    transcript_id_column: str | None = None,
    hgvs_c_column: str | None = None,
    hgvs_p_column: str | None = None,
    impact_column: str | None = None,
    canonical_transcript_column: str | None = None,
    source_record_id_column: str | None = None,
    population_frequency_columns: Mapping[str, str] | None = None,
    consequence_separator: str = "&",
) -> VariantAnnotationBatch:
    """Read a saved VEP-like table without performing a live provider query."""
    required = {
        chromosome_column,
        position_column,
        reference_column,
        alternate_column,
        consequence_column,
    }
    optional = {
        value
        for value in (
            gene_id_column,
            transcript_id_column,
            hgvs_c_column,
            hgvs_p_column,
            impact_column,
            canonical_transcript_column,
            source_record_id_column,
        )
        if value is not None
    }
    frequency_columns = dict(population_frequency_columns or {})
    missing = sorted((required | optional | set(frequency_columns.values())).difference(frame))
    if missing:
        raise ValueError(f"variant annotation columns are absent: {missing}")
    if gene_id_column is not None and gene_namespace is None:
        raise ValueError("gene_namespace is required when gene IDs are imported")
    if not consequence_separator:
        raise ValueError("consequence_separator must be nonempty")
    if frame.columns.duplicated().any():
        raise ValueError("variant annotation DataFrame column labels must be unique")

    request = VariantAnnotationRequest(
        species_taxon_id=species_taxon_id,
        variants=requested_variants,
        resource=resource,
    )
    requested_ids = {item.canonical_id for item in request.variants}

    grouped_consequences: defaultdict[str, list[VariantConsequence]] = defaultdict(list)
    grouped_keys: dict[str, VariantKey] = {}
    grouped_frequencies: defaultdict[str, dict[str, float]] = defaultdict(dict)
    grouped_source_ids: defaultdict[str, list[str]] = defaultdict(list)
    for _, row in frame.iterrows():
        key = VariantKey(
            genome_assembly=genome_assembly,
            chromosome=row[chromosome_column],
            position_1_based=row[position_column],
            reference_allele=row[reference_column],
            alternate_allele=row[alternate_column],
        )
        canonical_id = key.canonical_id
        if canonical_id not in requested_ids:
            raise ValueError(
                f"variant annotation response contains allele outside the exact request: "
                f"{canonical_id}"
            )
        grouped_keys[canonical_id] = key
        consequence_text = _required_scalar_text(row[consequence_column], "variant consequence")
        terms = tuple(
            sorted(
                {
                    value.strip()
                    for value in consequence_text.split(consequence_separator)
                    if value.strip()
                }
            )
        )
        if not terms:
            raise ValueError("variant consequence must contain a nonempty term")
        consequence = VariantConsequence(
            sequence_ontology_terms=terms,
            gene_id=_optional_string(row, gene_id_column),
            gene_namespace=gene_namespace if gene_id_column is not None else None,
            transcript_id=_optional_string(row, transcript_id_column),
            hgvs_c=_optional_string(row, hgvs_c_column),
            hgvs_p=_optional_string(row, hgvs_p_column),
            impact_label=_optional_string(row, impact_column),
            canonical_transcript=_optional_bool(row, canonical_transcript_column),
        )
        if consequence not in grouped_consequences[canonical_id]:
            grouped_consequences[canonical_id].append(consequence)
        for population, column in frequency_columns.items():
            if pd.isna(row[column]):
                continue
            value = float(row[column])
            previous = grouped_frequencies[canonical_id].get(population)
            if previous is not None and previous != value:
                raise ValueError(
                    f"conflicting {population!r} population frequencies for {canonical_id}"
                )
            grouped_frequencies[canonical_id][population] = value
        source_id = _optional_string(row, source_record_id_column)
        if source_id is not None and source_id not in grouped_source_ids[canonical_id]:
            grouped_source_ids[canonical_id].append(source_id)

    records = tuple(
        VariantAnnotationRecord(
            variant=grouped_keys[identifier],
            consequences=tuple(grouped_consequences[identifier]),
            population_frequencies=grouped_frequencies[identifier],
            source_record_ids=tuple(grouped_source_ids[identifier]),
        )
        for identifier in sorted(grouped_keys)
    )
    unmatched = tuple(sorted(requested_ids.difference(grouped_keys)))
    warnings = list(provenance.warnings)
    _append_once(warnings, "variant_equivalence_normalization_not_performed")
    if unmatched:
        _append_once(warnings, "requested_variants_without_annotations")
    if not provenance.complete:
        _append_once(warnings, "variant_annotation_result_incomplete")
    normalized_hash = _variant_response_hash(
        request_hash=request.input_hash,
        records=records,
        unmatched_variant_ids=unmatched,
        complete=provenance.complete,
        warnings=tuple(warnings),
    )
    if provenance.response_checksum is not None and provenance.response_checksum != normalized_hash:
        raise ValueError("supplied response checksum does not match normalized variant annotations")
    normalized_provenance = provenance.model_copy(
        update={"response_checksum": normalized_hash, "warnings": tuple(warnings)}
    )
    return VariantAnnotationBatch(
        request=request,
        records=records,
        unmatched_variant_ids=unmatched,
        resource=resource,
        provenance=normalized_provenance,
        warnings=tuple(warnings),
    )


def join_variant_annotations(
    matrix: GenomicMatrix,
    annotations: VariantAnnotationBatch,
) -> VariantAnnotationJoin:
    """Join annotations using assembly, chromosome, position, reference, and alternate."""
    annotations = VariantAnnotationBatch.model_validate(annotations.model_dump(mode="python"))
    if matrix.scale is not MatrixScale.VARIANT_DOSAGE:
        raise ValueError("variant annotation joins require a variant-dosage matrix")
    matrix_species = {sample.species_taxon_id for sample in matrix.samples}
    if matrix_species != {annotations.request.species_taxon_id}:
        raise ValueError("matrix and annotation species do not match")
    matrix_assemblies = {feature.genome_assembly for feature in matrix.features}
    annotation_assembly = annotations.provenance.domain.genome_assembly
    if matrix_assemblies != {annotation_assembly}:
        raise ValueError("matrix and annotation genome assemblies do not match")
    lookup = {record.variant.canonical_id: record for record in annotations.records}
    matches: list[VariantAnnotationMatch] = []
    unmatched_matrix: list[str] = []
    matched_annotation_ids: set[str] = set()
    for feature in matrix.features:
        if feature.feature_type is not GenomicFeatureType.VARIANT:
            raise ValueError("variant annotation joins require only variant matrix features")
        reference = feature.attributes.get("reference")
        alternate = feature.attributes.get("alternate")
        if (
            feature.genome_assembly is None
            or feature.chromosome is None
            or feature.start is None
            or not isinstance(reference, str)
            or not isinstance(alternate, str)
        ):
            raise ValueError(
                f"matrix feature {feature.feature_id!r} lacks assembly-aware allele metadata"
            )
        if feature.end != feature.start + len(reference):
            raise ValueError(
                f"matrix feature {feature.feature_id!r} interval length does not match "
                "its reference allele"
            )
        key = VariantKey(
            genome_assembly=feature.genome_assembly,
            chromosome=feature.chromosome,
            position_1_based=feature.start + 1,
            reference_allele=reference,
            alternate_allele=alternate,
        )
        annotation = lookup.get(key.canonical_id)
        if annotation is None:
            unmatched_matrix.append(feature.feature_id)
            continue
        matches.append(
            VariantAnnotationMatch(matrix_feature_id=feature.feature_id, annotation=annotation)
        )
        matched_annotation_ids.add(key.canonical_id)
    matrix_feature_ids = tuple(sorted(feature.feature_id for feature in matrix.features))
    annotation_variant_ids = tuple(sorted(lookup))
    normalized_matches = tuple(
        sorted(
            matches,
            key=lambda item: (
                item.matrix_feature_id,
                item.annotation.variant.canonical_id,
            ),
        )
    )
    normalized_unmatched_matrix = tuple(sorted(unmatched_matrix))
    unmatched_annotations = tuple(sorted(set(lookup).difference(matched_annotation_ids)))
    warnings: list[str] = []
    if unmatched_matrix:
        warnings.append("matrix_variants_without_annotations")
    if unmatched_annotations:
        warnings.append("annotation_variants_absent_from_matrix")
    response_checksum = annotations.provenance.response_checksum
    if response_checksum is None:  # pragma: no cover - enforced by VariantAnnotationBatch
        raise ValueError("variant annotation batch lacks a normalized response checksum")
    annotation_query_hash = annotations.provenance.query_hash
    join_checksum = _variant_join_hash(
        matrix_artifact_hash=matrix.artifact_hash,
        annotation_query_hash=annotation_query_hash,
        annotation_response_checksum=response_checksum,
        matrix_feature_ids=matrix_feature_ids,
        annotation_variant_ids=annotation_variant_ids,
        matches=normalized_matches,
        unmatched_matrix_feature_ids=normalized_unmatched_matrix,
        unmatched_annotation_variant_ids=unmatched_annotations,
        warnings=tuple(warnings),
    )
    return VariantAnnotationJoin(
        matrix_artifact_hash=matrix.artifact_hash,
        annotation_query_hash=annotation_query_hash,
        annotation_response_checksum=response_checksum,
        matrix_feature_ids=matrix_feature_ids,
        annotation_variant_ids=annotation_variant_ids,
        matches=normalized_matches,
        unmatched_matrix_feature_ids=normalized_unmatched_matrix,
        unmatched_annotation_variant_ids=unmatched_annotations,
        join_checksum=join_checksum,
        warnings=tuple(warnings),
    )


def _optional_string(row: pd.Series, column: str | None) -> str | None:
    if column is None or pd.isna(row[column]):
        return None
    value = str(row[column]).strip()
    return value or None


def _required_scalar_text(value: object, field_name: str) -> str:
    """Return normalized text while rejecting missing/non-scalar values."""
    missing = pd.isna(cast(Any, value))
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError(f"{field_name} cannot be missing")
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a scalar value")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _positive_integral_value(value: object, field_name: str) -> int:
    """Return a positive integer without truncating floating-point positions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be an integer, not a boolean")
    if value is None or bool(pd.isna(cast(Any, value))):
        raise ValueError(f"{field_name} cannot be missing")
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{field_name} must be a finite integer without truncation")
        parsed = int(numeric)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.isdecimal():
            raise ValueError(f"{field_name} must be a positive base-10 integer")
        parsed = int(stripped)
    else:
        raise ValueError(f"{field_name} must be an integer")
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _variant_response_hash(
    *,
    request_hash: str,
    records: tuple[VariantAnnotationRecord, ...],
    unmatched_variant_ids: tuple[str, ...],
    complete: bool,
    warnings: tuple[str, ...],
) -> str:
    """Hash the canonical typed import separately from archived provider bytes."""
    return canonical_sha256(
        {
            "request_hash": request_hash,
            "records": [item.model_dump(mode="json") for item in records],
            "unmatched_variant_ids": sorted(unmatched_variant_ids),
            "complete": complete,
            "warnings": sorted(warnings),
        }
    )


def _variant_join_hash(
    *,
    matrix_artifact_hash: str,
    annotation_query_hash: str,
    annotation_response_checksum: str,
    matrix_feature_ids: tuple[str, ...],
    annotation_variant_ids: tuple[str, ...],
    matches: tuple[VariantAnnotationMatch, ...],
    unmatched_matrix_feature_ids: tuple[str, ...],
    unmatched_annotation_variant_ids: tuple[str, ...],
    warnings: tuple[str, ...],
) -> str:
    """Hash the complete normalized join audit and its two source identities."""
    return canonical_sha256(
        {
            "schema": "rejuvenationkit.variant-annotation-join/v2",
            "matrix_artifact_hash": matrix_artifact_hash,
            "annotation_query_hash": annotation_query_hash,
            "annotation_response_checksum": annotation_response_checksum,
            "matrix_feature_ids": sorted(matrix_feature_ids),
            "annotation_variant_ids": sorted(annotation_variant_ids),
            "matches": [item.model_dump(mode="json") for item in matches],
            "unmatched_matrix_feature_ids": sorted(unmatched_matrix_feature_ids),
            "unmatched_annotation_variant_ids": sorted(unmatched_annotation_variant_ids),
            "warnings": warnings,
            "fusion_eligibility": "not_fusible",
        }
    )


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_unique_clean_strings(values: tuple[str, ...], field_name: str) -> None:
    """Require an immutable identifier partition without duplicates."""
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _optional_bool(row: pd.Series, column: str | None) -> bool:
    if column is None or pd.isna(row[column]):
        return False
    value = row[column]
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"cannot interpret canonical-transcript value {value!r} as boolean")
