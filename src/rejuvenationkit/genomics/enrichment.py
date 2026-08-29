"""Offline, typed gene-set overrepresentation analysis.

The models in this module deliberately describe functional context rather than
efficacy evidence.  Overrepresentation p-values cannot be converted to Phase 2
``EvidenceEstimate`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal, Self, cast

import pandas as pd
import scipy
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import hypergeom

from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)
from rejuvenationkit.genomics.schemas import GenomicFeatureType

_OFFLINE_ORA_PROVIDER_ID = "rejuvenationkit.offline_ora"
_OFFLINE_ORA_PROVIDER_VERSION = "1.0.0"


class FunctionalGeneSet(BaseModel):
    """One immutable functional term and its resource-versioned members."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gene_set_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    domain: FeatureDomain
    member_feature_ids: tuple[str, ...]
    resource: ResourceSnapshot

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        """Require clean, nonempty, unique feature membership."""
        _require_clean_text(self.gene_set_id, "gene_set_id")
        _require_clean_text(self.name, "name")
        if self.description is not None:
            _require_clean_text(self.description, "description")
        if not self.member_feature_ids:
            raise ValueError("member_feature_ids must be nonempty")
        _require_unique_clean_strings(self.member_feature_ids, "member_feature_ids")
        if self.domain.feature_type not in {
            GenomicFeatureType.GENE,
            GenomicFeatureType.PROTEIN,
        }:
            raise ValueError("functional sets require a gene or protein feature domain")
        return self

    @property
    def content_hash(self) -> str:
        """Hash the term independently of member input order."""
        return canonical_sha256(
            {
                "gene_set_id": self.gene_set_id,
                "name": self.name,
                "description": self.description,
                "domain_hash": self.domain.domain_hash,
                "member_feature_ids": sorted(self.member_feature_ids),
                "resource_snapshot_id": self.resource.snapshot_id,
            }
        )


def _gene_set_collection_hash(
    *,
    collection_id: str,
    name: str,
    domain: FeatureDomain,
    gene_sets: tuple[FunctionalGeneSet, ...],
    resource: ResourceSnapshot,
    membership_policy: str,
    source_row_count: int | None,
) -> str:
    """Hash the normalized typed collection separately from archived raw bytes."""
    return canonical_sha256(
        {
            "schema": "rejuvenationkit.gene-set-collection/v1",
            "collection_id": collection_id,
            "name": name,
            "domain_hash": domain.domain_hash,
            "resource_snapshot_id": resource.snapshot_id,
            "gene_set_hashes": sorted(item.content_hash for item in gene_sets),
            "membership_policy": membership_policy,
            "source_row_count": source_row_count,
        }
    )


class GeneSetCollection(BaseModel):
    """A coherent collection of gene sets from one exact resource snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    domain: FeatureDomain
    gene_sets: tuple[FunctionalGeneSet, ...]
    resource: ResourceSnapshot
    membership_policy: str = "all_input_rows_no_hierarchy_expansion"
    source_row_count: int | None = Field(default=None, ge=0)
    normalized_import_sha256: str = ""

    @model_validator(mode="after")
    def validate_collection(self) -> Self:
        """Reject mixed domains, snapshots, or duplicate term identifiers."""
        _require_clean_text(self.collection_id, "collection_id")
        _require_clean_text(self.name, "name")
        if not self.gene_sets:
            raise ValueError("gene_sets must be nonempty")
        identifiers = tuple(item.gene_set_id for item in self.gene_sets)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("gene_set_id values must be unique within a collection")
        if any(item.domain != self.domain for item in self.gene_sets):
            raise ValueError("all gene sets must match the collection domain")
        if any(item.resource.snapshot_id != self.resource.snapshot_id for item in self.gene_sets):
            raise ValueError("all gene sets must match the collection resource snapshot")
        _require_clean_text(self.membership_policy, "membership_policy")
        expected_hash = _gene_set_collection_hash(
            collection_id=self.collection_id,
            name=self.name,
            domain=self.domain,
            gene_sets=self.gene_sets,
            resource=self.resource,
            membership_policy=self.membership_policy,
            source_row_count=self.source_row_count,
        )
        if self.normalized_import_sha256 and self.normalized_import_sha256 != expected_hash:
            raise ValueError(
                "normalized_import_sha256 does not match the canonical gene-set collection"
            )
        object.__setattr__(self, "normalized_import_sha256", expected_hash)
        return self

    @property
    def content_hash(self) -> str:
        """Hash collection identity, resource, domain, and all term definitions."""
        return self.normalized_import_sha256


def gene_set_collection_from_frame(
    frame: pd.DataFrame,
    *,
    collection_id: str,
    collection_name: str,
    domain: FeatureDomain,
    resource: ResourceSnapshot,
    gene_set_id_column: str,
    member_feature_id_column: str,
    gene_set_name_column: str | None = None,
    description_column: str | None = None,
    membership_policy: str = "all_input_rows_no_hierarchy_expansion",
    expected_normalized_import_sha256: str | None = None,
) -> GeneSetCollection:
    """Load a long-form offline gene-set table through explicit column mappings."""
    required_columns = {gene_set_id_column, member_feature_id_column}
    optional_columns = {
        column for column in (gene_set_name_column, description_column) if column is not None
    }
    missing_columns = sorted((required_columns | optional_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"gene-set columns are absent: {missing_columns}")
    if frame.empty:
        raise ValueError("gene-set frame cannot be empty")
    if frame.columns.duplicated().any():
        raise ValueError("gene-set DataFrame column labels must be unique")

    selected_columns = sorted(required_columns | optional_columns)
    records = cast(list[dict[str, object]], frame.loc[:, selected_columns].to_dict("records"))
    members_by_id: dict[str, list[str]] = {}
    names_by_id: dict[str, str] = {}
    descriptions_by_id: dict[str, str | None] = {}
    seen_memberships: set[tuple[str, str]] = set()
    for record in records:
        gene_set_id = _required_frame_text(record[gene_set_id_column], gene_set_id_column)
        member_id = _required_frame_text(record[member_feature_id_column], member_feature_id_column)
        name = (
            gene_set_id
            if gene_set_name_column is None
            else _required_frame_text(record[gene_set_name_column], gene_set_name_column)
        )
        description = (
            None
            if description_column is None
            else _optional_frame_text(record[description_column], description_column)
        )
        pair = (gene_set_id, member_id)
        if pair in seen_memberships:
            raise ValueError(f"duplicate gene-set membership row: {gene_set_id}, {member_id}")
        seen_memberships.add(pair)
        if gene_set_id in names_by_id and names_by_id[gene_set_id] != name:
            raise ValueError(f"gene set {gene_set_id} has inconsistent names")
        if gene_set_id in descriptions_by_id and descriptions_by_id[gene_set_id] != description:
            raise ValueError(f"gene set {gene_set_id} has inconsistent descriptions")
        names_by_id[gene_set_id] = name
        descriptions_by_id[gene_set_id] = description
        members_by_id.setdefault(gene_set_id, []).append(member_id)

    gene_sets = tuple(
        FunctionalGeneSet(
            gene_set_id=gene_set_id,
            name=names_by_id[gene_set_id],
            description=descriptions_by_id[gene_set_id],
            domain=domain,
            member_feature_ids=tuple(sorted(members_by_id[gene_set_id])),
            resource=resource,
        )
        for gene_set_id in sorted(members_by_id)
    )
    collection = GeneSetCollection(
        collection_id=collection_id,
        name=collection_name,
        domain=domain,
        gene_sets=gene_sets,
        resource=resource,
        membership_policy=membership_policy,
        source_row_count=len(frame),
    )
    if (
        expected_normalized_import_sha256 is not None
        and collection.normalized_import_sha256 != expected_normalized_import_sha256
    ):
        raise ValueError(
            "expected normalized gene-set checksum does not match the imported collection"
        )
    return collection


class MultipleTestingMethod(StrEnum):
    """Local family-wise or false-discovery correction applied to ORA terms."""

    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BONFERRONI = "bonferroni"
    NONE = "none"


class MultipleTestingAudit(BaseModel):
    """Exact correction method and hypothesis family used by one ORA result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: MultipleTestingMethod
    alpha: float = Field(gt=0, lt=1)
    family_size: int = Field(ge=0)
    family_definition: str = Field(min_length=1)


class OverrepresentationRequest(BaseModel):
    """One directionless ORA query with an explicit measured background."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_type: Literal["overrepresentation"] = "overrepresentation"
    selected_features: FeatureCollection
    background_universe: FeatureCollection
    gene_sets: GeneSetCollection
    multiple_testing_method: MultipleTestingMethod = MultipleTestingMethod.BENJAMINI_HOCHBERG
    alpha: float = Field(default=0.05, gt=0, lt=1)
    minimum_term_size: int = Field(default=5, ge=1)
    maximum_term_size: int = Field(default=500, ge=1)

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        """Enforce exact domain compatibility and selected-within-background membership."""
        if self.maximum_term_size < self.minimum_term_size:
            raise ValueError("maximum_term_size must be at least minimum_term_size")
        if self.selected_features.domain != self.background_universe.domain:
            raise ValueError("selected features and background universe must share one domain")
        if self.gene_sets.domain != self.background_universe.domain:
            raise ValueError("gene-set and background domains must match exactly")
        if not set(self.selected_features.feature_ids).issubset(
            self.background_universe.feature_ids
        ):
            raise ValueError("selected features must be a subset of the background universe")
        return self

    @property
    def input_hash(self) -> str:
        """Hash exact selected and background feature collections."""
        return canonical_sha256(
            {
                "selected_features": self.selected_features.content_hash,
                "background_universe": self.background_universe.content_hash,
            }
        )

    @property
    def query_parameters(self) -> dict[str, object]:
        """Return the complete secret-free local ORA parameterization."""
        return {
            "analysis_type": self.analysis_type,
            "gene_set_collection_id": self.gene_sets.collection_id,
            "gene_set_collection_hash": self.gene_sets.content_hash,
            "selected_feature_collection_hash": self.selected_features.content_hash,
            "background_universe_hash": self.background_universe.content_hash,
            "multiple_testing_method": self.multiple_testing_method.value,
            "alpha": self.alpha,
            "minimum_term_size": self.minimum_term_size,
            "maximum_term_size": self.maximum_term_size,
            "alternative": "greater",
        }


class OverrepresentationTerm(BaseModel):
    """One directionless one-sided hypergeometric gene-set result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gene_set_id: str = Field(min_length=1)
    gene_set_name: str = Field(min_length=1)
    description: str | None = None
    overlap_feature_ids: tuple[str, ...]
    selected_overlap_count: int = Field(ge=0)
    selected_size: int = Field(ge=1)
    background_gene_set_size: int = Field(ge=1)
    background_size: int = Field(ge=1)
    original_gene_set_size: int = Field(ge=1)
    members_outside_background_count: int = Field(ge=0)
    expected_overlap: float = Field(ge=0)
    fold_enrichment: float = Field(ge=0)
    p_value: float = Field(ge=0, le=1)
    adjusted_p_value: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Ensure counts, overlap identifiers, and derived metrics agree."""
        if len(set(self.overlap_feature_ids)) != len(self.overlap_feature_ids):
            raise ValueError("overlap_feature_ids must be unique")
        if len(self.overlap_feature_ids) != self.selected_overlap_count:
            raise ValueError("selected_overlap_count must match overlap_feature_ids")
        if self.selected_overlap_count > min(self.selected_size, self.background_gene_set_size):
            raise ValueError("selected overlap exceeds the query or term size")
        if self.background_gene_set_size > self.background_size:
            raise ValueError("background gene-set size exceeds the background universe")
        if (
            self.background_gene_set_size + self.members_outside_background_count
            != self.original_gene_set_size
        ):
            raise ValueError("original gene-set size does not match background membership audit")
        numerical = (
            self.expected_overlap,
            self.fold_enrichment,
            self.p_value,
            self.adjusted_p_value,
        )
        if not all(isfinite(value) for value in numerical):
            raise ValueError("ORA numerical values must be finite")
        return self


class OverrepresentationResult(BaseModel):
    """Descriptive ORA terms, multiplicity audit, and complete query provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_type: Literal["overrepresentation"] = "overrepresentation"
    request: OverrepresentationRequest
    terms: tuple[OverrepresentationTerm, ...]
    multiple_testing: MultipleTestingAudit
    tested_gene_set_ids: tuple[str, ...]
    unmatched_gene_set_ids: tuple[str, ...]
    size_filtered_gene_set_ids: tuple[str, ...]
    matched_selected_feature_ids: tuple[str, ...]
    unmatched_selected_feature_ids: tuple[str, ...]
    provenance: QueryProvenance
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_audit(self) -> Self:
        """Reconstruct the complete ORA and reject inconsistent public construction."""
        validated_request = OverrepresentationRequest.model_validate(
            self.request.model_dump(mode="python")
        )
        validated_provenance = QueryProvenance.model_validate(
            self.provenance.model_dump(mode="python")
        )
        if validated_request != self.request:
            raise ValueError("embedded ORA request must satisfy its complete public contract")
        if validated_provenance != self.provenance:
            raise ValueError("embedded ORA provenance must satisfy its complete public contract")
        expected = _calculate_overrepresentation(validated_request)
        _validate_reconstructed_terms(self.terms, expected.terms)
        if self.tested_gene_set_ids != expected.tested_gene_set_ids:
            raise ValueError(
                "tested_gene_set_ids must be deterministic and match reconstructed terms"
            )
        if self.unmatched_gene_set_ids != expected.unmatched_gene_set_ids:
            raise ValueError(
                "unmatched_gene_set_ids must match reconstructed background membership"
            )
        if self.size_filtered_gene_set_ids != expected.size_filtered_gene_set_ids:
            raise ValueError(
                "size_filtered_gene_set_ids must match reconstructed term-size filtering"
            )
        if self.matched_selected_feature_ids != expected.matched_selected_feature_ids:
            raise ValueError(
                "matched_selected_feature_ids must match reconstructed tested membership"
            )
        if self.unmatched_selected_feature_ids != expected.unmatched_selected_feature_ids:
            raise ValueError(
                "unmatched_selected_feature_ids must match reconstructed tested membership"
            )
        if self.multiple_testing != expected.multiple_testing:
            raise ValueError(
                "multiple-testing audit and family definition must match reconstructed query"
            )
        if self.warnings != expected.warnings:
            raise ValueError("result warnings must match reconstructed ORA warnings")
        if self.provenance.domain != self.request.background_universe.domain:
            raise ValueError("provenance domain must match the ORA request")
        if self.provenance.input_hash != self.request.input_hash:
            raise ValueError("provenance input hash must match the ORA request")
        if self.provenance.query_parameters != self.request.query_parameters:
            raise ValueError("provenance parameters must match the ORA request")
        if self.provenance.resources != (self.request.gene_sets.resource,):
            raise ValueError("provenance must contain exactly the gene-set resource snapshot")
        if self.provenance.provider_id != _OFFLINE_ORA_PROVIDER_ID:
            raise ValueError("provenance provider must identify the offline ORA implementation")
        if self.provenance.provider_version != _OFFLINE_ORA_PROVIDER_VERSION:
            raise ValueError("provenance version must identify this offline ORA implementation")
        if "scipy" not in self.provenance.software_versions:
            raise ValueError("offline ORA provenance must record the SciPy version")
        if not self.provenance.complete:
            raise ValueError("offline ORA results must have complete provenance")
        if self.provenance.warnings != self.warnings:
            raise ValueError("result and provenance warnings must match")
        if self.provenance.response_checksum != expected.response_checksum:
            raise ValueError("response checksum must match the reconstructed ORA result")
        return self


@dataclass(frozen=True, slots=True)
class _OverrepresentationCalculation:
    """Private deterministic reconstruction of every public ORA output field."""

    terms: tuple[OverrepresentationTerm, ...]
    multiple_testing: MultipleTestingAudit
    tested_gene_set_ids: tuple[str, ...]
    unmatched_gene_set_ids: tuple[str, ...]
    size_filtered_gene_set_ids: tuple[str, ...]
    matched_selected_feature_ids: tuple[str, ...]
    unmatched_selected_feature_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    response_checksum: str


def run_overrepresentation(
    request: OverrepresentationRequest,
    *,
    executed_at: datetime | None = None,
) -> OverrepresentationResult:
    """Run a local one-sided hypergeometric ORA with explicit multiplicity control."""
    calculation = _calculate_overrepresentation(request)
    provenance = QueryProvenance(
        provider_id=_OFFLINE_ORA_PROVIDER_ID,
        provider_version=_OFFLINE_ORA_PROVIDER_VERSION,
        resources=(request.gene_sets.resource,),
        domain=request.background_universe.domain,
        retrieved_at=executed_at or datetime.now(UTC),
        input_hash=request.input_hash,
        response_checksum=calculation.response_checksum,
        query_parameters=request.query_parameters,
        software_versions={"scipy": scipy.__version__},
        complete=True,
        warnings=calculation.warnings,
    )
    return OverrepresentationResult(
        request=request,
        terms=calculation.terms,
        multiple_testing=calculation.multiple_testing,
        tested_gene_set_ids=calculation.tested_gene_set_ids,
        unmatched_gene_set_ids=calculation.unmatched_gene_set_ids,
        size_filtered_gene_set_ids=calculation.size_filtered_gene_set_ids,
        matched_selected_feature_ids=calculation.matched_selected_feature_ids,
        unmatched_selected_feature_ids=calculation.unmatched_selected_feature_ids,
        provenance=provenance,
        warnings=calculation.warnings,
    )


def _calculate_overrepresentation(
    request: OverrepresentationRequest,
) -> _OverrepresentationCalculation:
    background = set(request.background_universe.feature_ids)
    selected = set(request.selected_features.feature_ids)
    background_size = len(background)
    selected_size = len(selected)
    raw_terms: list[OverrepresentationTerm] = []
    unmatched_gene_sets: list[str] = []
    size_filtered_gene_sets: list[str] = []
    tested_members: set[str] = set()

    for gene_set in sorted(request.gene_sets.gene_sets, key=lambda item: item.gene_set_id):
        original_members = set(gene_set.member_feature_ids)
        background_members = original_members.intersection(background)
        background_term_size = len(background_members)
        if background_term_size == 0:
            unmatched_gene_sets.append(gene_set.gene_set_id)
            continue
        if not request.minimum_term_size <= background_term_size <= request.maximum_term_size:
            size_filtered_gene_sets.append(gene_set.gene_set_id)
            continue
        overlap = tuple(sorted(background_members.intersection(selected)))
        overlap_count = len(overlap)
        expected_overlap = selected_size * background_term_size / background_size
        p_value = float(
            hypergeom.sf(
                overlap_count - 1,
                background_size,
                background_term_size,
                selected_size,
            )
        )
        raw_terms.append(
            OverrepresentationTerm(
                gene_set_id=gene_set.gene_set_id,
                gene_set_name=gene_set.name,
                description=gene_set.description,
                overlap_feature_ids=overlap,
                selected_overlap_count=overlap_count,
                selected_size=selected_size,
                background_gene_set_size=background_term_size,
                background_size=background_size,
                original_gene_set_size=len(original_members),
                members_outside_background_count=len(original_members - background),
                expected_overlap=expected_overlap,
                fold_enrichment=overlap_count / expected_overlap,
                p_value=p_value,
                adjusted_p_value=p_value,
            )
        )
        tested_members.update(background_members)

    adjusted = _adjust_p_values(
        tuple((item.gene_set_id, item.p_value) for item in raw_terms),
        request.multiple_testing_method,
    )
    terms = tuple(
        sorted(
            (
                item.model_copy(update={"adjusted_p_value": adjusted[item.gene_set_id]})
                for item in raw_terms
            ),
            key=lambda item: (item.adjusted_p_value, item.p_value, item.gene_set_id),
        )
    )
    tested_ids = tuple(sorted(item.gene_set_id for item in terms))
    matched_selected = tuple(sorted(selected.intersection(tested_members)))
    unmatched_selected = tuple(sorted(selected - tested_members))
    warnings: list[str] = []
    if unmatched_selected:
        warnings.append("selected_features_absent_from_tested_gene_sets")
    if unmatched_gene_sets:
        warnings.append("gene_sets_without_background_members")
    if size_filtered_gene_sets:
        warnings.append("gene_sets_filtered_by_background_term_size")
    if not terms:
        warnings.append("no_gene_sets_tested")

    family_definition = (
        "all collection terms with measured-background membership and background term size "
        f"in [{request.minimum_term_size}, {request.maximum_term_size}]"
    )
    audit = MultipleTestingAudit(
        method=request.multiple_testing_method,
        alpha=request.alpha,
        family_size=len(terms),
        family_definition=family_definition,
    )
    unmatched_ids = tuple(sorted(unmatched_gene_sets))
    filtered_ids = tuple(sorted(size_filtered_gene_sets))
    warning_tuple = tuple(warnings)
    response_checksum = _build_response_checksum(
        terms=terms,
        multiple_testing=audit,
        tested_gene_set_ids=tested_ids,
        unmatched_gene_set_ids=unmatched_ids,
        size_filtered_gene_set_ids=filtered_ids,
        matched_selected_feature_ids=matched_selected,
        unmatched_selected_feature_ids=unmatched_selected,
        warnings=warning_tuple,
    )
    return _OverrepresentationCalculation(
        terms=terms,
        multiple_testing=audit,
        tested_gene_set_ids=tested_ids,
        unmatched_gene_set_ids=unmatched_ids,
        size_filtered_gene_set_ids=filtered_ids,
        matched_selected_feature_ids=matched_selected,
        unmatched_selected_feature_ids=unmatched_selected,
        warnings=warning_tuple,
        response_checksum=response_checksum,
    )


def _build_response_checksum(
    *,
    terms: tuple[OverrepresentationTerm, ...],
    multiple_testing: MultipleTestingAudit,
    tested_gene_set_ids: tuple[str, ...],
    unmatched_gene_set_ids: tuple[str, ...],
    size_filtered_gene_set_ids: tuple[str, ...],
    matched_selected_feature_ids: tuple[str, ...],
    unmatched_selected_feature_ids: tuple[str, ...],
    warnings: tuple[str, ...],
) -> str:
    return canonical_sha256(
        {
            "terms": [item.model_dump(mode="json") for item in terms],
            "multiple_testing": multiple_testing.model_dump(mode="json"),
            "tested_gene_set_ids": tested_gene_set_ids,
            "unmatched_gene_set_ids": unmatched_gene_set_ids,
            "size_filtered_gene_set_ids": size_filtered_gene_set_ids,
            "matched_selected_feature_ids": matched_selected_feature_ids,
            "unmatched_selected_feature_ids": unmatched_selected_feature_ids,
            "warnings": warnings,
        }
    )


def _validate_reconstructed_terms(
    supplied: tuple[OverrepresentationTerm, ...],
    expected: tuple[OverrepresentationTerm, ...],
) -> None:
    supplied_ids = tuple(item.gene_set_id for item in supplied)
    expected_ids = tuple(item.gene_set_id for item in expected)
    if supplied_ids != expected_ids:
        raise ValueError("ORA terms must use reconstructed deterministic ordering")
    for actual, reconstructed in zip(supplied, expected, strict=True):
        if (
            actual.gene_set_name,
            actual.description,
            actual.original_gene_set_size,
            actual.members_outside_background_count,
        ) != (
            reconstructed.gene_set_name,
            reconstructed.description,
            reconstructed.original_gene_set_size,
            reconstructed.members_outside_background_count,
        ):
            raise ValueError(
                f"ORA term {actual.gene_set_id!r} metadata does not match its gene-set definition"
            )
        if actual.overlap_feature_ids != reconstructed.overlap_feature_ids:
            raise ValueError(
                f"ORA term {actual.gene_set_id!r} overlap does not match selected/background sets"
            )
        if (
            actual.selected_overlap_count,
            actual.selected_size,
            actual.background_gene_set_size,
            actual.background_size,
        ) != (
            reconstructed.selected_overlap_count,
            reconstructed.selected_size,
            reconstructed.background_gene_set_size,
            reconstructed.background_size,
        ):
            raise ValueError(
                f"ORA term {actual.gene_set_id!r} counts do not match the embedded request"
            )
        if actual.expected_overlap != reconstructed.expected_overlap:
            raise ValueError(
                f"ORA term {actual.gene_set_id!r} expected overlap does not match the null model"
            )
        if actual.fold_enrichment != reconstructed.fold_enrichment:
            raise ValueError(
                f"ORA term {actual.gene_set_id!r} fold enrichment does not match its counts"
            )
        if actual.p_value != reconstructed.p_value:
            raise ValueError(
                f"ORA term {actual.gene_set_id!r} raw hypergeometric p-value is inconsistent"
            )
        if actual.adjusted_p_value != reconstructed.adjusted_p_value:
            raise ValueError(
                "ORA adjusted p-values do not match the complete multiple-testing family"
            )


def _adjust_p_values(
    values: tuple[tuple[str, float], ...],
    method: MultipleTestingMethod,
) -> dict[str, float]:
    if not values:
        return {}
    family_size = len(values)
    if method is MultipleTestingMethod.NONE:
        return {identifier: p_value for identifier, p_value in values}
    if method is MultipleTestingMethod.BONFERRONI:
        return {identifier: min(1.0, p_value * family_size) for identifier, p_value in values}

    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_minimum = 1.0
    for reverse_index in range(family_size - 1, -1, -1):
        identifier, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running_minimum = min(running_minimum, p_value * family_size / rank)
        adjusted[identifier] = min(1.0, running_minimum)
    return adjusted


def _required_frame_text(value: object, column: str) -> str:
    if bool(pd.isna(cast(Any, value))):
        raise ValueError(f"gene-set column {column} contains a missing value")
    text = str(value)
    _require_clean_text(text, column)
    return text


def _optional_frame_text(value: object, column: str) -> str | None:
    if bool(pd.isna(cast(Any, value))):
        return None
    text = str(value)
    _require_clean_text(text, column)
    return text


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_unique_clean_strings(values: tuple[str, ...], field_name: str) -> None:
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
