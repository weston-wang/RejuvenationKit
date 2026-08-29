"""Directional gene-set analysis for a fixed, pre-ranked feature universe.

This module implements a weighted GSEA-like running-sum statistic and a
gene-set-membership permutation null.  The ranking is held fixed: permutations
sample feature sets of the same size and are not subject-, phenotype-, or
treatment-label permutations.  Results describe association with the supplied
ranking and are deliberately unavailable as Phase 2 efficacy evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from numbers import Real
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.genomics.enrichment import (
    GeneSetCollection,
    MultipleTestingAudit,
    MultipleTestingMethod,
)
from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)
from rejuvenationkit.genomics.schemas import GenomicFeatureType

_OFFLINE_RANKED_SET_PROVIDER_ID = "rejuvenationkit.offline_ranked_set"
_OFFLINE_RANKED_SET_PROVIDER_VERSION = "1.0.0"
_RANK_ORDERING = "score_descending_then_feature_id_ascending"
_NULL_NAME = "gene_set_membership_permutation_on_fixed_pre_ranked_universe"
_INFERENCE_SCOPE: Literal[
    "pre_ranked_feature_association_not_subject_level_treatment_inference"
] = "pre_ranked_feature_association_not_subject_level_treatment_inference"
_RNG_NAME = "numpy.random.PCG64"


class RankedSetDirection(StrEnum):
    """Location of a gene set within the fixed ranked feature universe."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class RankedSetPermutationNull(StrEnum):
    """Null model supported by the offline pre-ranked implementation."""

    GENE_SET_MEMBERSHIP = _NULL_NAME


class RankedFeature(BaseModel):
    """One feature and its finite pre-ranking score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: str = Field(min_length=1)
    score: float

    @model_validator(mode="after")
    def validate_feature(self) -> Self:
        """Reject ambiguous identifiers and non-finite ranking scores."""
        _require_clean_text(self.feature_id, "feature_id")
        if not isfinite(self.score):
            raise ValueError("ranked feature scores must be finite")
        if self.score == 0:
            object.__setattr__(self, "score", 0.0)
        return self


def _ranking_query_parameters(
    *,
    ranking_id: str,
    ranking_method: str,
    ranking_metric: str,
    higher_score_interpretation: str,
    feature_universe_hash: str,
) -> dict[str, object]:
    return {
        "artifact_type": "ranked_feature_universe",
        "ranking_id": ranking_id,
        "ranking_method": ranking_method,
        "ranking_metric": ranking_metric,
        "higher_score_interpretation": higher_score_interpretation,
        "feature_universe_hash": feature_universe_hash,
        "ordering": _RANK_ORDERING,
    }


def _ranking_checksum(
    *,
    ranking_id: str,
    ranking_method: str,
    ranking_metric: str,
    higher_score_interpretation: str,
    feature_universe: FeatureCollection,
    source_resource: ResourceSnapshot,
    ranked_features: Sequence[RankedFeature],
) -> str:
    return canonical_sha256(
        {
            "schema": "rejuvenationkit.ranked-feature-universe/v1",
            "ranking_id": ranking_id,
            "ranking_method": ranking_method,
            "ranking_metric": ranking_metric,
            "higher_score_interpretation": higher_score_interpretation,
            "ordering": _RANK_ORDERING,
            "feature_universe_hash": feature_universe.content_hash,
            "source_resource_snapshot_id": source_resource.snapshot_id,
            "ranked_features": [item.model_dump(mode="json") for item in ranked_features],
        }
    )


class RankedFeatureUniverse(BaseModel):
    """One complete deterministic ranking bound to its domain and source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ranking_id: str = Field(min_length=1)
    ranking_method: str = Field(min_length=1)
    ranking_metric: str = Field(min_length=1)
    higher_score_interpretation: str = Field(min_length=1)
    feature_universe: FeatureCollection
    source_resource: ResourceSnapshot
    ranked_features: tuple[RankedFeature, ...]
    provenance: QueryProvenance
    normalized_ranking_sha256: str = ""

    @model_validator(mode="after")
    def validate_ranking(self) -> Self:
        """Reconstruct ranking identity and enforce exact provenance binding."""
        _require_clean_text(self.ranking_id, "ranking_id")
        _require_clean_text(self.ranking_method, "ranking_method")
        _require_clean_text(self.ranking_metric, "ranking_metric")
        _require_clean_text(
            self.higher_score_interpretation,
            "higher_score_interpretation",
        )
        validated_universe = FeatureCollection.model_validate(
            self.feature_universe.model_dump(mode="python")
        )
        validated_resource = ResourceSnapshot.model_validate(
            self.source_resource.model_dump(mode="python")
        )
        validated_provenance = QueryProvenance.model_validate(
            self.provenance.model_dump(mode="python")
        )
        if validated_universe != self.feature_universe:
            raise ValueError("feature_universe must satisfy its complete public contract")
        if validated_resource != self.source_resource:
            raise ValueError("source_resource must satisfy its complete public contract")
        if validated_provenance != self.provenance:
            raise ValueError("ranking provenance must satisfy its complete public contract")
        if self.feature_universe.domain.feature_type not in {
            GenomicFeatureType.GENE,
            GenomicFeatureType.PROTEIN,
        }:
            raise ValueError("ranked-set analysis requires a gene or protein feature domain")
        if self.feature_universe.domain.namespace.is_legacy_ambiguous:
            raise ValueError("ranked-set analysis requires an explicit identifier namespace")
        if not self.ranked_features:
            raise ValueError("ranked_features must be nonempty")
        ranked_ids = tuple(item.feature_id for item in self.ranked_features)
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("ranked feature identifiers must be unique")
        if set(ranked_ids) != set(self.feature_universe.feature_ids):
            raise ValueError("ranked features must exactly equal the declared feature universe")
        expected_order = tuple(
            sorted(self.ranked_features, key=lambda item: (-item.score, item.feature_id))
        )
        if self.ranked_features != expected_order:
            raise ValueError(
                "ranked features must use score-descending, feature-ID-ascending order"
            )
        if self.feature_universe.source_snapshot_id != self.source_resource.snapshot_id:
            raise ValueError(
                "feature universe source_snapshot_id must match the ranking resource snapshot"
            )
        expected_parameters = _ranking_query_parameters(
            ranking_id=self.ranking_id,
            ranking_method=self.ranking_method,
            ranking_metric=self.ranking_metric,
            higher_score_interpretation=self.higher_score_interpretation,
            feature_universe_hash=self.feature_universe.content_hash,
        )
        if self.provenance.domain != self.feature_universe.domain:
            raise ValueError("ranking provenance domain must match the feature universe")
        if self.provenance.resources != (self.source_resource,):
            raise ValueError("ranking provenance must contain exactly its source resource")
        if self.provenance.input_hash != self.feature_universe.content_hash:
            raise ValueError("ranking provenance input hash must match the feature universe")
        if self.provenance.query_parameters != expected_parameters:
            raise ValueError("ranking provenance parameters do not match the ranking definition")
        if not self.provenance.complete:
            raise ValueError("ranked feature universes require complete provenance")
        expected_checksum = _ranking_checksum(
            ranking_id=self.ranking_id,
            ranking_method=self.ranking_method,
            ranking_metric=self.ranking_metric,
            higher_score_interpretation=self.higher_score_interpretation,
            feature_universe=self.feature_universe,
            source_resource=self.source_resource,
            ranked_features=self.ranked_features,
        )
        if self.normalized_ranking_sha256 and (self.normalized_ranking_sha256 != expected_checksum):
            raise ValueError("normalized_ranking_sha256 does not match the canonical ranking")
        if self.provenance.response_checksum != expected_checksum:
            raise ValueError(
                "ranking provenance response checksum must match the canonical ranking"
            )
        object.__setattr__(self, "normalized_ranking_sha256", expected_checksum)
        return self

    @property
    def artifact_hash(self) -> str:
        """Hash normalized scores together with exact source-query provenance."""
        return canonical_sha256(
            {
                "schema": "rejuvenationkit.ranked-feature-artifact/v1",
                "normalized_ranking_sha256": self.normalized_ranking_sha256,
                "provenance": self.provenance.model_dump(mode="json"),
            }
        )


def build_ranked_feature_universe(
    scores: Mapping[str, float],
    *,
    ranking_id: str,
    ranking_method: str,
    ranking_metric: str,
    higher_score_interpretation: str,
    feature_universe: FeatureCollection,
    source_resource: ResourceSnapshot,
    provider_id: str,
    provider_version: str,
    executed_at: datetime | None = None,
    software_versions: Mapping[str, str] | None = None,
    warnings: tuple[str, ...] = (),
) -> RankedFeatureUniverse:
    """Build a canonical complete ranking from an unordered score mapping."""
    supplied_ids: set[str] = set()
    ranked_items: list[RankedFeature] = []
    for raw_identifier, raw_score in scores.items():
        if not isinstance(raw_identifier, str):
            raise TypeError("ranked score keys must be strings")
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise TypeError("ranked scores must be real numbers, not booleans or strings")
        supplied_ids.add(raw_identifier)
        ranked_items.append(RankedFeature(feature_id=raw_identifier, score=float(raw_score)))
    expected_ids = set(feature_universe.feature_ids)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        unexpected = sorted(supplied_ids - expected_ids)
        raise ValueError(
            "ranked scores must exactly cover the feature universe; "
            f"missing={missing}, unexpected={unexpected}"
        )
    ranked_features = tuple(sorted(ranked_items, key=lambda item: (-item.score, item.feature_id)))
    checksum = _ranking_checksum(
        ranking_id=ranking_id,
        ranking_method=ranking_method,
        ranking_metric=ranking_metric,
        higher_score_interpretation=higher_score_interpretation,
        feature_universe=feature_universe,
        source_resource=source_resource,
        ranked_features=ranked_features,
    )
    provenance = QueryProvenance(
        provider_id=provider_id,
        provider_version=provider_version,
        resources=(source_resource,),
        domain=feature_universe.domain,
        retrieved_at=executed_at or datetime.now(UTC),
        input_hash=feature_universe.content_hash,
        response_checksum=checksum,
        query_parameters=_ranking_query_parameters(
            ranking_id=ranking_id,
            ranking_method=ranking_method,
            ranking_metric=ranking_metric,
            higher_score_interpretation=higher_score_interpretation,
            feature_universe_hash=feature_universe.content_hash,
        ),
        software_versions=software_versions or {},
        complete=True,
        warnings=warnings,
    )
    return RankedFeatureUniverse(
        ranking_id=ranking_id,
        ranking_method=ranking_method,
        ranking_metric=ranking_metric,
        higher_score_interpretation=higher_score_interpretation,
        feature_universe=feature_universe,
        source_resource=source_resource,
        ranked_features=ranked_features,
        provenance=provenance,
        normalized_ranking_sha256=checksum,
    )


class DirectionalRankedSetRequest(BaseModel):
    """One weighted directional query over a complete pre-ranked universe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_type: Literal["directional_ranked_set"] = "directional_ranked_set"
    ranking: RankedFeatureUniverse
    gene_sets: GeneSetCollection
    null_model: RankedSetPermutationNull = RankedSetPermutationNull.GENE_SET_MEMBERSHIP
    permutation_count: int = Field(default=1_000, ge=100, le=1_000_000)
    random_seed: int = 0
    weight_exponent: float = Field(default=1.0, ge=0, le=10)
    multiple_testing_method: MultipleTestingMethod = MultipleTestingMethod.BENJAMINI_HOCHBERG
    alpha: float = Field(default=0.05, gt=0, lt=1)
    minimum_term_size: int = Field(default=5, ge=1)
    maximum_term_size: int = Field(default=500, ge=1)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Require exact ranking/gene-set domain compatibility."""
        validated_ranking = RankedFeatureUniverse.model_validate(
            self.ranking.model_dump(mode="python")
        )
        validated_gene_sets = GeneSetCollection.model_validate(
            self.gene_sets.model_dump(mode="python")
        )
        if validated_ranking != self.ranking:
            raise ValueError("ranking must satisfy its complete public contract")
        if validated_gene_sets != self.gene_sets:
            raise ValueError("gene_sets must satisfy its complete public contract")
        if self.maximum_term_size < self.minimum_term_size:
            raise ValueError("maximum_term_size must be at least minimum_term_size")
        if self.gene_sets.domain != self.ranking.feature_universe.domain:
            raise ValueError("gene-set and ranked-universe domains must match exactly")
        if not isfinite(self.weight_exponent):
            raise ValueError("weight_exponent must be finite")
        return self

    @property
    def input_hash(self) -> str:
        """Hash the exact ranking artifact and exact gene-set collection."""
        return canonical_sha256(
            {
                "ranking_artifact_hash": self.ranking.artifact_hash,
                "gene_set_collection_hash": self.gene_sets.content_hash,
            }
        )

    @property
    def query_parameters(self) -> dict[str, object]:
        """Return the complete secret-free local analysis parameterization."""
        return {
            "analysis_type": self.analysis_type,
            "ranking_artifact_hash": self.ranking.artifact_hash,
            "ranked_feature_universe_hash": self.ranking.feature_universe.content_hash,
            "gene_set_collection_id": self.gene_sets.collection_id,
            "gene_set_collection_hash": self.gene_sets.content_hash,
            "null_model": self.null_model.value,
            "null_sampling": "uniform_without_replacement_conditioned_on_matched_set_size",
            "fixed_quantity": "rank_order_and_scores",
            "alternative": "two_sided_absolute_enrichment_score",
            "permutation_count": self.permutation_count,
            "random_seed": self.random_seed,
            "rng": _RNG_NAME,
            "per_term_rng_partition": "ranking_checksum_and_matched_set_size",
            "weight_exponent": self.weight_exponent,
            "multiple_testing_method": self.multiple_testing_method.value,
            "alpha": self.alpha,
            "minimum_term_size": self.minimum_term_size,
            "maximum_term_size": self.maximum_term_size,
        }


class RankedSetTerm(BaseModel):
    """One reconstructed weighted enrichment statistic and permutation p-value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gene_set_id: str = Field(min_length=1)
    gene_set_name: str = Field(min_length=1)
    description: str | None = None
    direction: RankedSetDirection
    enrichment_score: float = Field(ge=-1, le=1)
    peak_rank: int = Field(ge=1)
    leading_edge_feature_ids: tuple[str, ...]
    matched_feature_ids: tuple[str, ...]
    unmatched_member_feature_ids: tuple[str, ...]
    matched_size: int = Field(ge=1)
    original_gene_set_size: int = Field(ge=1)
    ranked_universe_size: int = Field(ge=2)
    null_mean: float
    null_standard_deviation: float = Field(ge=0)
    extreme_permutation_count: int = Field(ge=0)
    permutation_count: int = Field(ge=100)
    p_value: float = Field(gt=0, le=1)
    adjusted_p_value: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_term(self) -> Self:
        """Ensure the public term is internally coherent before reconstruction."""
        _require_clean_text(self.gene_set_id, "gene_set_id")
        _require_clean_text(self.gene_set_name, "gene_set_name")
        if self.description is not None:
            _require_clean_text(self.description, "description")
        _require_unique_clean_strings(self.matched_feature_ids, "matched_feature_ids")
        _require_unique_clean_strings(
            self.unmatched_member_feature_ids,
            "unmatched_member_feature_ids",
        )
        _require_unique_clean_strings(
            self.leading_edge_feature_ids,
            "leading_edge_feature_ids",
        )
        if len(self.matched_feature_ids) != self.matched_size:
            raise ValueError("matched_size must match matched_feature_ids")
        if (
            self.matched_size + len(self.unmatched_member_feature_ids)
            != self.original_gene_set_size
        ):
            raise ValueError("original_gene_set_size does not match membership audit")
        if self.matched_size >= self.ranked_universe_size:
            raise ValueError("tested gene sets require at least one nonmember feature")
        if not set(self.leading_edge_feature_ids).issubset(self.matched_feature_ids):
            raise ValueError("leading-edge features must be matched gene-set members")
        if not self.leading_edge_feature_ids:
            raise ValueError("leading_edge_feature_ids must be nonempty")
        if self.peak_rank > self.ranked_universe_size:
            raise ValueError("peak_rank cannot exceed the ranked universe")
        if self.direction is RankedSetDirection.POSITIVE and self.enrichment_score <= 0:
            raise ValueError("positive direction requires a positive enrichment score")
        if self.direction is RankedSetDirection.NEGATIVE and self.enrichment_score >= 0:
            raise ValueError("negative direction requires a negative enrichment score")
        if not all(
            isfinite(value)
            for value in (
                self.enrichment_score,
                self.null_mean,
                self.null_standard_deviation,
                self.p_value,
                self.adjusted_p_value,
            )
        ):
            raise ValueError("ranked-set numerical values must be finite")
        if self.extreme_permutation_count > self.permutation_count:
            raise ValueError("extreme permutation count cannot exceed permutation count")
        expected_p = (self.extreme_permutation_count + 1) / (self.permutation_count + 1)
        if self.p_value != expected_p:
            raise ValueError("p_value must use the plus-one permutation correction")
        return self


class DirectionalRankedSetResult(BaseModel):
    """Audited pre-ranked association results that cannot enter evidence fusion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_type: Literal["directional_ranked_set"] = "directional_ranked_set"
    request: DirectionalRankedSetRequest
    terms: tuple[RankedSetTerm, ...]
    multiple_testing: MultipleTestingAudit
    tested_gene_set_ids: tuple[str, ...]
    unmatched_gene_set_ids: tuple[str, ...]
    size_filtered_gene_set_ids: tuple[str, ...]
    zero_weight_gene_set_ids: tuple[str, ...]
    partially_matched_gene_set_ids: tuple[str, ...]
    matched_ranked_feature_ids: tuple[str, ...]
    unmatched_ranked_feature_ids: tuple[str, ...]
    null_model: RankedSetPermutationNull
    inference_scope: Literal[
        "pre_ranked_feature_association_not_subject_level_treatment_inference"
    ] = _INFERENCE_SCOPE
    provenance: QueryProvenance
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Recompute the entire analysis and reject forged public artifacts."""
        validated_request = DirectionalRankedSetRequest.model_validate(
            self.request.model_dump(mode="python")
        )
        validated_provenance = QueryProvenance.model_validate(
            self.provenance.model_dump(mode="python")
        )
        if validated_request != self.request:
            raise ValueError("embedded ranked-set request must satisfy its public contract")
        if validated_provenance != self.provenance:
            raise ValueError("embedded ranked-set provenance must satisfy its public contract")
        expected = _calculate_ranked_sets(validated_request)
        comparisons = (
            (self.terms, expected.terms, "terms"),
            (self.multiple_testing, expected.multiple_testing, "multiple-testing audit"),
            (self.tested_gene_set_ids, expected.tested_gene_set_ids, "tested_gene_set_ids"),
            (
                self.unmatched_gene_set_ids,
                expected.unmatched_gene_set_ids,
                "unmatched_gene_set_ids",
            ),
            (
                self.size_filtered_gene_set_ids,
                expected.size_filtered_gene_set_ids,
                "size_filtered_gene_set_ids",
            ),
            (
                self.zero_weight_gene_set_ids,
                expected.zero_weight_gene_set_ids,
                "zero_weight_gene_set_ids",
            ),
            (
                self.partially_matched_gene_set_ids,
                expected.partially_matched_gene_set_ids,
                "partially_matched_gene_set_ids",
            ),
            (
                self.matched_ranked_feature_ids,
                expected.matched_ranked_feature_ids,
                "matched_ranked_feature_ids",
            ),
            (
                self.unmatched_ranked_feature_ids,
                expected.unmatched_ranked_feature_ids,
                "unmatched_ranked_feature_ids",
            ),
            (self.warnings, expected.warnings, "warnings"),
        )
        for actual, reconstructed, field_name in comparisons:
            if actual != reconstructed:
                raise ValueError(f"{field_name} must match the reconstructed ranked-set analysis")
        if self.null_model is not self.request.null_model:
            raise ValueError("result null_model must match the ranked-set request")
        if self.provenance.provider_id != _OFFLINE_RANKED_SET_PROVIDER_ID:
            raise ValueError("provenance provider must identify the offline implementation")
        if self.provenance.provider_version != _OFFLINE_RANKED_SET_PROVIDER_VERSION:
            raise ValueError("provenance version must identify this offline implementation")
        if self.provenance.domain != self.request.ranking.feature_universe.domain:
            raise ValueError("provenance domain must match the ranked-set request")
        if self.provenance.resources != _analysis_resources(self.request):
            raise ValueError("provenance resources must match ranking and gene-set snapshots")
        if self.provenance.input_hash != self.request.input_hash:
            raise ValueError("provenance input hash must match the ranked-set request")
        if self.provenance.query_parameters != self.request.query_parameters:
            raise ValueError("provenance parameters must match the ranked-set request")
        if self.provenance.software_versions != {"numpy": np.__version__}:
            raise ValueError("offline ranked-set provenance must record the exact NumPy version")
        if not self.provenance.complete:
            raise ValueError("offline ranked-set results require complete provenance")
        if self.provenance.warnings != self.warnings:
            raise ValueError("result and provenance warnings must match")
        if self.provenance.response_checksum != expected.response_checksum:
            raise ValueError("response checksum must match the reconstructed ranked-set result")
        return self


@dataclass(frozen=True, slots=True)
class _EnrichmentStatistic:
    score: float
    direction: RankedSetDirection
    peak_rank: int
    leading_edge_feature_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RankedSetCalculation:
    terms: tuple[RankedSetTerm, ...]
    multiple_testing: MultipleTestingAudit
    tested_gene_set_ids: tuple[str, ...]
    unmatched_gene_set_ids: tuple[str, ...]
    size_filtered_gene_set_ids: tuple[str, ...]
    zero_weight_gene_set_ids: tuple[str, ...]
    partially_matched_gene_set_ids: tuple[str, ...]
    matched_ranked_feature_ids: tuple[str, ...]
    unmatched_ranked_feature_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    response_checksum: str


def run_directional_ranked_set_analysis(
    request: DirectionalRankedSetRequest,
    *,
    executed_at: datetime | None = None,
) -> DirectionalRankedSetResult:
    """Run weighted enrichment with a fixed-ranking gene-set permutation null."""
    calculation = _calculate_ranked_sets(request)
    provenance = QueryProvenance(
        provider_id=_OFFLINE_RANKED_SET_PROVIDER_ID,
        provider_version=_OFFLINE_RANKED_SET_PROVIDER_VERSION,
        resources=_analysis_resources(request),
        domain=request.ranking.feature_universe.domain,
        retrieved_at=executed_at or datetime.now(UTC),
        input_hash=request.input_hash,
        response_checksum=calculation.response_checksum,
        query_parameters=request.query_parameters,
        software_versions={"numpy": np.__version__},
        complete=True,
        warnings=calculation.warnings,
    )
    return DirectionalRankedSetResult(
        request=request,
        terms=calculation.terms,
        multiple_testing=calculation.multiple_testing,
        tested_gene_set_ids=calculation.tested_gene_set_ids,
        unmatched_gene_set_ids=calculation.unmatched_gene_set_ids,
        size_filtered_gene_set_ids=calculation.size_filtered_gene_set_ids,
        zero_weight_gene_set_ids=calculation.zero_weight_gene_set_ids,
        partially_matched_gene_set_ids=calculation.partially_matched_gene_set_ids,
        matched_ranked_feature_ids=calculation.matched_ranked_feature_ids,
        unmatched_ranked_feature_ids=calculation.unmatched_ranked_feature_ids,
        null_model=request.null_model,
        provenance=provenance,
        warnings=calculation.warnings,
    )


def _calculate_ranked_sets(request: DirectionalRankedSetRequest) -> _RankedSetCalculation:
    ranked_ids = tuple(item.feature_id for item in request.ranking.ranked_features)
    scores = np.asarray([item.score for item in request.ranking.ranked_features], dtype=float)
    rank_lookup = {identifier: index for index, identifier in enumerate(ranked_ids)}
    ranked_id_set = set(ranked_ids)
    universe_size = len(ranked_ids)
    null_cache: dict[int, npt.NDArray[np.float64]] = {}
    raw_terms: list[RankedSetTerm] = []
    unmatched_gene_sets: list[str] = []
    size_filtered_gene_sets: list[str] = []
    zero_weight_gene_sets: list[str] = []
    partially_matched_gene_sets: list[str] = []
    tested_members: set[str] = set()

    for gene_set in sorted(request.gene_sets.gene_sets, key=lambda item: item.gene_set_id):
        original_members = set(gene_set.member_feature_ids)
        matched_members = original_members.intersection(ranked_id_set)
        outside_members = original_members - ranked_id_set
        matched_size = len(matched_members)
        if matched_members and outside_members:
            partially_matched_gene_sets.append(gene_set.gene_set_id)
        if matched_size == 0:
            unmatched_gene_sets.append(gene_set.gene_set_id)
            continue
        if (
            not request.minimum_term_size <= matched_size <= request.maximum_term_size
            or matched_size >= universe_size
        ):
            size_filtered_gene_sets.append(gene_set.gene_set_id)
            continue
        matched_indices = np.asarray(
            sorted(rank_lookup[identifier] for identifier in matched_members),
            dtype=np.int64,
        )
        if _matched_weight(scores, matched_indices, request.weight_exponent) <= 0:
            zero_weight_gene_sets.append(gene_set.gene_set_id)
            continue
        statistic = _enrichment_statistic(
            ranked_ids,
            scores,
            matched_indices,
            request.weight_exponent,
        )
        if matched_size not in null_cache:
            null_cache[matched_size] = _permuted_null_scores(
                scores=scores,
                matched_size=matched_size,
                request=request,
            )
        null_scores = null_cache[matched_size]
        extreme_count = int(np.count_nonzero(np.abs(null_scores) >= abs(statistic.score)))
        p_value = (extreme_count + 1) / (request.permutation_count + 1)
        matched_in_rank_order = tuple(ranked_ids[index] for index in matched_indices)
        raw_terms.append(
            RankedSetTerm(
                gene_set_id=gene_set.gene_set_id,
                gene_set_name=gene_set.name,
                description=gene_set.description,
                direction=statistic.direction,
                enrichment_score=statistic.score,
                peak_rank=statistic.peak_rank,
                leading_edge_feature_ids=statistic.leading_edge_feature_ids,
                matched_feature_ids=matched_in_rank_order,
                unmatched_member_feature_ids=tuple(sorted(outside_members)),
                matched_size=matched_size,
                original_gene_set_size=len(original_members),
                ranked_universe_size=universe_size,
                null_mean=float(np.mean(null_scores)),
                null_standard_deviation=float(np.std(null_scores, ddof=1)),
                extreme_permutation_count=extreme_count,
                permutation_count=request.permutation_count,
                p_value=p_value,
                adjusted_p_value=p_value,
            )
        )
        tested_members.update(matched_members)

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
    unmatched_ids = tuple(sorted(unmatched_gene_sets))
    filtered_ids = tuple(sorted(size_filtered_gene_sets))
    zero_weight_ids = tuple(sorted(zero_weight_gene_sets))
    partial_ids = tuple(sorted(partially_matched_gene_sets))
    matched_features = tuple(
        identifier for identifier in ranked_ids if identifier in tested_members
    )
    unmatched_features = tuple(
        identifier for identifier in ranked_ids if identifier not in tested_members
    )
    warnings = _analysis_warnings(
        terms=terms,
        unmatched_gene_set_ids=unmatched_ids,
        size_filtered_gene_set_ids=filtered_ids,
        zero_weight_gene_set_ids=zero_weight_ids,
        partially_matched_gene_set_ids=partial_ids,
        unmatched_ranked_feature_ids=unmatched_features,
    )
    family_definition = (
        "all collection terms with ranked-universe membership, matched size in "
        f"[{request.minimum_term_size}, {request.maximum_term_size}], at least one "
        "nonmember feature, and positive total hit weight"
    )
    multiple_testing = MultipleTestingAudit(
        method=request.multiple_testing_method,
        alpha=request.alpha,
        family_size=len(terms),
        family_definition=family_definition,
    )
    response_checksum = _response_checksum(
        terms=terms,
        multiple_testing=multiple_testing,
        tested_gene_set_ids=tested_ids,
        unmatched_gene_set_ids=unmatched_ids,
        size_filtered_gene_set_ids=filtered_ids,
        zero_weight_gene_set_ids=zero_weight_ids,
        partially_matched_gene_set_ids=partial_ids,
        matched_ranked_feature_ids=matched_features,
        unmatched_ranked_feature_ids=unmatched_features,
        null_model=request.null_model,
        inference_scope=_INFERENCE_SCOPE,
        warnings=warnings,
    )
    return _RankedSetCalculation(
        terms=terms,
        multiple_testing=multiple_testing,
        tested_gene_set_ids=tested_ids,
        unmatched_gene_set_ids=unmatched_ids,
        size_filtered_gene_set_ids=filtered_ids,
        zero_weight_gene_set_ids=zero_weight_ids,
        partially_matched_gene_set_ids=partial_ids,
        matched_ranked_feature_ids=matched_features,
        unmatched_ranked_feature_ids=unmatched_features,
        warnings=warnings,
        response_checksum=response_checksum,
    )


def _matched_weight(
    scores: npt.NDArray[np.float64],
    matched_indices: npt.NDArray[np.int64],
    weight_exponent: float,
) -> float:
    if weight_exponent == 0:
        return float(len(matched_indices))
    return float(np.sum(np.abs(scores[matched_indices]) ** weight_exponent))


def _enrichment_statistic(
    ranked_ids: tuple[str, ...],
    scores: npt.NDArray[np.float64],
    matched_indices: npt.NDArray[np.int64],
    weight_exponent: float,
) -> _EnrichmentStatistic:
    universe_size = len(ranked_ids)
    hit_mask = np.zeros(universe_size, dtype=bool)
    hit_mask[matched_indices] = True
    hit_weights = np.zeros(universe_size, dtype=float)
    if weight_exponent == 0:
        hit_weights[matched_indices] = 1.0
    else:
        hit_weights[matched_indices] = np.abs(scores[matched_indices]) ** weight_exponent
    total_hit_weight = float(np.sum(hit_weights))
    if total_hit_weight <= 0:
        # An observed all-zero set is filtered before reaching this function. A
        # randomly sampled null set can still contain only zero-score features;
        # use the exponent-zero statistic for that draw rather than emitting NaN.
        hit_weights[matched_indices] = 1.0 / len(matched_indices)
    else:
        hit_weights /= total_hit_weight
    miss_decrement = 1.0 / (universe_size - len(matched_indices))
    running = np.cumsum(np.where(hit_mask, hit_weights, -miss_decrement))
    maximum_index = int(np.argmax(running))
    minimum_index = int(np.argmin(running))
    # The normalized running sum is mathematically bounded by [-1, 1], but an
    # accumulated sequence of binary fractions can overshoot by a few ulps.
    maximum = float(np.clip(running[maximum_index], -1.0, 1.0))
    minimum = float(np.clip(running[minimum_index], -1.0, 1.0))
    if maximum >= abs(minimum):
        leading_indices = matched_indices[matched_indices <= maximum_index]
        return _EnrichmentStatistic(
            score=maximum,
            direction=RankedSetDirection.POSITIVE,
            peak_rank=maximum_index + 1,
            leading_edge_feature_ids=tuple(ranked_ids[index] for index in leading_indices),
        )
    leading_indices = matched_indices[matched_indices >= minimum_index]
    return _EnrichmentStatistic(
        score=minimum,
        direction=RankedSetDirection.NEGATIVE,
        peak_rank=minimum_index + 1,
        leading_edge_feature_ids=tuple(ranked_ids[index] for index in leading_indices),
    )


def _permuted_null_scores(
    *,
    scores: npt.NDArray[np.float64],
    matched_size: int,
    request: DirectionalRankedSetRequest,
) -> npt.NDArray[np.float64]:
    seed_digest = canonical_sha256(
        {
            "schema": "rejuvenationkit.ranked-set-permutation-seed/v1",
            "random_seed": request.random_seed,
            "normalized_ranking_sha256": request.ranking.normalized_ranking_sha256,
            "matched_size": matched_size,
            "permutation_count": request.permutation_count,
            "weight_exponent": request.weight_exponent,
            "rng": _RNG_NAME,
        }
    )
    generator = np.random.Generator(np.random.PCG64(int(seed_digest[:16], 16)))
    ranked_ids = tuple(item.feature_id for item in request.ranking.ranked_features)
    values = np.empty(request.permutation_count, dtype=float)
    for index in range(request.permutation_count):
        sampled = np.sort(generator.choice(len(ranked_ids), size=matched_size, replace=False))
        values[index] = _enrichment_statistic(
            ranked_ids,
            scores,
            sampled,
            request.weight_exponent,
        ).score
    values.flags.writeable = False
    return values


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


def _analysis_resources(
    request: DirectionalRankedSetRequest,
) -> tuple[ResourceSnapshot, ...]:
    by_snapshot_id = {
        item.snapshot_id: item
        for item in (request.ranking.source_resource, request.gene_sets.resource)
    }
    return tuple(by_snapshot_id[key] for key in sorted(by_snapshot_id))


def _analysis_warnings(
    *,
    terms: tuple[RankedSetTerm, ...],
    unmatched_gene_set_ids: tuple[str, ...],
    size_filtered_gene_set_ids: tuple[str, ...],
    zero_weight_gene_set_ids: tuple[str, ...],
    partially_matched_gene_set_ids: tuple[str, ...],
    unmatched_ranked_feature_ids: tuple[str, ...],
) -> tuple[str, ...]:
    warnings = [
        "gene_set_membership_permutation_is_not_subject_level_inference",
        "feature_correlation_is_not_preserved_by_gene_set_membership_permutation",
    ]
    if unmatched_ranked_feature_ids:
        warnings.append("ranked_features_absent_from_tested_gene_sets")
    if unmatched_gene_set_ids:
        warnings.append("gene_sets_without_ranked_members")
    if size_filtered_gene_set_ids:
        warnings.append("gene_sets_filtered_by_ranked_term_size_or_degenerate_complement")
    if zero_weight_gene_set_ids:
        warnings.append("gene_sets_with_zero_weight_under_requested_exponent")
    if partially_matched_gene_set_ids:
        warnings.append("gene_sets_partially_outside_ranked_universe")
    if not terms:
        warnings.append("no_gene_sets_tested")
    return tuple(warnings)


def _response_checksum(
    *,
    terms: tuple[RankedSetTerm, ...],
    multiple_testing: MultipleTestingAudit,
    tested_gene_set_ids: tuple[str, ...],
    unmatched_gene_set_ids: tuple[str, ...],
    size_filtered_gene_set_ids: tuple[str, ...],
    zero_weight_gene_set_ids: tuple[str, ...],
    partially_matched_gene_set_ids: tuple[str, ...],
    matched_ranked_feature_ids: tuple[str, ...],
    unmatched_ranked_feature_ids: tuple[str, ...],
    null_model: RankedSetPermutationNull,
    inference_scope: str,
    warnings: tuple[str, ...],
) -> str:
    return canonical_sha256(
        {
            "terms": [item.model_dump(mode="json") for item in terms],
            "multiple_testing": multiple_testing.model_dump(mode="json"),
            "tested_gene_set_ids": tested_gene_set_ids,
            "unmatched_gene_set_ids": unmatched_gene_set_ids,
            "size_filtered_gene_set_ids": size_filtered_gene_set_ids,
            "zero_weight_gene_set_ids": zero_weight_gene_set_ids,
            "partially_matched_gene_set_ids": partially_matched_gene_set_ids,
            "matched_ranked_feature_ids": matched_ranked_feature_ids,
            "unmatched_ranked_feature_ids": unmatched_ranked_feature_ids,
            "null_model": null_model.value,
            "inference_scope": inference_scope,
            "warnings": warnings,
        }
    )


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_unique_clean_strings(values: Sequence[str], field_name: str) -> None:
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
