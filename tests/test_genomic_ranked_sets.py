from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.enrichment import (
    FunctionalGeneSet,
    GeneSetCollection,
    MultipleTestingMethod,
)
from rejuvenationkit.genomics.ranked_sets import (
    DirectionalRankedSetRequest,
    DirectionalRankedSetResult,
    RankedFeature,
    RankedFeatureUniverse,
    RankedSetDirection,
    RankedSetPermutationNull,
    build_ranked_feature_universe,
    run_directional_ranked_set_analysis,
)
from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    ResourceSnapshot,
)
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType


def _domain(**updates: object) -> FeatureDomain:
    values: dict[str, object] = {
        "species_taxon_id": 9615,
        "feature_type": GenomicFeatureType.GENE,
        "namespace": FeatureNamespace.ENSEMBL,
        "genome_assembly": "CanFam4",
    }
    values.update(updates)
    return FeatureDomain(**values)


def _snapshot(resource_id: str, response_character: str) -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id="offline-fixture",
        resource_id=resource_id,
        resource_release="2026-08-15",
        retrieved_at=datetime(2026, 8, 15, tzinfo=UTC),
        response_sha256=response_character * 64,
        source_uri=f"https://example.org/{resource_id}.tsv",
        license_id="CC0-1.0",
    )


def _ranking(
    scores: dict[str, float] | None = None,
    *,
    domain: FeatureDomain | None = None,
    source: ResourceSnapshot | None = None,
    executed_at: datetime | None = None,
) -> RankedFeatureUniverse:
    score_values = scores or {
        "g1": 4.0,
        "g2": 3.0,
        "g3": 2.0,
        "g4": 1.0,
        "g5": -1.0,
        "g6": -2.0,
        "g7": -3.0,
        "g8": -4.0,
    }
    feature_domain = domain or _domain()
    resource = source or _snapshot("ranked-effects", "a")
    universe = FeatureCollection(
        collection_id="all-tested-features",
        domain=feature_domain,
        feature_ids=tuple(score_values),
        source_snapshot_id=resource.snapshot_id,
    )
    return build_ranked_feature_universe(
        score_values,
        ranking_id="rapamycin-ranked-effects",
        ranking_method="prespecified_wald_statistic",
        ranking_metric="signed_wald_statistic",
        higher_score_interpretation="greater expression after rapamycin",
        feature_universe=universe,
        source_resource=resource,
        provider_id="fixture.differential_expression",
        provider_version="1.0.0",
        executed_at=executed_at or datetime(2026, 8, 15, 1, tzinfo=UTC),
        software_versions={"fixture": "1.0"},
    )


def _gene_set(
    gene_set_id: str,
    members: tuple[str, ...],
    *,
    domain: FeatureDomain,
    resource: ResourceSnapshot,
) -> FunctionalGeneSet:
    return FunctionalGeneSet(
        gene_set_id=gene_set_id,
        name=f"Pathway {gene_set_id}",
        description=f"Fixture pathway {gene_set_id}",
        domain=domain,
        member_feature_ids=members,
        resource=resource,
    )


def _collection(
    definitions: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    *,
    domain: FeatureDomain | None = None,
    resource: ResourceSnapshot | None = None,
) -> GeneSetCollection:
    feature_domain = domain or _domain()
    source = resource or _snapshot("gene-sets", "b")
    rows = definitions or (
        ("TOP", ("g1", "g2")),
        ("BOTTOM", ("g7", "g8")),
        ("SPREAD", ("g1", "g4", "g5", "g8")),
    )
    return GeneSetCollection(
        collection_id="fixture-pathways",
        name="Fixture pathways",
        domain=feature_domain,
        gene_sets=tuple(
            _gene_set(identifier, members, domain=feature_domain, resource=source)
            for identifier, members in rows
        ),
        resource=source,
    )


def _request(
    *,
    ranking: RankedFeatureUniverse | None = None,
    gene_sets: GeneSetCollection | None = None,
    permutation_count: int = 400,
    random_seed: int = 29,
    weight_exponent: float = 1.0,
    method: MultipleTestingMethod = MultipleTestingMethod.BENJAMINI_HOCHBERG,
    minimum_term_size: int = 2,
    maximum_term_size: int = 20,
) -> DirectionalRankedSetRequest:
    return DirectionalRankedSetRequest(
        ranking=ranking or _ranking(),
        gene_sets=gene_sets or _collection(),
        permutation_count=permutation_count,
        random_seed=random_seed,
        weight_exponent=weight_exponent,
        multiple_testing_method=method,
        minimum_term_size=minimum_term_size,
        maximum_term_size=maximum_term_size,
    )


def _revalidate_result(
    result: DirectionalRankedSetResult,
    **updates: object,
) -> DirectionalRankedSetResult:
    payload = result.model_dump(mode="python")
    payload.update(updates)
    return DirectionalRankedSetResult.model_validate(payload)


def test_weighted_running_sum_matches_a_simple_reference_calculation() -> None:
    ranking = _ranking({"g1": 4.0, "g2": 3.0, "g3": 2.0, "g4": 1.0, "g5": 0.0})
    collection = _collection(
        (("REFERENCE", ("g1", "g4")),),
        domain=ranking.feature_universe.domain,
    )

    result = run_directional_ranked_set_analysis(
        _request(ranking=ranking, gene_sets=collection, permutation_count=200)
    )
    term = result.terms[0]

    # Hit increments are 4/5 and 1/5; each of three misses decrements 1/3.
    assert term.enrichment_score == pytest.approx(0.8)
    assert term.direction is RankedSetDirection.POSITIVE
    assert term.peak_rank == 1
    assert term.leading_edge_feature_ids == ("g1",)
    assert term.matched_feature_ids == ("g1", "g4")


def test_gene_set_permutation_null_is_fixed_rank_two_sided_and_nonzero() -> None:
    scores = {f"g{index:02d}": float(11 - index) for index in range(1, 22)}
    ranking = _ranking(scores)
    spread_members = tuple(f"g{index:02d}" for index in (1, 5, 9, 13, 17, 21))
    collection = _collection(
        (("SPREAD", spread_members),),
        domain=ranking.feature_universe.domain,
    )

    result = run_directional_ranked_set_analysis(
        _request(
            ranking=ranking,
            gene_sets=collection,
            permutation_count=1_000,
            random_seed=71,
        )
    )
    term = result.terms[0]

    assert result.null_model is RankedSetPermutationNull.GENE_SET_MEMBERSHIP
    assert result.inference_scope.endswith("not_subject_level_treatment_inference")
    assert "fixed_pre_ranked_universe" in result.null_model.value
    assert term.p_value >= 1 / 1_001
    assert term.p_value > 0.1
    assert abs(term.null_mean) < 0.15
    assert term.null_standard_deviation > 0
    assert "gene_set_membership_permutation_is_not_subject_level_inference" in result.warnings


def test_local_bh_multiplicity_uses_only_the_complete_tested_family() -> None:
    result = run_directional_ranked_set_analysis(_request(permutation_count=800))
    ordered = sorted(
        ((term.gene_set_id, term.p_value) for term in result.terms),
        key=lambda item: (item[1], item[0]),
    )
    expected: dict[str, float] = {}
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        identifier, p_value = ordered[index]
        running = min(running, p_value * len(ordered) / (index + 1))
        expected[identifier] = min(1.0, running)

    assert result.multiple_testing.family_size == 3
    assert result.multiple_testing.method is MultipleTestingMethod.BENJAMINI_HOCHBERG
    assert {term.gene_set_id for term in result.terms} == {"TOP", "BOTTOM", "SPREAD"}
    for term in result.terms:
        assert term.adjusted_p_value == pytest.approx(expected[term.gene_set_id])


def test_complete_membership_and_filtering_audit_partitions_gene_sets() -> None:
    ranking = _ranking(
        {
            "g1": 4.0,
            "g2": 3.0,
            "g3": 2.0,
            "g4": 1.0,
            "g5": -1.0,
            "g6": -2.0,
            "g7": 0.0,
            "g8": 0.0,
        }
    )
    collection = _collection(
        (
            ("TEST", ("g1", "g2", "outside-1")),
            ("SMALL", ("g3",)),
            ("ABSENT", ("outside-1", "outside-2")),
            ("ALL", tuple(f"g{index}" for index in range(1, 9))),
            ("ZERO", ("g7", "g8")),
        ),
        domain=ranking.feature_universe.domain,
    )

    result = run_directional_ranked_set_analysis(
        _request(ranking=ranking, gene_sets=collection, permutation_count=200)
    )

    assert result.tested_gene_set_ids == ("TEST",)
    assert result.unmatched_gene_set_ids == ("ABSENT",)
    assert result.size_filtered_gene_set_ids == ("ALL", "SMALL")
    assert result.zero_weight_gene_set_ids == ("ZERO",)
    assert result.partially_matched_gene_set_ids == ("TEST",)
    assert result.matched_ranked_feature_ids == ("g1", "g2")
    assert result.unmatched_ranked_feature_ids == (
        "g3",
        "g4",
        "g7",
        "g8",
        "g5",
        "g6",
    )
    assert result.terms[0].unmatched_member_feature_ids == ("outside-1",)
    assert "gene_sets_without_ranked_members" in result.warnings
    assert "gene_sets_with_zero_weight_under_requested_exponent" in result.warnings


def test_input_order_does_not_change_ranking_identity_results_or_null_draws() -> None:
    timestamp = datetime(2026, 8, 15, 2, tzinfo=UTC)
    original_scores = {
        "g1": 4.0,
        "g2": 3.0,
        "g3": 2.0,
        "g4": 1.0,
        "g5": -1.0,
        "g6": -2.0,
        "g7": -3.0,
        "g8": -4.0,
    }
    source = _snapshot("ranked-effects", "a")
    first_ranking = _ranking(original_scores, source=source, executed_at=timestamp)
    second_ranking = _ranking(
        dict(reversed(tuple(original_scores.items()))),
        source=source,
        executed_at=timestamp,
    )
    first_collection = _collection()
    second_collection = _collection(
        tuple(
            (item.gene_set_id, tuple(reversed(item.member_feature_ids)))
            for item in reversed(first_collection.gene_sets)
        )
    )

    first = run_directional_ranked_set_analysis(
        _request(ranking=first_ranking, gene_sets=first_collection),
        executed_at=timestamp,
    )
    second = run_directional_ranked_set_analysis(
        _request(ranking=second_ranking, gene_sets=second_collection),
        executed_at=timestamp,
    )

    assert first_ranking.normalized_ranking_sha256 == second_ranking.normalized_ranking_sha256
    assert first_ranking.artifact_hash == second_ranking.artifact_hash
    assert first.terms == second.terms
    assert first.provenance.query_hash == second.provenance.query_hash
    assert first.provenance.response_checksum == second.provenance.response_checksum


def test_ranking_requires_exact_coverage_order_domain_resource_and_provenance() -> None:
    baseline = _ranking()
    scores = {item.feature_id: item.score for item in baseline.ranked_features}
    with pytest.raises(ValueError, match="exactly cover"):
        build_ranked_feature_universe(
            {key: value for key, value in scores.items() if key != "g8"},
            ranking_id=baseline.ranking_id,
            ranking_method=baseline.ranking_method,
            ranking_metric=baseline.ranking_metric,
            higher_score_interpretation=baseline.higher_score_interpretation,
            feature_universe=baseline.feature_universe,
            source_resource=baseline.source_resource,
            provider_id="fixture",
            provider_version="1.0.0",
        )
    with pytest.raises(TypeError, match="real numbers"):
        build_ranked_feature_universe(
            {**scores, "g1": True},  # type: ignore[dict-item]
            ranking_id=baseline.ranking_id,
            ranking_method=baseline.ranking_method,
            ranking_metric=baseline.ranking_metric,
            higher_score_interpretation=baseline.higher_score_interpretation,
            feature_universe=baseline.feature_universe,
            source_resource=baseline.source_resource,
            provider_id="fixture",
            provider_version="1.0.0",
        )
    with pytest.raises(ValidationError, match="finite"):
        RankedFeature(feature_id="g1", score=float("nan"))

    reversed_payload = baseline.model_dump(mode="python")
    reversed_payload["ranked_features"] = tuple(reversed(baseline.ranked_features))
    with pytest.raises(ValidationError, match="score-descending"):
        RankedFeatureUniverse.model_validate(reversed_payload)

    checksum_payload = baseline.model_dump(mode="python")
    checksum_payload["normalized_ranking_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="normalized_ranking_sha256"):
        RankedFeatureUniverse.model_validate(checksum_payload)

    provenance_payload = baseline.model_dump(mode="python")
    provenance_payload["provenance"] = baseline.provenance.model_copy(
        update={"response_checksum": "f" * 64}
    )
    with pytest.raises(ValidationError, match="response checksum"):
        RankedFeatureUniverse.model_validate(provenance_payload)

    legacy_domain = _domain(namespace=FeatureNamespace.HGNC, genome_assembly=None)
    with pytest.raises(ValidationError, match="explicit identifier namespace"):
        _ranking(domain=legacy_domain)


def test_request_rejects_domain_mismatch_and_degenerate_parameterization() -> None:
    with pytest.raises(ValidationError, match="domains must match exactly"):
        _request(gene_sets=_collection(domain=_domain(species_taxon_id=10090)))
    with pytest.raises(ValidationError, match="at least minimum"):
        _request(minimum_term_size=5, maximum_term_size=2)
    with pytest.raises(ValidationError, match="greater than or equal to 100"):
        _request(permutation_count=99)


def test_result_is_immutable_not_fusible_and_provenance_bound() -> None:
    request = _request(permutation_count=200)
    result = run_directional_ranked_set_analysis(
        request,
        executed_at=datetime(2026, 8, 15, 3, tzinfo=UTC),
    )

    assert result.fusion_eligibility == "not_fusible"
    assert not hasattr(result, "to_evidence")
    assert result.provenance.input_hash == request.input_hash
    assert result.provenance.query_parameters == request.query_parameters
    assert set(result.provenance.resources) == {
        request.ranking.source_resource,
        request.gene_sets.resource,
    }
    assert result.provenance.response_checksum is not None
    assert DirectionalRankedSetResult.model_validate(result.model_dump(mode="python")) == result
    with pytest.raises(ValidationError, match="frozen"):
        result.warnings = ()  # type: ignore[misc]


def test_public_result_rejects_forged_terms_audits_warnings_and_checksum() -> None:
    result = run_directional_ranked_set_analysis(_request(permutation_count=200))
    first = result.terms[0]
    forged_term = first.model_copy(update={"enrichment_score": first.enrichment_score * 0.9})
    with pytest.raises(ValidationError, match="terms"):
        _revalidate_result(result, terms=(forged_term, *result.terms[1:]))
    with pytest.raises(ValidationError, match="tested_gene_set_ids"):
        _revalidate_result(result, tested_gene_set_ids=("forged",))
    forged_audit = result.multiple_testing.model_copy(
        update={"family_definition": "only favorable pathways"}
    )
    with pytest.raises(ValidationError, match="multiple-testing audit"):
        _revalidate_result(result, multiple_testing=forged_audit)

    reduced_warnings = result.warnings[1:]
    provenance_with_reduced_warnings = result.provenance.model_copy(
        update={"warnings": reduced_warnings}
    )
    with pytest.raises(ValidationError, match="warnings"):
        _revalidate_result(
            result,
            warnings=reduced_warnings,
            provenance=provenance_with_reduced_warnings,
        )
    forged_provenance = result.provenance.model_copy(update={"response_checksum": "f" * 64})
    with pytest.raises(ValidationError, match="response checksum"):
        _revalidate_result(result, provenance=forged_provenance)


def test_plus_one_permutation_p_value_cannot_be_forged_or_equal_zero() -> None:
    result = run_directional_ranked_set_analysis(_request(permutation_count=200))
    first = result.terms[0]
    assert first.p_value > 0
    forged = first.model_dump(mode="python")
    forged["p_value"] = max(first.p_value / 2, 1e-12)

    with pytest.raises(ValidationError, match="plus-one"):
        type(first).model_validate(forged)
