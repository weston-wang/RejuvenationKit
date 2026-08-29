from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from itertools import product

import numpy as np
import pytest
from pydantic import ValidationError

from rejuvenationkit.combinations import (
    AssignmentMechanism,
    CovarianceEstimator,
    EndpointWeighting,
    ExtraInterventionPolicy,
    FactorialCombinationAnalysis,
    FactorialCombinationConfig,
    InteractionEstimate,
    MissingEndpointPolicy,
    MultiplicityMethod,
    TwoByTwoDesignConfig,
    balanced_factorial_allocation,
    recommend_two_by_two_design,
)
from rejuvenationkit.endpoints import (
    EndpointExclusion,
    SubjectEndpoint,
    SubjectEndpointBatch,
    study_artifact_hash,
)
from rejuvenationkit.evidence import EffectDirection, Estimand
from rejuvenationkit.schemas import Study, Subject

START = datetime(2026, 1, 1, tzinfo=UTC)
ESTIMAND = Estimand(
    name="frailty_change",
    unit="score",
    direction=EffectDirection.LOWER_IS_BETTER,
    population="factorial-study",
    time_contrast="month-6-minus-baseline",
)


def factorial_fixture(
    *,
    interventions: tuple[str, ...] = ("rapamycin", "senolytic"),
    per_cell: int = 8,
    interaction: float = -1.5,
    observational: bool = False,
) -> tuple[Study, SubjectEndpointBatch]:
    subjects: list[Subject] = []
    endpoints: list[SubjectEndpoint] = []
    noise = np.linspace(-0.35, 0.35, per_cell)
    for bits in product((0, 1), repeat=len(interventions)):
        active = tuple(name for name, enabled in zip(interventions, bits, strict=True) if enabled)
        for index in range(per_cell):
            subject_id = "-".join((*map(str, bits), f"{index:02d}"))
            subjects.append(
                Subject(
                    subject_id=subject_id,
                    cohort="observational" if observational else "randomized",
                    interventions=active,
                    attributes={"age": 7.0 + 0.1 * index},
                )
            )
            value = 1.0 - 2.0 * bits[0] - 1.0 * bits[1]
            if len(bits) == 2:
                value += interaction * bits[0] * bits[1]
            endpoints.append(
                SubjectEndpoint(
                    subject_id=subject_id,
                    estimand=ESTIMAND,
                    estimate=float(value + noise[index]),
                    standard_error=0.2 + 0.01 * index,
                    baseline_timestamp=START,
                    endpoint_timestamp=START + timedelta(days=180),
                    baseline_estimate=5.0 + 0.05 * index,
                    provenance_id=f"endpoint:{subject_id}",
                )
            )
    study = Study(study_id="factorial", subjects=tuple(subjects), observations=())
    batch = SubjectEndpointBatch(
        study_id=study.study_id,
        batch_id="frailty-v1",
        estimand=ESTIMAND,
        endpoints=tuple(endpoints),
        source_artifact_hash=study_artifact_hash(study),
    )
    return study, batch


def config(**updates: object) -> FactorialCombinationConfig:
    values: dict[str, object] = {
        "interventions": ("rapamycin", "senolytic"),
        "assignment_mechanism": AssignmentMechanism.RANDOMIZED,
        "minimum_cell_size": 4,
        "covariance_estimator": CovarianceEstimator.HC3,
    }
    values.update(updates)
    return FactorialCombinationConfig(**values)


def test_two_by_two_analysis_recovers_declared_interaction() -> None:
    study, endpoints = factorial_fixture(interaction=-1.5)
    analysis = FactorialCombinationAnalysis(config())
    report = analysis.analyze(study, endpoints=endpoints)

    assert len(report.interactions) == 1
    estimate = report.interactions[0]
    assert estimate.interventions == ("rapamycin", "senolytic")
    assert estimate.interaction == pytest.approx(-1.5)
    assert estimate.confidence_interval is not None
    assert estimate.confidence_interval[1] < 0
    assert estimate.p_value is not None and estimate.p_value < 0.001
    assert estimate.adjusted_p_value == estimate.p_value
    assert sum(item.coefficient for item in estimate.contrast_weights) == 0
    assert report.diagnostics.observed_cells == 4
    assert report.diagnostics.residual_degrees_of_freedom == 28
    assert report.study_artifact_hash == study_artifact_hash(study)
    assert report.endpoint_source_artifact_hash == endpoints.source_artifact_hash
    assert (
        report.artifact_hash == report.model_validate_json(report.model_dump_json()).artifact_hash
    )
    assert len(report.interactions_frame()) == 1
    assert len(report.cells_frame()) == 4


def test_combination_report_rejects_cross_field_forgery() -> None:
    study, endpoints = factorial_fixture(interaction=-1.5)
    report = FactorialCombinationAnalysis(config()).analyze(study, endpoints=endpoints)

    assignment_payload = deepcopy(report.model_dump(mode="python"))
    assignment_payload["interactions"][0]["assignment_mechanism"] = (
        AssignmentMechanism.OBSERVATIONAL
    )
    with pytest.raises(ValidationError, match="interaction assignment"):
        type(report).model_validate(assignment_payload)

    cell_payload = deepcopy(report.model_dump(mode="python"))
    cell_payload["cells"][0]["assigned_subjects"] += 1
    with pytest.raises(ValidationError, match="cell counts"):
        type(report).model_validate(cell_payload)

    coefficient_payload = deepcopy(report.model_dump(mode="python"))
    coefficient_payload["coefficients"][1]["estimate"] += 0.25
    with pytest.raises(ValidationError, match="coefficient statistic"):
        type(report).model_validate(coefficient_payload)

    tiny_probability_payload = deepcopy(report.model_dump(mode="python"))
    assert tiny_probability_payload["coefficients"][1]["p_value"] < 1e-12
    tiny_probability_payload["coefficients"][1]["p_value"] = 0.0
    with pytest.raises(ValidationError, match="coefficient p-value"):
        type(report).model_validate(tiny_probability_payload)

    interaction_payload = deepcopy(report.model_dump(mode="python"))
    interaction_payload["interactions"][0]["adjusted_p_value"] = 0.5
    with pytest.raises(ValidationError, match="adjusted p-value"):
        type(report).model_validate(interaction_payload)

    diagnostic_payload = deepcopy(report.model_dump(mode="python"))
    diagnostic_payload["diagnostics"]["required_cells"] = 8
    with pytest.raises(ValidationError, match="required-cell count"):
        type(report).model_validate(diagnostic_payload)

    forged_exclusion_payload = deepcopy(report.model_dump(mode="python"))
    forged_exclusion_payload["diagnostics"]["excluded_subject_ids"] = ("ghost",)
    forged_exclusion_payload["warnings"] = ("subjects_excluded_from_factorial_analysis",)
    with pytest.raises(ValidationError, match="exactly match design diagnostics"):
        type(report).model_validate(forged_exclusion_payload)

    source_payload = deepcopy(report.model_dump(mode="python"))
    source_payload["endpoint_source_artifact_hash"] = "f" * 64
    changed_source_report = type(report).model_validate(source_payload)
    assert changed_source_report.artifact_hash != report.artifact_hash

    changed_study = study.model_copy(update={"metadata": {"changed": True}})
    changed_report = FactorialCombinationAnalysis(config()).analyze(
        changed_study,
        endpoints=endpoints,
    )
    assert changed_report.study_id == report.study_id
    assert changed_report.study_artifact_hash != report.study_artifact_hash
    assert changed_report.artifact_hash != report.artifact_hash


def test_additive_null_and_observational_warning_are_not_called_synergy() -> None:
    study, endpoints = factorial_fixture(interaction=0.0, observational=True)
    report = FactorialCombinationAnalysis(
        config(assignment_mechanism=AssignmentMechanism.OBSERVATIONAL)
    ).analyze(study, endpoints=endpoints)

    estimate = report.interactions[0]
    assert estimate.interaction == pytest.approx(0.0, abs=1e-12)
    assert estimate.reference_model == "additive_on_declared_outcome_scale"
    assert "observational_assignment_noncausal" in estimate.quality_flags
    assert "observational_assignment_noncausal" in report.warnings


def test_interaction_estimate_preserves_legacy_required_fields() -> None:
    item = InteractionEstimate(
        interventions=("a", "b"),
        outcome="biological_age_delta",
        interaction=-1.2,
        standard_error=0.3,
        reference_model="additive",
    )
    assert item.interaction == -1.2
    assert item.estimand is None


def test_factorial_cells_and_subject_units_fail_closed() -> None:
    study, endpoints = factorial_fixture()
    incomplete = endpoints.model_copy(update={"endpoints": endpoints.endpoints[:-8]})
    with pytest.raises(ValueError, match="factorial cell"):
        FactorialCombinationAnalysis(config()).analyze(study, endpoints=incomplete)

    missing_one = endpoints.model_copy(update={"endpoints": endpoints.endpoints[1:]})
    with pytest.raises(ValueError, match="missing endpoints"):
        FactorialCombinationAnalysis(
            config(missing_endpoint_policy=MissingEndpointPolicy.ERROR)
        ).analyze(study, endpoints=missing_one)

    extra_subject = study.subjects[0].model_copy(
        update={"interventions": (*study.subjects[0].interventions, "exercise")}
    )
    changed_study = study.model_copy(update={"subjects": (extra_subject, *study.subjects[1:])})
    with pytest.raises(ValueError, match="undeclared"):
        FactorialCombinationAnalysis(config()).analyze(changed_study, endpoints=endpoints)

    excluded = FactorialCombinationAnalysis(
        config(extra_intervention_policy=ExtraInterventionPolicy.EXCLUDE)
    ).analyze(changed_study, endpoints=endpoints)
    assert extra_subject.subject_id in excluded.diagnostics.excluded_subject_ids
    assert extra_subject.subject_id in excluded.cells[0].excluded_subject_ids

    unknown_exclusion = endpoints.model_copy(
        update={"excluded": (EndpointExclusion(subject_id="ghost", reason="missing_followup"),)}
    )
    with pytest.raises(ValueError, match="unknown subjects"):
        FactorialCombinationAnalysis(config()).analyze(
            study,
            endpoints=unknown_exclusion,
        )


def test_covariate_rank_and_endpoint_uncertainty_policies_are_audited() -> None:
    study, endpoints = factorial_fixture()
    with pytest.raises(ValueError, match="rank deficient"):
        FactorialCombinationAnalysis(config(covariates=("constant",))).analyze(
            study.model_copy(
                update={
                    "subjects": tuple(
                        item.model_copy(update={"attributes": {"constant": 1.0}})
                        for item in study.subjects
                    )
                }
            ),
            endpoints=endpoints,
        )

    weighted = FactorialCombinationAnalysis(
        config(endpoint_weighting=EndpointWeighting.INVERSE_VARIANCE_REQUIRED)
    ).analyze(study, endpoints=endpoints)
    assert (
        weighted.diagnostics.endpoint_weighting_used is EndpointWeighting.INVERSE_VARIANCE_REQUIRED
    )

    partly_missing = endpoints.model_copy(
        update={
            "endpoints": (
                endpoints.endpoints[0].model_copy(update={"standard_error": None}),
                *endpoints.endpoints[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="every endpoint"):
        FactorialCombinationAnalysis(
            config(endpoint_weighting=EndpointWeighting.INVERSE_VARIANCE_REQUIRED)
        ).analyze(study, endpoints=partly_missing)


def test_three_factor_family_and_multiplicity_are_explicit() -> None:
    interventions = ("a", "b", "c")
    study, endpoints = factorial_fixture(interventions=interventions, per_cell=6)
    report = FactorialCombinationAnalysis(
        FactorialCombinationConfig(
            interventions=interventions,
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
            minimum_cell_size=4,
            multiplicity_method=MultiplicityMethod.BONFERRONI,
        )
    ).analyze(study, endpoints=endpoints)

    assert {item.interventions for item in report.interactions} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "b", "c"),
    }
    assert all(
        item.adjusted_p_value is not None
        and item.p_value is not None
        and item.adjusted_p_value >= item.p_value
        for item in report.interactions
    )


def test_design_planner_and_balanced_allocation_are_monotone_and_deterministic() -> None:
    base = recommend_two_by_two_design(
        TwoByTwoDesignConfig(
            interventions=("rapamycin", "exercise"),
            target_interaction=1.0,
            residual_standard_deviation=2.0,
        )
    )
    smaller_effect = recommend_two_by_two_design(
        TwoByTwoDesignConfig(
            interventions=("rapamycin", "exercise"),
            target_interaction=0.5,
            residual_standard_deviation=2.0,
            expected_dropout_fraction=0.2,
        )
    )
    assert smaller_effect.analyzable_subjects_per_cell > base.analyzable_subjects_per_cell
    assert smaller_effect.enrolled_subjects_per_cell > smaller_effect.analyzable_subjects_per_cell
    assert "not_for_survival_or_repeated_measure_endpoints" in base.assumptions

    subjects = tuple(f"dog-{index}" for index in range(17))
    first = balanced_factorial_allocation(
        subjects, interventions=("rapamycin", "exercise"), random_seed=4
    )
    second = balanced_factorial_allocation(
        subjects, interventions=("rapamycin", "exercise"), random_seed=4
    )
    assert first == second
    counts: dict[tuple[str, ...], int] = {}
    for item in first:
        counts[item.active_interventions] = counts.get(item.active_interventions, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_configuration_rejects_duplicate_or_impossible_declarations() -> None:
    with pytest.raises(ValidationError, match="assignment_mechanism"):
        FactorialCombinationConfig(interventions=("a", "b"))
    with pytest.raises(ValidationError, match="unique"):
        FactorialCombinationConfig(
            interventions=("a", "a"),
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        FactorialCombinationConfig(
            interventions=("a", "b"),
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
            maximum_interaction_order=3,
        )
    with pytest.raises(ValidationError, match="ambiguous factorial design terms"):
        FactorialCombinationConfig(
            interventions=("a", "b", "a:b"),
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        )
    with pytest.raises(ValidationError, match="nonzero"):
        TwoByTwoDesignConfig(
            interventions=("a", "b"),
            target_interaction=0,
            residual_standard_deviation=1,
        )


def test_estimate_requires_explicit_subject_endpoint_batch() -> None:
    study, _ = factorial_fixture()
    with pytest.raises(ValueError, match="SubjectEndpointBatch"):
        FactorialCombinationAnalysis(config()).estimate(study, outcome=ESTIMAND.name)
