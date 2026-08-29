"""Demonstrate Phase 4 on a synthetic randomized canine 2 x 2 study.

Every subject, endpoint, uncertainty, and effect in this example is synthetic.
The fitted interaction is a departure from additivity on the declared outcome
scale; it is not evidence of efficacy or automatic evidence of biological synergy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from rejuvenationkit.combinations import (
    AssignmentMechanism,
    CovarianceEstimator,
    EndpointWeighting,
    FactorialCombinationAnalysis,
    FactorialCombinationConfig,
    MultiplicityMethod,
    TwoByTwoDesignConfig,
    balanced_factorial_allocation,
    recommend_two_by_two_design,
)
from rejuvenationkit.endpoints import (
    SubjectEndpoint,
    SubjectEndpointBatch,
    study_artifact_hash,
)
from rejuvenationkit.evidence import EffectDirection, Estimand
from rejuvenationkit.schemas import Study, Subject

START = datetime(2026, 1, 12, tzinfo=UTC)
INTERVENTIONS = ("rapamycin", "senolytic")
SUBJECTS_PER_CELL = 18
ESTIMAND = Estimand(
    name="month_6_frailty",
    unit="canine_frailty_index_points",
    direction=EffectDirection.LOWER_IS_BETTER,
    population="synthetic randomized older companion dogs",
    time_contrast="month 6 follow-up adjusted for baseline frailty",
)


def build_synthetic_factorial_study() -> tuple[Study, SubjectEndpointBatch]:
    """Create balanced assignments and one baseline-aware endpoint per dog."""
    random = np.random.default_rng(20260820)
    subjects: list[Subject] = []
    endpoints: list[SubjectEndpoint] = []
    subject_ids = tuple(f"dog-{index:03d}" for index in range(4 * SUBJECTS_PER_CELL))
    allocations = balanced_factorial_allocation(
        subject_ids,
        interventions=INTERVENTIONS,
        random_seed=9173,
    )

    for index, allocation in enumerate(allocations):
        active = allocation.active_interventions
        rapamycin = int(INTERVENTIONS[0] in active)
        senolytic = int(INTERVENTIONS[1] in active)
        baseline = float(random.normal(6.0, 0.75))
        endpoint_standard_error = float(0.13 + 0.015 * (index % 4))
        heteroskedastic_scale = 0.28 + 0.05 * (rapamycin + senolytic)
        endpoint = (
            5.4
            + 0.68 * (baseline - 6.0)
            - 0.55 * rapamycin
            - 0.30 * senolytic
            - 0.48 * rapamycin * senolytic
            + random.normal(0.0, heteroskedastic_scale)
        )
        subjects.append(
            Subject(
                subject_id=allocation.subject_id,
                cohort="randomized-2x2",
                interventions=active,
                attributes={
                    "randomization_cell": "control" if not active else "+".join(active),
                    "randomization_seed": 9173,
                },
            )
        )
        endpoints.append(
            SubjectEndpoint(
                subject_id=allocation.subject_id,
                estimand=ESTIMAND,
                estimate=float(endpoint),
                standard_error=endpoint_standard_error,
                baseline_timestamp=START,
                endpoint_timestamp=START + timedelta(days=182),
                baseline_estimate=baseline,
                provenance_id=f"synthetic-endpoint-v1:{allocation.subject_id}",
                quality_flags=("synthetic_example_only",),
            )
        )

    study = Study(
        study_id="synthetic-canine-rapamycin-senolytic-2x2",
        subjects=tuple(subjects),
        observations=(),
        metadata={
            "assignment": "synthetic balanced randomization; dog-level seed 9173",
            "data_status": "fully synthetic; no efficacy inference",
        },
    )
    endpoint_batch = SubjectEndpointBatch(
        study_id=study.study_id,
        batch_id="synthetic-month-6-frailty-v1",
        estimand=ESTIMAND,
        endpoints=tuple(endpoints),
        source_artifact_hash=study_artifact_hash(study),
    )
    return study, endpoint_batch


def analysis_config() -> FactorialCombinationConfig:
    """Prespecify baseline adjustment, robust covariance, and interaction multiplicity."""
    return FactorialCombinationConfig(
        interventions=INTERVENTIONS,
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        covariance_estimator=CovarianceEstimator.HC3,
        multiplicity_method=MultiplicityMethod.BENJAMINI_HOCHBERG,
        minimum_cell_size=16,
        include_baseline_covariate=True,
        endpoint_weighting=EndpointWeighting.INVERSE_VARIANCE_REQUIRED,
    )


def main() -> None:
    """Fit the synthetic analysis and print its estimand and design diagnostics."""
    study, endpoints = build_synthetic_factorial_study()
    report = FactorialCombinationAnalysis(analysis_config()).analyze(
        study,
        endpoints=endpoints,
    )
    interaction = report.interactions[0]
    design = recommend_two_by_two_design(
        TwoByTwoDesignConfig(
            interventions=INTERVENTIONS,
            target_interaction=-0.65,
            residual_standard_deviation=0.45,
            alpha=0.05,
            power=0.80,
            multiplicity_tests=1,
            expected_dropout_fraction=0.10,
        )
    )

    print("Phase 4 synthetic randomized canine 2 x 2 example")
    print(
        f"Subjects: {len(study.subjects)} ({SUBJECTS_PER_CELL} assigned per cell); "
        "dog-level randomization seed=9173"
    )
    print(
        "Model: month-6 frailty adjusted for baseline; "
        f"covariance={report.config.covariance_estimator.value}; "
        f"weighting={report.diagnostics.endpoint_weighting_used.value}"
    )
    print(
        "Interaction family: "
        f"{len(report.interactions)} term; multiplicity={report.config.multiplicity_method.value}"
    )
    for cell in report.cells:
        label = "+".join(cell.active_interventions) or "control"
        print(
            f"Cell {label}: assigned={cell.assigned_subjects}, "
            f"analyzable={cell.analyzable_subjects}, mean={cell.mean_endpoint:.3f}"
        )
    low, high = interaction.confidence_interval or (float("nan"), float("nan"))
    print(
        "Rapamycin:senolytic departure from additivity: "
        f"{interaction.interaction:+.3f} (95% CI {low:+.3f} to {high:+.3f}); "
        f"raw p={interaction.p_value:.4f}; adjusted p={interaction.adjusted_p_value:.4f}"
    )
    print(
        "Design diagnostic: "
        f"rank={report.diagnostics.design_rank}/"
        f"{len(report.diagnostics.design_columns)}, "
        f"cells={report.diagnostics.observed_cells}/{report.diagnostics.required_cells}, "
        f"minimum analyzable cell={report.diagnostics.minimum_analyzable_cell_size}"
    )
    print(
        "Planning illustration: "
        f"{design.analyzable_subjects_per_cell} analyzable and "
        f"{design.enrolled_subjects_per_cell} enrolled dogs per cell under the declared assumptions"
    )
    print(f"Audit artifact: {report.artifact_hash[:16]}...")
    print(
        "Guardrail: synthetic data only. The interaction is departure from additivity on the "
        "declared frailty scale, not an efficacy claim or automatic evidence of synergy."
    )


if __name__ == "__main__":
    main()
