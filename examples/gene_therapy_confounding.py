"""Expose hidden site, vector-lot, and timepoint/plate confounding."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rejuvenationkit import (
    BaselineLongitudinalQC,
    ExpectedVisit,
    Modality,
    Observation,
    QCConfig,
    Study,
    Subject,
    VisitFeature,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def main() -> None:
    """Run a deliberately flawed multicenter canine gene-therapy study."""
    treated_ids = tuple(f"treated-{index}" for index in range(1, 5))
    control_ids = tuple(f"control-{index}" for index in range(1, 5))
    sites = {
        **{subject_id: "north" for subject_id in treated_ids[:3]},
        treated_ids[3]: "south",
        control_ids[0]: "north",
        **{subject_id: "south" for subject_id in control_ids[1:]},
    }
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="treated" if subject_id in treated_ids else "control",
            interventions=("SIRT6_gene_therapy",) if subject_id in treated_ids else (),
            attributes={"site": sites[subject_id]},
        )
        for subject_id in (*treated_ids, *control_ids)
    )
    observations = tuple(
        Observation(
            subject_id=subject.subject_id,
            timestamp=START + timedelta(days=28 * visit_index),
            modality=Modality.CLINICAL,
            feature="ALT",
            value=35 + visit_index,
            unit="U/L",
            attributes={
                "plate": "baseline-plate" if visit_index == 0 else "followup-plate",
                "vector_lot": (
                    "active-lot-A" if subject.subject_id in treated_ids else "sham-lot-B"
                ),
            },
        )
        for subject in subjects
        for visit_index in range(2)
    )
    visits = (
        ExpectedVisit(
            visit_id="baseline",
            scheduled_at=START,
            required_features=(VisitFeature(feature="ALT", modality=Modality.CLINICAL),),
        ),
        ExpectedVisit(
            visit_id="day-28",
            scheduled_at=START + timedelta(days=28),
            required_features=(VisitFeature(feature="ALT", modality=Modality.CLINICAL),),
        ),
    )
    report = BaselineLongitudinalQC(QCConfig(expected_visits=visits)).run(
        Study(
            study_id="confounded-gene-therapy-demo",
            subjects=subjects,
            observations=observations,
        )
    )

    print(report.summary())
    for finding in report.findings:
        if "confounding" not in finding.code:
            continue
        print(
            f"{finding.severity.value.upper()}: {finding.code}: {finding.message}; "
            f"nuisance={finding.context['nuisance_factor']}"
        )


if __name__ == "__main__":
    main()
