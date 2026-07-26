from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from rejuvenationkit.qc import (
    BaselineLongitudinalQC,
    FeatureRule,
    QCConfig,
    Severity,
)
from rejuvenationkit.schemas import Modality, Observation, Study, Subject

START = datetime(2026, 1, 1, tzinfo=UTC)


def row(
    subject_id: str,
    value: float,
    *,
    day: int = 0,
    feature: str = "body_mass",
    unit: str = "g",
    batch_id: str | None = None,
    replicate_id: str | None = None,
) -> Observation:
    return Observation(
        subject_id=subject_id,
        timestamp=START + timedelta(days=day),
        modality=Modality.CLINICAL,
        feature=feature,
        value=value,
        unit=unit,
        batch_id=batch_id,
        replicate_id=replicate_id,
    )


def make_study(*observations: Observation, subjects: tuple[str, ...] = ("s1", "s2")) -> Study:
    return Study(
        study_id="qc-study",
        subjects=tuple(Subject(subject_id=item, cohort="test") for item in subjects),
        observations=observations,
    )


def codes(report: object) -> set[str]:
    assert hasattr(report, "findings")
    return {finding.code for finding in report.findings}


def test_clean_study_passes_and_summarizes() -> None:
    config = QCConfig(
        feature_rules=(
            FeatureRule(
                feature="body_mass",
                expected_unit="g",
                minimum=10,
                maximum=60,
                required=True,
            ),
        ),
        missingness_warning_fraction=0,
        missingness_error_fraction=0,
    )
    report = BaselineLongitudinalQC(config).run(make_study(row("s1", 30), row("s2", 31)))
    assert report.passed
    assert report.counts == {
        Severity.INFO: 0,
        Severity.WARNING: 0,
        Severity.ERROR: 0,
    }
    assert report.summary() == "qc-study: PASS (0 errors, 0 warnings, 0 info)"
    assert report.metrics == {"subjects": 2, "observations": 2, "features": 1, "batches": 0}


def test_range_unit_and_nonfinite_checks() -> None:
    config = QCConfig(
        feature_rules=(FeatureRule(feature="body_mass", expected_unit="g", minimum=10, maximum=60),)
    )
    report = BaselineLongitudinalQC(config).run(
        make_study(row("s1", 70, unit="kg"), row("s2", float("nan")))
    )
    assert codes(report) == {"nonfinite_value", "out_of_range", "unexpected_unit"}
    assert not report.passed
    assert report.counts[Severity.ERROR] == 3


def test_missing_required_feature_uses_configured_severity() -> None:
    rule = FeatureRule(feature="body_mass", required=True)
    warning_config = QCConfig(
        feature_rules=(rule,),
        missingness_warning_fraction=0.25,
        missingness_error_fraction=0.75,
    )
    warning = BaselineLongitudinalQC(warning_config).run(
        make_study(row("s1", 30), subjects=("s1", "s2"))
    )
    assert warning.findings[0].severity is Severity.WARNING
    assert warning.findings[0].subject_ids == ("s2",)

    error_config = warning_config.model_copy(update={"missingness_error_fraction": 0.50})
    error = BaselineLongitudinalQC(error_config).run(
        make_study(row("s1", 30), subjects=("s1", "s2"))
    )
    assert error.findings[0].severity is Severity.ERROR


def test_temporal_order_can_be_detected_or_disabled() -> None:
    study = make_study(row("s1", 31, day=2), row("s1", 30, day=1), subjects=("s1",))
    report = BaselineLongitudinalQC().run(study)
    assert codes(report) == {"timestamp_out_of_order"}
    assert report.findings[0].observation_indices == (0, 1)

    disabled = BaselineLongitudinalQC(QCConfig(check_input_order=False)).run(study)
    assert disabled.findings == ()


def test_replicate_disagreement() -> None:
    study = make_study(
        row("s1", 10, batch_id="a", replicate_id="r1"),
        row("s1", 20, batch_id="a", replicate_id="r2"),
        subjects=("s1",),
    )
    report = BaselineLongitudinalQC(QCConfig(replicate_relative_tolerance=0.25)).run(study)
    assert codes(report) == {"replicate_disagreement"}
    assert report.findings[0].context["relative_spread"] == pytest.approx(2 / 3)


def test_batch_mean_shift_requires_sufficient_samples() -> None:
    observations = tuple(
        row(f"s{index}", value, batch_id=batch)
        for index, (value, batch) in enumerate(
            [
                (10.0, "a"),
                (10.1, "a"),
                (9.9, "a"),
                (20.0, "b"),
                (20.1, "b"),
                (19.9, "b"),
            ],
            start=1,
        )
    )
    study = make_study(*observations, subjects=tuple(f"s{index}" for index in range(1, 7)))
    report = BaselineLongitudinalQC(QCConfig(batch_z_threshold=3, minimum_batch_size=3)).run(study)
    assert codes(report) == {"batch_mean_shift"}
    assert len(report.findings) == 2

    undersized = BaselineLongitudinalQC(QCConfig(batch_z_threshold=3, minimum_batch_size=4)).run(
        study
    )
    assert undersized.findings == ()


def test_modality_specific_rule_wins_over_generic_rule() -> None:
    config = QCConfig(
        feature_rules=(
            FeatureRule(feature="age", maximum=200),
            FeatureRule(feature="age", modality=Modality.CLINICAL, maximum=20),
        )
    )
    report = BaselineLongitudinalQC(config).run(
        make_study(row("s1", 30, feature="age", unit="years"), subjects=("s1",))
    )
    assert codes(report) == {"out_of_range"}


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="minimum cannot exceed maximum"):
        FeatureRule(feature="body_mass", minimum=60, maximum=10)
    with pytest.raises(ValidationError, match="warning threshold"):
        QCConfig(missingness_warning_fraction=0.8, missingness_error_fraction=0.2)
    with pytest.raises(ValidationError, match="unique"):
        QCConfig(
            feature_rules=(
                FeatureRule(feature="body_mass"),
                FeatureRule(feature="body_mass"),
            )
        )
