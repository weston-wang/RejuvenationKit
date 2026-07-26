from datetime import UTC, datetime

import pytest

from rejuvenationkit.combinations import FactorialCombinationAnalysis, InteractionEstimate
from rejuvenationkit.fusion import ModalityEstimate, PrecisionWeightedFusion
from rejuvenationkit.qc import BaselineLongitudinalQC
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.state import LinearGaussianStateEstimator


@pytest.fixture
def study() -> Study:
    return Study(
        study_id="study",
        subjects=(Subject(subject_id="s1", cohort="treated"),),
        observations=(
            Observation(
                subject_id="s1",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                modality=Modality.CLINICAL,
                feature="body_mass",
                value=30.0,
                unit="g",
            ),
        ),
    )


def test_phase_one_returns_report(study: Study) -> None:
    report = BaselineLongitudinalQC().run(study)
    assert report.passed
    assert report.metrics["observations"] == 1


def test_phase_two_stubs_are_explicit(study: Study) -> None:
    estimator = PrecisionWeightedFusion()
    with pytest.raises(NotImplementedError, match="calibration"):
        estimator.fit(study)
    inputs = (
        ModalityEstimate(
            modality=Modality.CLINICAL,
            estimate=1.0,
            standard_error=0.2,
            target="biological_age_delta",
        ),
    )
    with pytest.raises(NotImplementedError, match="fusion"):
        estimator.fuse(inputs)


def test_phase_three_stubs_are_explicit(study: Study) -> None:
    estimator = LinearGaussianStateEstimator()
    with pytest.raises(NotImplementedError, match="state model"):
        estimator.fit(study)
    with pytest.raises(NotImplementedError, match="state estimation"):
        estimator.estimate(study)


def test_phase_four_stub_and_schema_are_explicit(study: Study) -> None:
    result = InteractionEstimate(
        interventions=("therapy-a", "therapy-b"),
        outcome="biological_age_delta",
        interaction=-1.2,
        standard_error=0.3,
        reference_model="additive",
    )
    assert result.interventions == ("therapy-a", "therapy-b")
    with pytest.raises(NotImplementedError, match="combination"):
        FactorialCombinationAnalysis().estimate(study, outcome=result.outcome)
