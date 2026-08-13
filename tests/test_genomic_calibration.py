import numpy as np
import pytest
from scipy import sparse

from rejuvenationkit import EffectDirection, Modality
from rejuvenationkit.genomics import (
    FeatureNamespace,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    GenomicTargetCalibrator,
    GenomicTargetConfig,
    MatrixScale,
)


def genomic_matrix(
    prefix: str,
    values: np.ndarray,
    *,
    tissue: str = "blood",
    species_taxon_id: int = 9615,
    scale: MatrixScale = MatrixScale.NORMALIZED_EXPRESSION,
    subject_ids: tuple[str, ...] | None = None,
) -> GenomicMatrix:
    samples = tuple(
        GenomicSample(
            sample_id=f"{prefix}-sample-{index}",
            subject_id=(subject_ids[index] if subject_ids else f"{prefix}-subject-{index}"),
            tissue=tissue,
            species_taxon_id=species_taxon_id,
            cohort="calibration" if prefix == "train" else "evaluation",
        )
        for index in range(values.shape[0])
    )
    features = tuple(
        GenomicFeature(
            feature_id=f"feature-{index}",
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
        )
        for index in range(values.shape[1])
    )
    return GenomicMatrix(
        values=values,
        samples=samples,
        features=features,
        scale=scale,
        provenance=GenomicMatrixProvenance(source_id=f"{prefix}-matrix"),
    )


def config() -> GenomicTargetConfig:
    return GenomicTargetConfig(
        target_name="transcriptomic_age",
        target_unit="years",
        direction=EffectDirection.LOWER_IS_BETTER,
        alpha=0.1,
        cross_validation_folds=5,
        minimum_training_samples=20,
        random_seed=4,
    )


def training_fixture() -> tuple[GenomicMatrix, dict[str, float]]:
    generator = np.random.default_rng(10)
    values = generator.normal(size=(30, 5))
    matrix = genomic_matrix("train", values)
    target_values = values @ np.asarray([1.5, -0.8, 0.3, 0.0, 0.5])
    target_values += generator.normal(scale=0.15, size=30)
    targets = {
        sample_id: float(target)
        for sample_id, target in zip(matrix.sample_ids, target_values, strict=True)
    }
    return matrix, targets


def test_genomic_target_calibrator_uses_out_of_fold_subject_uncertainty() -> None:
    train, targets = training_fixture()
    generator = np.random.default_rng(20)
    evaluation = genomic_matrix("eval", generator.normal(size=(5, 5)))
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)
    result = calibrator.predict(evaluation)

    assert calibrator.calibration_id is not None
    assert result.calibration_id == calibrator.calibration_id
    assert result.training_sample_count == 30
    assert result.training_subject_count == 30
    assert result.cross_validated_rmse > 0
    assert result.empirical_absolute_error_quantile > 0
    assert len(result.predictions) == 5
    assert result.feature_ids == train.feature_ids
    prediction = result.predictions[0]
    assert prediction.confidence_interval[0] < prediction.estimate
    assert prediction.confidence_interval[1] > prediction.estimate
    evidence = prediction.to_evidence(
        modality=Modality.TRANSCRIPTOMICS,
        provenance_id="evaluation-matrix",
    )
    assert evidence.subject_id == prediction.subject_id
    assert evidence.calibration_id == result.calibration_id
    assert evidence.standard_error >= prediction.standard_error
    assert "subject_balanced_cross_validated_rmse" in prediction.uncertainty_method
    assert prediction.species_taxon_id == 9615
    assert prediction.tissue == "blood"
    assert prediction.out_of_domain_threshold == result.out_of_domain_threshold


def test_grouped_cross_validation_counts_independent_subjects() -> None:
    generator = np.random.default_rng(30)
    values = generator.normal(size=(30, 4))
    subjects = tuple(f"dog-{index // 2}" for index in range(30))
    train = genomic_matrix("train", values, subject_ids=subjects)
    targets = {
        sample_id: float(value)
        for sample_id, value in zip(
            train.sample_ids,
            values[:, 0] + 0.1 * values[:, 1],
            strict=True,
        )
    }
    evaluation = genomic_matrix("eval", generator.normal(size=(3, 4)))
    result = GenomicTargetCalibrator(config()).fit(train, targets).predict(evaluation)

    assert result.training_sample_count == 30
    assert result.training_subject_count == 15


def test_calibrator_requires_independent_training_subjects() -> None:
    generator = np.random.default_rng(31)
    values = generator.normal(size=(30, 4))
    subjects = tuple(f"dog-{index // 10}" for index in range(30))
    train = genomic_matrix("train", values, subject_ids=subjects)
    targets = {
        sample_id: float(value)
        for sample_id, value in zip(train.sample_ids, values[:, 0], strict=True)
    }

    with pytest.raises(ValueError, match="independent training subjects"):
        GenomicTargetCalibrator(config()).fit(train, targets)


def test_calibrator_rejects_training_and_subject_leakage() -> None:
    train, targets = training_fixture()
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)
    with pytest.raises(ValueError, match="training samples"):
        calibrator.predict(train)

    generator = np.random.default_rng(40)
    overlapping_subjects = (
        train.samples[0].subject_id,
        "new-subject-1",
        "new-subject-2",
    )
    evaluation = genomic_matrix(
        "eval",
        generator.normal(size=(3, 5)),
        subject_ids=overlapping_subjects,
    )
    with pytest.raises(ValueError, match="training subjects"):
        calibrator.predict(evaluation)


@pytest.mark.parametrize(
    ("tissue", "taxon", "scale", "message"),
    [
        ("liver", 9615, MatrixScale.NORMALIZED_EXPRESSION, "tissue"),
        ("blood", 9606, MatrixScale.NORMALIZED_EXPRESSION, "species"),
        ("blood", 9615, MatrixScale.LOG_CPM, "scale"),
    ],
)
def test_calibrator_rejects_domain_shift(
    tissue: str,
    taxon: int,
    scale: MatrixScale,
    message: str,
) -> None:
    train, targets = training_fixture()
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)
    evaluation = genomic_matrix(
        "eval",
        np.ones((3, 5)),
        tissue=tissue,
        species_taxon_id=taxon,
        scale=scale,
    )
    with pytest.raises(ValueError, match=message):
        calibrator.predict(evaluation)


def test_calibrator_rejects_invalid_training_inputs_and_unfitted_use() -> None:
    train, targets = training_fixture()
    calibrator = GenomicTargetCalibrator(config())
    with pytest.raises(RuntimeError, match="fitted"):
        calibrator.predict(genomic_matrix("eval", np.ones((3, 5))))
    with pytest.raises(ValueError, match="exactly match"):
        calibrator.fit(train, dict(list(targets.items())[:-1]))
    bad_targets = dict(targets)
    bad_targets[train.sample_ids[0]] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        calibrator.fit(train, bad_targets)
    too_small = genomic_matrix("small", np.ones((10, 5)))
    with pytest.raises(ValueError, match="at least 20"):
        calibrator.fit(too_small, dict.fromkeys(too_small.sample_ids, 1.0))
    raw = genomic_matrix("train", np.ones((30, 5)), scale=MatrixScale.RAW_COUNTS)
    with pytest.raises(ValueError, match="not allowed"):
        calibrator.fit(raw, dict.fromkeys(raw.sample_ids, 1.0))


def test_evaluation_features_must_match_training_set() -> None:
    train, targets = training_fixture()
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)
    evaluation = genomic_matrix("eval", np.ones((3, 4)))
    with pytest.raises(ValueError, match="features"):
        calibrator.predict(evaluation)


def test_sparse_domain_distance_is_centered_on_training_distribution() -> None:
    generator = np.random.default_rng(50)
    dense_values = 10 + generator.normal(scale=0.5, size=(30, 5))
    dense_train = genomic_matrix("train", dense_values)
    train = GenomicMatrix(
        values=sparse.csr_matrix(dense_values),
        samples=dense_train.samples,
        features=dense_train.features,
        scale=dense_train.scale,
        provenance=dense_train.provenance,
    )
    targets = {
        sample_id: float(value)
        for sample_id, value in zip(train.sample_ids, dense_values[:, 0], strict=True)
    }
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)
    near = genomic_matrix("near", np.full((2, 5), 10.0))
    far = genomic_matrix("far", np.zeros((2, 5)))

    near_result = calibrator.predict(
        GenomicMatrix(
            values=sparse.csr_matrix(near.values),
            samples=near.samples,
            features=near.features,
            scale=near.scale,
            provenance=near.provenance,
        )
    )
    far_result = calibrator.predict(
        GenomicMatrix(
            values=sparse.csr_matrix(far.values),
            samples=far.samples,
            features=far.features,
            scale=far.scale,
            provenance=far.provenance,
        )
    )

    assert near_result.predictions[0].out_of_domain_score < near_result.out_of_domain_threshold
    assert far_result.predictions[0].out_of_domain_score > far_result.out_of_domain_threshold
    assert "high_standardized_feature_distance" in far_result.predictions[0].warnings
