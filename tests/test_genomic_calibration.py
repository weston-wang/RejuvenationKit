from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy import sparse

from rejuvenationkit import EffectDirection, Modality
from rejuvenationkit.evidence import CalibrationValidationStatus
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
    values: npt.NDArray[np.float64],
    *,
    tissue: str = "blood",
    species_taxon_id: int = 9615,
    scale: MatrixScale = MatrixScale.NORMALIZED_EXPRESSION,
    subject_ids: tuple[str, ...] | None = None,
    cohort: str | None = None,
    sample_attributes: dict[str, str] | None = None,
    feature_type: GenomicFeatureType = GenomicFeatureType.GENE,
    namespace: FeatureNamespace = FeatureNamespace.ENSEMBL,
    genome_assembly: str | None = None,
    feature_symbol_prefix: str | None = None,
    feature_attributes: dict[str, str] | None = None,
    preprocessing: tuple[str, ...] = (),
    software_versions: dict[str, str] | None = None,
    reference_resource_ids: tuple[str, ...] = (),
) -> GenomicMatrix:
    samples = tuple(
        GenomicSample(
            sample_id=f"{prefix}-sample-{index}",
            subject_id=(subject_ids[index] if subject_ids else f"{prefix}-subject-{index}"),
            tissue=tissue,
            species_taxon_id=species_taxon_id,
            cohort=(
                cohort
                if cohort is not None
                else "calibration"
                if prefix == "train"
                else "evaluation"
            ),
            attributes=sample_attributes or {},
        )
        for index in range(values.shape[0])
    )
    features = tuple(
        GenomicFeature(
            feature_id=f"feature-{index}",
            feature_type=feature_type,
            namespace=namespace,
            genome_assembly=genome_assembly,
            symbol=(f"{feature_symbol_prefix}{index}" if feature_symbol_prefix else None),
            attributes=feature_attributes or {},
        )
        for index in range(values.shape[1])
    )
    return GenomicMatrix(
        values=values,
        samples=samples,
        features=features,
        scale=scale,
        provenance=GenomicMatrixProvenance(
            source_id=f"{prefix}-matrix",
            preprocessing=preprocessing,
            software_versions=software_versions or {},
            reference_resource_ids=reference_resource_ids,
        ),
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
    assert result.training_artifact_hash == train.artifact_hash
    assert result.evaluation_artifact_hash == evaluation.artifact_hash
    assert result.training_provenance_id == "train-matrix"
    assert result.evaluation_provenance_id == "eval-matrix"
    assert result.training_domain.species_taxon_id == 9615
    assert result.training_domain.feature_types == (GenomicFeatureType.GENE,)
    assert result.training_domain.feature_namespaces == (FeatureNamespace.ENSEMBL,)
    assert result.training_domain.feature_ids == train.feature_ids
    assert result.training_domain == result.evaluation_domain
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
    assert evidence.calibration_reference is not None
    assert (
        evidence.calibration_reference.status
        is CalibrationValidationStatus.INTERNAL_CROSS_VALIDATED
    )
    assert not evidence.calibration_reference.fusion_eligible
    assert evidence.calibration_reference.estimand == evidence.estimand
    assert evidence.calibration_reference.artifact_hash == prediction.calibration_artifact_hash
    assert evidence.calibration_reference.validation_provenance_id == result.training_provenance_id
    assert evidence.standard_error >= prediction.standard_error
    assert "subject_balanced_cross_validated_rmse" in prediction.uncertainty_method
    assert prediction.species_taxon_id == 9615
    assert prediction.tissue == "blood"
    assert prediction.training_artifact_hash == train.artifact_hash
    assert prediction.calibration_artifact_hash != result.predictions[1].calibration_artifact_hash
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


def test_calibration_identity_binds_complete_training_artifact() -> None:
    train, targets = training_fixture()
    values = train.dense_values()
    baseline_id = GenomicTargetCalibrator(config()).fit(train, targets).calibration_id
    changed_artifacts = (
        genomic_matrix("train", values, tissue="liver"),
        genomic_matrix("train", values, species_taxon_id=9606),
        genomic_matrix("train", values, scale=MatrixScale.LOG_CPM),
        genomic_matrix("train", values, namespace=FeatureNamespace.REFSEQ),
        genomic_matrix(
            "train",
            values,
            feature_type=GenomicFeatureType.TRANSCRIPT,
        ),
        genomic_matrix("train", values, genome_assembly="CanFam3.1"),
        genomic_matrix("train", values, preprocessing=("log-cpm:v2",)),
        genomic_matrix("train", values, software_versions={"normalizer": "2.0"}),
        genomic_matrix("train", values, reference_resource_ids=("Ensembl-110",)),
        genomic_matrix("train", values, sample_attributes={"collection_site": "A"}),
    )

    assert all(item.content_hash == train.content_hash for item in changed_artifacts)
    assert all(item.artifact_hash != train.artifact_hash for item in changed_artifacts)
    changed_ids = {
        GenomicTargetCalibrator(config()).fit(item, targets).calibration_id
        for item in changed_artifacts
    }
    assert baseline_id not in changed_ids
    assert len(changed_ids) == len(changed_artifacts)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"namespace": FeatureNamespace.REFSEQ}, "namespace"),
        ({"feature_type": GenomicFeatureType.TRANSCRIPT}, "feature type"),
        ({"genome_assembly": "CanFam4"}, "genome assembly"),
        ({"feature_symbol_prefix": "changed-"}, "feature metadata"),
        ({"feature_attributes": {"annotation_release": "111"}}, "feature metadata"),
        ({"preprocessing": ("log-cpm:v2",)}, "preprocessing differs"),
        ({"software_versions": {"normalizer": "2.0"}}, "software differs"),
        ({"reference_resource_ids": ("Ensembl-111",)}, "reference resources"),
    ],
)
def test_calibrator_rejects_incompatible_feature_and_pipeline_domain(
    changes: dict[str, Any],
    message: str,
) -> None:
    generator = np.random.default_rng(41)
    training_values = generator.normal(size=(30, 5))
    domain: dict[str, Any] = {
        "genome_assembly": "CanFam3.1",
        "feature_symbol_prefix": "gene-",
        "preprocessing": ("log-cpm:v1",),
        "software_versions": {"normalizer": "1.0"},
        "reference_resource_ids": ("Ensembl-110",),
    }
    train = genomic_matrix("train", training_values, **domain)
    targets = {
        sample_id: float(value)
        for sample_id, value in zip(train.sample_ids, training_values[:, 0], strict=True)
    }
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)
    evaluation_domain = domain | changes
    evaluation = genomic_matrix(
        "eval",
        generator.normal(size=(3, 5)),
        **evaluation_domain,
    )

    with pytest.raises(ValueError, match=message):
        calibrator.predict(evaluation)


def test_evaluation_sample_metadata_is_artifact_bound_but_not_a_domain_constraint() -> None:
    train, targets = training_fixture()
    generator = np.random.default_rng(42)
    values = generator.normal(size=(3, 5))
    baseline = genomic_matrix("eval", values, sample_attributes={"site": "A"})
    changed = genomic_matrix("eval", values, sample_attributes={"site": "B"})
    calibrator = GenomicTargetCalibrator(config()).fit(train, targets)

    baseline_result = calibrator.predict(baseline)
    changed_result = calibrator.predict(changed)

    assert baseline.content_hash == changed.content_hash
    assert baseline.artifact_hash != changed.artifact_hash
    assert baseline_result.evaluation_artifact_hash == baseline.artifact_hash
    assert changed_result.evaluation_artifact_hash == changed.artifact_hash
    assert baseline_result.evaluation_domain == changed_result.evaluation_domain
    assert [item.estimate for item in baseline_result.predictions] == [
        item.estimate for item in changed_result.predictions
    ]


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
