"""Leakage-aware calibration of genomic features to declared scalar targets."""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from math import isfinite
from statistics import NormalDist
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse

from rejuvenationkit.evidence import EffectDirection, Estimand, EvidenceEstimate
from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicMatrix,
    MatrixScale,
)
from rejuvenationkit.schemas import Modality


class GenomicTargetConfig(BaseModel):
    """Domain, regularization, and uncertainty policy for a genomic calibrator."""

    model_config = ConfigDict(frozen=True)

    target_name: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    direction: EffectDirection = EffectDirection.UNSPECIFIED
    alpha: float = Field(default=1.0, gt=0)
    cross_validation_folds: int = Field(default=5, ge=3)
    minimum_training_samples: int = Field(default=20, ge=6)
    minimum_training_subjects: int = Field(default=10, ge=3)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    out_of_domain_quantile: float = Field(default=0.99, gt=0, lt=1)
    minimum_out_of_domain_threshold: float = Field(default=1.5, gt=0)
    random_seed: int = 0
    reject_training_subject_overlap: bool = True
    allowed_scales: tuple[MatrixScale, ...] = (
        MatrixScale.LOG_CPM,
        MatrixScale.NORMALIZED_EXPRESSION,
        MatrixScale.METHYLATION_M,
        MatrixScale.EMBEDDING,
        MatrixScale.VARIANT_DOSAGE,
    )

    @model_validator(mode="after")
    def validate_domain_policy(self) -> Self:
        """Require a coherent independent-subject and scale contract."""
        if self.minimum_training_samples < self.minimum_training_subjects:
            raise ValueError("minimum_training_samples cannot be below minimum_training_subjects")
        if not self.allowed_scales or len(set(self.allowed_scales)) != len(self.allowed_scales):
            raise ValueError("allowed_scales must be nonempty and unique")
        return self


class GenomicPrediction(BaseModel):
    """One held-out calibrated prediction with empirical uncertainty."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    subject_id: str
    estimate: float
    standard_error: float = Field(gt=0)
    uncertainty_method: str = Field(min_length=1)
    empirical_interval_half_width: float = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    confidence_interval: tuple[float, float]
    out_of_domain_score: float = Field(ge=0)
    out_of_domain_threshold: float = Field(gt=0)
    calibration_id: str
    target_name: str
    target_unit: str
    direction: EffectDirection
    species_taxon_id: int = Field(gt=0)
    tissue: str = Field(min_length=1)
    matrix_scale: MatrixScale
    feature_namespace: FeatureNamespace
    warnings: tuple[str, ...] = ()

    def to_evidence(
        self,
        *,
        modality: Modality,
        provenance_id: str,
        assay_id: str | None = None,
    ) -> EvidenceEstimate:
        """Convert a held-out calibrated prediction to subject-level evidence."""
        z_value = NormalDist().inv_cdf(0.5 + self.confidence_level / 2)
        interval_equivalent_error = self.empirical_interval_half_width / z_value
        evidence_error = max(self.standard_error, interval_equivalent_error)
        flags = list(self.warnings)
        if evidence_error > self.standard_error:
            flags.append("empirical_interval_converted_to_normal_equivalent_standard_error")
        return EvidenceEstimate(
            evidence_id=f"{self.calibration_id}:{self.sample_id}",
            modality=modality,
            estimand=Estimand(
                name=self.target_name,
                unit=self.target_unit,
                direction=self.direction,
                population=f"subject:{self.subject_id}",
            ),
            estimate=self.estimate,
            standard_error=evidence_error,
            calibration_id=self.calibration_id,
            provenance_id=provenance_id,
            subject_id=self.subject_id,
            sample_id=self.sample_id,
            tissue=self.tissue,
            species_taxon_id=self.species_taxon_id,
            assay_id=assay_id,
            quality_flags=tuple(flags),
        )


class GenomicPredictionBatch(BaseModel):
    """Predictions sharing one model and training-domain contract."""

    model_config = ConfigDict(frozen=True)

    calibration_id: str
    predictions: tuple[GenomicPrediction, ...]
    training_sample_count: int
    training_subject_count: int
    cross_validated_rmse: float = Field(gt=0)
    empirical_absolute_error_quantile: float = Field(gt=0)
    out_of_domain_threshold: float = Field(gt=0)
    feature_ids: tuple[str, ...]
    training_provenance_id: str


class GenomicTargetCalibrator:
    """Cross-validated ridge mapping from genomic features to one target.

    The estimator uses subject-grouped folds when a subject has multiple
    samples. Prediction rejects reuse of training subjects by default. Its
    uncertainty is the held-out residual error, not variation across genomic
    features or embedding dimensions.
    """

    def __init__(self, config: GenomicTargetConfig) -> None:
        """Initialize an unfitted genomic target calibrator."""
        self.config = config
        self._pipeline: Any | None = None
        self._training_sample_ids: frozenset[str] = frozenset()
        self._training_subject_ids: frozenset[str] = frozenset()
        self._feature_ids: tuple[str, ...] = ()
        self._species_taxon_id: int | None = None
        self._tissue: str | None = None
        self._scale: MatrixScale | None = None
        self._namespace: FeatureNamespace | None = None
        self._rmse: float | None = None
        self._absolute_error_quantile: float | None = None
        self._calibration_id: str | None = None
        self._training_provenance_id: str | None = None
        self._feature_means: npt.NDArray[np.float64] | None = None
        self._feature_scales: npt.NDArray[np.float64] | None = None
        self._out_of_domain_threshold: float | None = None

    @property
    def calibration_id(self) -> str | None:
        """Return the immutable calibration fingerprint after fitting."""
        return self._calibration_id

    def fit(
        self,
        matrix: GenomicMatrix,
        targets: dict[str, float],
    ) -> GenomicTargetCalibrator:
        """Fit with subject-grouped out-of-fold residual calibration."""
        self._validate_training_domain(matrix)
        if len(matrix.samples) < self.config.minimum_training_samples:
            raise ValueError(
                f"at least {self.config.minimum_training_samples} training samples are required"
            )
        if set(targets) != set(matrix.sample_ids):
            raise ValueError("targets must exactly match matrix sample identifiers")
        target_values = np.asarray([targets[item] for item in matrix.sample_ids], dtype=float)
        if not np.isfinite(target_values).all():
            raise ValueError("training targets must be finite")
        if not isinstance(matrix.values, sparse.csr_matrix) and np.isnan(matrix.values).any():
            raise ValueError("calibrator does not accept missing genomic feature values")

        modules = _sklearn_modules()
        with_mean = not isinstance(matrix.values, sparse.csr_matrix)
        pipeline = modules["Pipeline"](
            [
                ("scale", modules["StandardScaler"](with_mean=with_mean)),
                ("ridge", modules["Ridge"](alpha=self.config.alpha)),
            ]
        )
        groups = np.asarray([item.subject_id for item in matrix.samples])
        unique_groups = np.unique(groups)
        if len(unique_groups) < self.config.minimum_training_subjects:
            raise ValueError(
                f"at least {self.config.minimum_training_subjects} independent training "
                "subjects are required"
            )
        folds = min(self.config.cross_validation_folds, len(unique_groups))
        if folds < 3:
            raise ValueError("at least three independent subjects are required for calibration")
        if len(unique_groups) < len(groups):
            splitter = modules["GroupKFold"](n_splits=folds)
            predictions = modules["cross_val_predict"](
                pipeline,
                matrix.values,
                target_values,
                cv=splitter,
                groups=groups,
            )
        else:
            splitter = modules["KFold"](
                n_splits=folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )
            predictions = modules["cross_val_predict"](
                pipeline,
                matrix.values,
                target_values,
                cv=splitter,
            )
        residuals = target_values - np.asarray(predictions, dtype=float)
        subject_mean_squared_errors = np.asarray(
            [np.mean(residuals[groups == group] ** 2) for group in unique_groups],
            dtype=float,
        )
        subject_mean_absolute_errors = np.asarray(
            [np.mean(np.abs(residuals[groups == group])) for group in unique_groups],
            dtype=float,
        )
        rmse = float(np.sqrt(np.mean(subject_mean_squared_errors)))
        if rmse <= 0 or not isfinite(rmse):
            raise ValueError("cross-validated residual uncertainty must be positive and finite")
        quantile_probability = self.config.confidence_level
        absolute_quantile = float(
            np.quantile(subject_mean_absolute_errors, quantile_probability, method="higher")
        )
        if absolute_quantile <= 0 or not isfinite(absolute_quantile):
            raise ValueError("empirical prediction error quantile must be positive and finite")
        pipeline.fit(matrix.values, target_values)
        scaler = pipeline.named_steps["scale"]
        feature_means = np.asarray(scaler.mean_, dtype=float)
        feature_scales = np.asarray(scaler.scale_, dtype=float)
        training_domain_scores = _row_standardized_root_mean_square(
            matrix.values,
            feature_means,
            feature_scales,
        )
        out_of_domain_threshold = max(
            self.config.minimum_out_of_domain_threshold,
            float(np.quantile(training_domain_scores, self.config.out_of_domain_quantile)),
        )

        species = {item.species_taxon_id for item in matrix.samples}
        tissues = {item.tissue for item in matrix.samples}
        namespaces = {item.namespace for item in matrix.features}
        domain_fingerprint = (
            f"species={next(iter(species))}|tissue={next(iter(tissues))}|"
            f"scale={matrix.scale.value}|namespace={next(iter(namespaces)).value}|"
            f"preprocessing={matrix.provenance.preprocessing}|"
            f"software={sorted(matrix.provenance.software_versions.items())}|"
            f"references={matrix.provenance.reference_resource_ids}"
        )
        self._pipeline = pipeline
        self._training_sample_ids = frozenset(matrix.sample_ids)
        self._training_subject_ids = frozenset(groups.tolist())
        self._feature_ids = matrix.feature_ids
        self._species_taxon_id = next(iter(species))
        self._tissue = next(iter(tissues))
        self._scale = matrix.scale
        self._namespace = next(iter(namespaces))
        self._rmse = rmse
        self._absolute_error_quantile = absolute_quantile
        self._training_provenance_id = matrix.provenance.source_id
        self._feature_means = feature_means
        self._feature_scales = feature_scales
        self._out_of_domain_threshold = out_of_domain_threshold
        target_fingerprint = sha256(target_values.tobytes()).hexdigest()
        subject_fingerprint = sha256(
            "|".join(f"{item.sample_id}:{item.subject_id}" for item in matrix.samples).encode()
        ).hexdigest()
        fingerprint = (
            f"{matrix.content_hash}|{target_fingerprint}|{subject_fingerprint}|"
            f"{self.config.model_dump_json()}|"
            f"{matrix.provenance.source_id}|{domain_fingerprint}"
        )
        self._calibration_id = f"genomic-ridge:{sha256(fingerprint.encode()).hexdigest()[:16]}"
        return self

    def predict(self, matrix: GenomicMatrix) -> GenomicPredictionBatch:
        """Predict only after exact feature and training-domain compatibility checks."""
        if self._pipeline is None or self._rmse is None or self._calibration_id is None:
            raise RuntimeError("genomic target calibrator must be fitted before prediction")
        if self._absolute_error_quantile is None or self._training_provenance_id is None:
            raise RuntimeError("calibrator uncertainty provenance is incomplete")
        if (
            self._feature_means is None
            or self._feature_scales is None
            or self._out_of_domain_threshold is None
            or self._species_taxon_id is None
            or self._tissue is None
            or self._scale is None
            or self._namespace is None
        ):
            raise RuntimeError("calibrator domain-distance state is incomplete")
        self._validate_evaluation_domain(matrix)
        overlapping_samples = self._training_sample_ids.intersection(matrix.sample_ids)
        if overlapping_samples:
            raise ValueError(f"evaluation reuses training samples: {sorted(overlapping_samples)}")
        evaluation_subjects = {item.subject_id for item in matrix.samples}
        overlapping_subjects = self._training_subject_ids.intersection(evaluation_subjects)
        if overlapping_subjects and self.config.reject_training_subject_overlap:
            raise ValueError(f"evaluation reuses training subjects: {sorted(overlapping_subjects)}")
        indices = matrix.feature_indices(self._feature_ids)
        if isinstance(matrix.values, sparse.csr_matrix):
            values: Any = cast(Any, matrix.values)[:, list(indices)]
        else:
            values = matrix.values[:, list(indices)]
        if not isinstance(values, sparse.csr_matrix) and np.isnan(values).any():
            raise ValueError("calibrator does not accept missing genomic feature values")
        predictions = np.asarray(self._pipeline.predict(values), dtype=float)
        domain_scores = _row_standardized_root_mean_square(
            values,
            self._feature_means,
            self._feature_scales,
        )
        z_value = NormalDist().inv_cdf(0.5 + self.config.confidence_level / 2)
        output: list[GenomicPrediction] = []
        for sample, estimate, domain_score in zip(
            matrix.samples,
            predictions,
            domain_scores,
            strict=True,
        ):
            warnings = (
                ("high_standardized_feature_distance",)
                if domain_score > self._out_of_domain_threshold
                else ()
            )
            half_width = max(
                z_value * self._rmse,
                self._absolute_error_quantile,
            )
            output.append(
                GenomicPrediction(
                    sample_id=sample.sample_id,
                    subject_id=sample.subject_id,
                    estimate=float(estimate),
                    standard_error=self._rmse,
                    uncertainty_method=(
                        "subject_balanced_cross_validated_rmse_with_empirical_absolute_error_interval"
                    ),
                    empirical_interval_half_width=half_width,
                    confidence_level=self.config.confidence_level,
                    confidence_interval=(
                        float(estimate - half_width),
                        float(estimate + half_width),
                    ),
                    out_of_domain_score=float(domain_score),
                    out_of_domain_threshold=self._out_of_domain_threshold,
                    calibration_id=self._calibration_id,
                    target_name=self.config.target_name,
                    target_unit=self.config.target_unit,
                    direction=self.config.direction,
                    species_taxon_id=self._species_taxon_id,
                    tissue=self._tissue,
                    matrix_scale=self._scale,
                    feature_namespace=self._namespace,
                    warnings=warnings,
                )
            )
        return GenomicPredictionBatch(
            calibration_id=self._calibration_id,
            predictions=tuple(output),
            training_sample_count=len(self._training_sample_ids),
            training_subject_count=len(self._training_subject_ids),
            cross_validated_rmse=self._rmse,
            empirical_absolute_error_quantile=self._absolute_error_quantile,
            out_of_domain_threshold=self._out_of_domain_threshold,
            feature_ids=self._feature_ids,
            training_provenance_id=self._training_provenance_id,
        )

    def _validate_training_domain(self, matrix: GenomicMatrix) -> None:
        if matrix.scale not in self.config.allowed_scales:
            raise ValueError(f"matrix scale {matrix.scale.value} is not allowed for calibration")
        species = {item.species_taxon_id for item in matrix.samples}
        tissues = {item.tissue for item in matrix.samples}
        namespaces = {item.namespace for item in matrix.features}
        if len(species) != 1 or len(tissues) != 1 or len(namespaces) != 1:
            raise ValueError("training matrix must have one species, tissue, and feature namespace")

    def _validate_evaluation_domain(self, matrix: GenomicMatrix) -> None:
        species = {item.species_taxon_id for item in matrix.samples}
        tissues = {item.tissue for item in matrix.samples}
        namespaces = {item.namespace for item in matrix.features}
        if species != {self._species_taxon_id}:
            raise ValueError("evaluation species differs from calibration domain")
        if tissues != {self._tissue}:
            raise ValueError("evaluation tissue differs from calibration domain")
        if matrix.scale is not self._scale:
            raise ValueError("evaluation scale differs from calibration domain")
        if namespaces != {self._namespace}:
            raise ValueError("evaluation feature namespace differs from calibration domain")
        if set(matrix.feature_ids) != set(self._feature_ids):
            raise ValueError("evaluation features must exactly match calibration features")


def _sklearn_modules() -> dict[str, Any]:
    try:
        linear_model = import_module("sklearn.linear_model")
        model_selection = import_module("sklearn.model_selection")
        pipeline = import_module("sklearn.pipeline")
        preprocessing = import_module("sklearn.preprocessing")
    except ModuleNotFoundError as error:
        raise ImportError(
            "GenomicTargetCalibrator requires optional dependencies; "
            "install rejuvenationkit[genomics]"
        ) from error
    return {
        "Ridge": linear_model.Ridge,
        "GroupKFold": model_selection.GroupKFold,
        "KFold": model_selection.KFold,
        "cross_val_predict": model_selection.cross_val_predict,
        "Pipeline": pipeline.Pipeline,
        "StandardScaler": preprocessing.StandardScaler,
    }


def _row_standardized_root_mean_square(
    values: Any,
    feature_means: npt.NDArray[np.float64],
    feature_scales: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    inverse_scales = 1 / feature_scales
    if isinstance(values, sparse.csr_matrix):
        scaled = values.multiply(inverse_scales)
        raw_squared_sum = np.asarray(scaled.multiply(scaled).sum(axis=1)).ravel()
        cross = np.asarray(values @ (feature_means / feature_scales**2)).ravel()
        mean_squared_sum = float(np.sum((feature_means * inverse_scales) ** 2))
        centered_squared_sum = raw_squared_sum - 2 * cross + mean_squared_sum
    else:
        array = np.asarray(values, dtype=np.float64)
        centered_squared_sum = np.sum(((array - feature_means) * inverse_scales) ** 2, axis=1)
    squared_mean = centered_squared_sum / len(feature_means)
    return cast(npt.NDArray[np.float64], np.sqrt(np.maximum(0.0, squared_mean)))
