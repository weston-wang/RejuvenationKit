"""Genome-scale signature scoring and uncertainty-aware contrasts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite, sqrt
from statistics import NormalDist
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse

from rejuvenationkit.evidence import (
    CalibrationReference,
    EffectDirection,
    Estimand,
    EvidenceCovariance,
    EvidenceEstimate,
)
from rejuvenationkit.genomics.effects import FeatureEffect, FeatureEffectBatch
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicMatrix, MatrixScale
from rejuvenationkit.schemas import Modality


class MissingFeaturePolicy(StrEnum):
    """Handling of absent signature features or sample-level missing values."""

    ERROR = "error"
    DROP_AND_RENORMALIZE = "drop_and_renormalize"
    IMPUTE_ZERO = "impute_zero"


class SignatureContrastMode(StrEnum):
    """How sample-level scores become one value per independent subject."""

    CROSS_SECTIONAL = "cross_sectional"
    PAIRED_CHANGE = "paired_change"


class IncompletePairPolicy(StrEnum):
    """Handling of subjects without both prespecified longitudinal windows."""

    ERROR = "error"
    EXCLUDE = "exclude"


class SignatureTimeWindow(BaseModel):
    """Inclusive, timezone-aware sample-selection window."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require an ordered pair of timezone-aware timestamps."""
        for value in (self.start, self.end):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("signature time-window bounds must be timezone-aware")
        if self.end < self.start:
            raise ValueError("signature time-window end must not precede start")
        return self


class SignatureFeature(BaseModel):
    """One signed and weighted member of a genomic signature."""

    model_config = ConfigDict(frozen=True)

    feature_id: str = Field(min_length=1)
    weight: float

    @model_validator(mode="after")
    def validate_weight(self) -> Self:
        """Require a finite, nonzero feature weight."""
        if not isfinite(self.weight) or self.weight == 0:
            raise ValueError("signature feature weight must be finite and nonzero")
        return self


class GeneSignature(BaseModel):
    """Versioned, species- and namespace-specific genomic feature signature."""

    model_config = ConfigDict(frozen=True)

    signature_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    features: tuple[SignatureFeature, ...]
    namespace: FeatureNamespace
    species_taxon_id: int = Field(gt=0)
    tissue: str | None = None
    target_name: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    direction: EffectDirection = EffectDirection.UNSPECIFIED
    resource_id: str = Field(min_length=1)
    allowed_scales: tuple[MatrixScale, ...] = (MatrixScale.NORMALIZED_EXPRESSION,)

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        """Reject empty and duplicate feature definitions."""
        if not self.features:
            raise ValueError("signature must contain at least one feature")
        identifiers = [item.feature_id for item in self.features]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("signature feature identifiers must be unique")
        if not self.allowed_scales or len(set(self.allowed_scales)) != len(self.allowed_scales):
            raise ValueError("signature allowed_scales must be nonempty and unique")
        return self

    @property
    def fingerprint(self) -> str:
        """Hash the complete signature definition, not only its display version."""
        return sha256(self.model_dump_json().encode()).hexdigest()


class SignatureSampleScore(BaseModel):
    """One sample-level score and observed signature-weight fraction."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    subject_id: str
    cohort: str | None
    timestamp: datetime | None = None
    score: float
    observed_weight_fraction: float = Field(ge=0, le=1)


class SignatureScores(BaseModel):
    """Aligned sample scores with feature-coverage and provenance details."""

    model_config = ConfigDict(frozen=True)

    signature_id: str
    signature_version: str
    signature_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: tuple[SignatureSampleScore, ...]
    matched_feature_ids: tuple[str, ...]
    missing_feature_ids: tuple[str, ...]
    feature_coverage: float = Field(ge=0, le=1)
    matrix_content_hash: str
    matrix_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_assignment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    matrix_provenance_id: str
    matrix_scale: str
    species_taxon_id: int = Field(gt=0)
    tissue: str | None
    warnings: tuple[str, ...] = ()

    def to_frame(self) -> pd.DataFrame:
        """Return one row per sample for downstream study joins."""
        return pd.DataFrame([item.model_dump() for item in self.scores])


class SignatureContrastConfig(BaseModel):
    """Bootstrap policy for a subject-clustered group contrast."""

    model_config = ConfigDict(frozen=True)

    treated_cohort: str = Field(min_length=1)
    control_cohort: str = Field(min_length=1)
    bootstrap_iterations: int = Field(default=2_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    random_seed: int = 0
    minimum_subjects_per_group: int = Field(default=3, ge=2)
    covariance_shrinkage: float = Field(default=0.1, ge=0, le=1)
    estimand_population: str | None = None
    time_contrast: str | None = None
    mode: SignatureContrastMode = SignatureContrastMode.CROSS_SECTIONAL
    baseline_window: SignatureTimeWindow | None = None
    followup_window: SignatureTimeWindow | None = None
    incomplete_pair_policy: IncompletePairPolicy = IncompletePairPolicy.ERROR

    @model_validator(mode="after")
    def validate_longitudinal_policy(self) -> Self:
        """Require explicit, non-overlapping windows for a claimed change estimand."""
        if self.mode is SignatureContrastMode.CROSS_SECTIONAL:
            if self.baseline_window is not None or self.followup_window is not None:
                raise ValueError("time windows require paired_change mode")
            if self.incomplete_pair_policy is not IncompletePairPolicy.ERROR:
                raise ValueError("incomplete_pair_policy applies only to paired_change mode")
            return self
        if self.baseline_window is None or self.followup_window is None:
            raise ValueError("paired_change mode requires baseline_window and followup_window")
        if self.baseline_window.end >= self.followup_window.start:
            raise ValueError("baseline and follow-up windows must be ordered and non-overlapping")
        if self.time_contrast is None:
            raise ValueError("paired_change mode requires a declared time_contrast")
        return self


class SignatureEstimate(BaseModel):
    """One genomic signature contrast with subject-level uncertainty."""

    model_config = ConfigDict(frozen=True)

    signature_id: str
    signature_version: str
    contrast: str
    estimand_population: str
    time_contrast: str | None = None
    estimate: float
    standard_error: float = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    confidence_interval: tuple[float, float]
    target_name: str
    target_unit: str
    direction: EffectDirection
    treated_subjects: int = Field(ge=0)
    control_subjects: int = Field(ge=0)
    matched_feature_ids: tuple[str, ...]
    missing_feature_ids: tuple[str, ...]
    feature_coverage: float = Field(ge=0, le=1)
    uncertainty_method: str
    provenance_id: str
    species_taxon_id: int = Field(gt=0)
    tissue: str | None
    excluded_subject_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_evidence(
        self,
        *,
        evidence_id: str,
        modality: Modality,
        calibration_id: str | None = None,
        calibration_reference: CalibrationReference | None = None,
        assay_id: str | None = None,
        correlation_group: str | None = None,
    ) -> EvidenceEstimate:
        """Convert a calibrated signature result into evidence for Phase 2 fusion."""
        resolved_calibration_id = calibration_id or (
            calibration_reference.calibration_id if calibration_reference is not None else None
        )
        if resolved_calibration_id is None:
            raise ValueError("calibration_id or calibration_reference is required")
        if (
            calibration_reference is not None
            and calibration_reference.calibration_id != resolved_calibration_id
        ):
            raise ValueError("calibration_id and calibration_reference disagree")
        return EvidenceEstimate(
            evidence_id=evidence_id,
            modality=modality,
            estimand=Estimand(
                name=self.target_name,
                unit=self.target_unit,
                direction=self.direction,
                population=self.estimand_population,
                time_contrast=self.time_contrast,
            ),
            estimate=self.estimate,
            standard_error=self.standard_error,
            calibration_id=resolved_calibration_id,
            calibration_reference=calibration_reference,
            provenance_id=self.provenance_id,
            tissue=self.tissue,
            species_taxon_id=self.species_taxon_id,
            assay_id=assay_id,
            correlation_group=correlation_group,
            quality_flags=self.warnings,
        )


class SignatureContrastBatch(BaseModel):
    """Joint signature contrasts with subject-bootstrap covariance."""

    model_config = ConfigDict(frozen=True)

    estimates: tuple[SignatureEstimate, ...]
    covariance: EvidenceCovariance
    correlation: tuple[tuple[float, ...], ...]
    bootstrap_iterations: int = Field(ge=100)
    random_seed: int
    treated_subject_ids: tuple[str, ...]
    control_subject_ids: tuple[str, ...]
    excluded_subject_ids: tuple[str, ...] = ()

    def to_evidence(
        self,
        *,
        modality: Modality,
        calibration_id: str | None = None,
        calibration_reference: CalibrationReference | None = None,
        assay_id: str | None = None,
        correlation_group: str | None = None,
    ) -> tuple[tuple[EvidenceEstimate, ...], EvidenceCovariance]:
        """Convert joint estimates while retaining covariance across unlike targets.

        Covariance is useful for a multivariate response audit. Only evidence that
        shares a complete estimand may be passed together to a fusion estimator.
        """
        evidence = tuple(
            item.to_evidence(
                evidence_id=item.signature_id,
                modality=modality,
                calibration_id=calibration_id,
                calibration_reference=calibration_reference,
                assay_id=assay_id,
                correlation_group=correlation_group,
            )
            for item in self.estimates
        )
        return evidence, self.covariance


def score_weighted_signature(
    matrix: GenomicMatrix,
    signature: GeneSignature,
    *,
    missing_policy: MissingFeaturePolicy = MissingFeaturePolicy.DROP_AND_RENORMALIZE,
    minimum_feature_coverage: float = 0.7,
    minimum_sample_coverage: float = 0.7,
) -> SignatureScores:
    """Calculate signed weighted-mean scores without expanding matrix rows."""
    if not 0 < minimum_feature_coverage <= 1:
        raise ValueError("minimum_feature_coverage must lie in (0, 1]")
    if not 0 < minimum_sample_coverage <= 1:
        raise ValueError("minimum_sample_coverage must lie in (0, 1]")
    if matrix.scale not in signature.allowed_scales:
        raise ValueError(
            f"matrix scale {matrix.scale.value} is not allowed by signature "
            f"{signature.signature_id}"
        )
    feature_types = {item.feature_type.value for item in matrix.features}
    if feature_types != {"gene"}:
        raise ValueError("weighted gene signatures require gene features")
    species = {item.species_taxon_id for item in matrix.samples}
    if species != {signature.species_taxon_id}:
        raise ValueError("matrix and signature species_taxon_id must match")
    tissues = {item.tissue for item in matrix.samples}
    if len(tissues) != 1:
        raise ValueError("one signature score matrix cannot pool multiple tissues")
    observed_tissue = next(iter(tissues))
    namespaces = {item.namespace for item in matrix.features}
    if namespaces != {signature.namespace}:
        raise ValueError("matrix and signature feature namespace must match exactly")
    if signature.tissue is not None:
        if tissues != {signature.tissue}:
            raise ValueError("matrix sample tissue does not match signature tissue")

    matrix_lookup = {identifier: index for index, identifier in enumerate(matrix.feature_ids)}
    matched = tuple(item for item in signature.features if item.feature_id in matrix_lookup)
    missing = tuple(
        item.feature_id for item in signature.features if item.feature_id not in matrix_lookup
    )
    total_weight = sum(abs(item.weight) for item in signature.features)
    matched_weight = sum(abs(item.weight) for item in matched)
    coverage = matched_weight / total_weight
    if missing and missing_policy is MissingFeaturePolicy.ERROR:
        raise ValueError(f"signature features are absent from matrix: {list(missing)}")
    if not matched:
        raise ValueError("no signature features are present in the matrix")
    if coverage < minimum_feature_coverage:
        raise ValueError(
            f"signature feature coverage {coverage:.3f} is below minimum "
            f"{minimum_feature_coverage:.3f}"
        )
    indices = tuple(matrix_lookup[item.feature_id] for item in matched)
    if isinstance(matrix.values, sparse.csr_matrix):
        values: Any = cast(Any, matrix.values)[:, list(indices)]
    else:
        values = matrix.values[:, list(indices)]
    dense = values.toarray() if isinstance(values, sparse.csr_matrix) else np.asarray(values)
    dense = np.asarray(dense, dtype=float)
    weights = np.asarray([item.weight for item in matched], dtype=float)
    denominator_if_zero = (
        total_weight if missing_policy is MissingFeaturePolicy.IMPUTE_ZERO else None
    )
    sample_scores: list[SignatureSampleScore] = []
    for row_index, sample in enumerate(matrix.samples):
        row = dense[row_index]
        observed = np.isfinite(row)
        observed_weight = float(np.abs(weights[observed]).sum())
        if not observed.all() and missing_policy is MissingFeaturePolicy.ERROR:
            raise ValueError(f"sample {sample.sample_id} has missing signature values")
        if observed_weight == 0:
            raise ValueError(f"sample {sample.sample_id} has no observed signature weight")
        observed_fraction = observed_weight / total_weight
        if observed_fraction < minimum_sample_coverage:
            raise ValueError(
                f"sample {sample.sample_id} signature coverage {observed_fraction:.3f} is below "
                f"minimum {minimum_sample_coverage:.3f}"
            )
        denominator = denominator_if_zero or observed_weight
        score = float(weights[observed] @ row[observed] / denominator)
        sample_scores.append(
            SignatureSampleScore(
                sample_id=sample.sample_id,
                subject_id=sample.subject_id,
                cohort=sample.cohort,
                timestamp=sample.timestamp,
                score=score,
                observed_weight_fraction=observed_fraction,
            )
        )
    warnings: list[str] = []
    if missing:
        warnings.append("signature_features_missing")
    if any(item.observed_weight_fraction < 1 for item in sample_scores):
        warnings.append("sample_signature_values_missing")
    return SignatureScores(
        signature_id=signature.signature_id,
        signature_version=signature.version,
        signature_fingerprint=signature.fingerprint,
        scores=tuple(sample_scores),
        matched_feature_ids=tuple(item.feature_id for item in matched),
        missing_feature_ids=missing,
        feature_coverage=coverage,
        matrix_content_hash=matrix.content_hash,
        matrix_artifact_hash=matrix.artifact_hash,
        sample_assignment_hash=sha256(
            "|".join(
                f"{item.sample_id}:{item.subject_id}:{item.cohort}:"
                f"{item.timestamp.isoformat() if item.timestamp is not None else ''}"
                for item in matrix.samples
            ).encode()
        ).hexdigest(),
        matrix_provenance_id=matrix.provenance.source_id,
        matrix_scale=matrix.scale.value,
        species_taxon_id=signature.species_taxon_id,
        tissue=observed_tissue,
        warnings=tuple(warnings),
    )


def estimate_signature_contrast(
    scores: SignatureScores,
    signature: GeneSignature,
    config: SignatureContrastConfig,
) -> SignatureEstimate:
    """Estimate treated-minus-control change with a subject-cluster bootstrap."""
    if (scores.signature_id, scores.signature_version) != (
        signature.signature_id,
        signature.version,
    ):
        raise ValueError("scores and signature identifiers must match")
    if scores.signature_fingerprint != signature.fingerprint:
        raise ValueError("scores and signature definitions must match exactly")
    frame = scores.to_frame()
    treated, treated_excluded = _subject_values(
        frame.loc[frame["cohort"] == config.treated_cohort], config
    )
    controls, control_excluded = _subject_values(
        frame.loc[frame["cohort"] == config.control_cohort], config
    )
    _reject_cohort_subject_overlap(treated, controls)
    if len(treated) < config.minimum_subjects_per_group:
        raise ValueError("treated cohort has too few independent subjects")
    if len(controls) < config.minimum_subjects_per_group:
        raise ValueError("control cohort has too few independent subjects")
    estimate = float(treated.mean() - controls.mean())
    generator = np.random.default_rng(config.random_seed)
    bootstrap = np.empty(config.bootstrap_iterations, dtype=float)
    treated_values = treated.to_numpy(dtype=float)
    control_values = controls.to_numpy(dtype=float)
    for index in range(config.bootstrap_iterations):
        treated_sample = generator.choice(treated_values, size=len(treated_values), replace=True)
        control_sample = generator.choice(control_values, size=len(control_values), replace=True)
        bootstrap[index] = treated_sample.mean() - control_sample.mean()
    standard_error = float(bootstrap.std(ddof=1))
    if standard_error <= 0 or not isfinite(standard_error):
        raise ValueError("bootstrap produced zero or non-finite uncertainty")
    alpha = (1 - config.confidence_level) / 2
    interval = tuple(float(value) for value in np.quantile(bootstrap, [alpha, 1 - alpha]))
    contrast = f"{config.treated_cohort}-minus-{config.control_cohort}"
    config_fingerprint = sha256(config.model_dump_json().encode()).hexdigest()
    excluded = tuple(sorted((*treated_excluded, *control_excluded)))
    warnings = list(scores.warnings)
    if excluded:
        warnings.append("incomplete_longitudinal_pairs_excluded")
    return SignatureEstimate(
        signature_id=signature.signature_id,
        signature_version=signature.version,
        contrast=contrast,
        estimand_population=config.estimand_population or contrast,
        time_contrast=config.time_contrast,
        estimate=estimate,
        standard_error=standard_error,
        confidence_level=config.confidence_level,
        confidence_interval=(interval[0], interval[1]),
        target_name=signature.target_name,
        target_unit=signature.target_unit,
        direction=signature.direction,
        treated_subjects=len(treated),
        control_subjects=len(controls),
        matched_feature_ids=scores.matched_feature_ids,
        missing_feature_ids=scores.missing_feature_ids,
        feature_coverage=scores.feature_coverage,
        uncertainty_method=f"subject_cluster_bootstrap:{config.bootstrap_iterations}",
        provenance_id=(
            f"{scores.matrix_provenance_id}:{scores.matrix_artifact_hash}:"
            f"{scores.sample_assignment_hash}:{scores.signature_fingerprint}:"
            f"config={config_fingerprint}"
        ),
        warnings=tuple(warnings),
        species_taxon_id=scores.species_taxon_id,
        tissue=scores.tissue,
        excluded_subject_ids=excluded,
    )


def estimate_signature_contrasts(
    inputs: tuple[tuple[SignatureScores, GeneSignature], ...],
    config: SignatureContrastConfig,
) -> SignatureContrastBatch:
    """Estimate several signatures with one aligned subject-cluster bootstrap.

    Joint resampling preserves covariance caused by shared animals and is the
    preferred bridge from several pathway or clock scores into evidence-level
    generalized least squares.
    """
    if not inputs:
        raise ValueError("at least one signature score set is required")
    identifiers = [signature.signature_id for _, signature in inputs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("joint signature identifiers must be unique")
    treated_by_signature: list[pd.Series] = []
    control_by_signature: list[pd.Series] = []
    excluded_by_signature: list[tuple[str, ...]] = []
    for scores, signature in inputs:
        if (scores.signature_id, scores.signature_version) != (
            signature.signature_id,
            signature.version,
        ):
            raise ValueError("scores and signature identifiers must match")
        if scores.signature_fingerprint != signature.fingerprint:
            raise ValueError("scores and signature definitions must match exactly")
        frame = scores.to_frame()
        treated, treated_excluded = _subject_values(
            frame.loc[frame["cohort"] == config.treated_cohort], config
        )
        controls, control_excluded = _subject_values(
            frame.loc[frame["cohort"] == config.control_cohort], config
        )
        _reject_cohort_subject_overlap(treated, controls)
        if len(treated) < config.minimum_subjects_per_group:
            raise ValueError("treated cohort has too few independent subjects")
        if len(controls) < config.minimum_subjects_per_group:
            raise ValueError("control cohort has too few independent subjects")
        treated_by_signature.append(treated)
        control_by_signature.append(controls)
        excluded_by_signature.append(tuple(sorted((*treated_excluded, *control_excluded))))

    treated_ids = tuple(str(value) for value in treated_by_signature[0].index)
    control_ids = tuple(str(value) for value in control_by_signature[0].index)
    treated_mismatch = any(
        tuple(str(value) for value in series.index) != treated_ids
        for series in treated_by_signature
    )
    if treated_mismatch:
        raise ValueError("joint signatures must contain identical treated subjects in one order")
    control_mismatch = any(
        tuple(str(value) for value in series.index) != control_ids
        for series in control_by_signature
    )
    if control_mismatch:
        raise ValueError("joint signatures must contain identical control subjects in one order")
    if any(item != excluded_by_signature[0] for item in excluded_by_signature):
        raise ValueError("joint signatures must exclude identical longitudinal subjects")

    treated_values = np.vstack([series.to_numpy(dtype=float) for series in treated_by_signature])
    control_values = np.vstack([series.to_numpy(dtype=float) for series in control_by_signature])
    point_estimates = treated_values.mean(axis=1) - control_values.mean(axis=1)
    generator = np.random.default_rng(config.random_seed)
    bootstrap = np.empty((config.bootstrap_iterations, len(inputs)), dtype=float)
    for index in range(config.bootstrap_iterations):
        treated_indices = generator.integers(0, len(treated_ids), size=len(treated_ids))
        control_indices = generator.integers(0, len(control_ids), size=len(control_ids))
        bootstrap[index] = treated_values[:, treated_indices].mean(axis=1) - control_values[
            :, control_indices
        ].mean(axis=1)
    raw_covariance = np.atleast_2d(np.cov(bootstrap, rowvar=False, ddof=1))
    diagonal_target = np.diag(np.diag(raw_covariance))
    covariance_array = (
        1 - config.covariance_shrinkage
    ) * raw_covariance + config.covariance_shrinkage * diagonal_target
    standard_errors = np.sqrt(np.diag(covariance_array))
    if not np.isfinite(covariance_array).all() or np.any(standard_errors <= 0):
        raise ValueError("joint bootstrap produced invalid covariance")
    alpha = (1 - config.confidence_level) / 2
    intervals = np.quantile(bootstrap, [alpha, 1 - alpha], axis=0)
    denominator = np.outer(standard_errors, standard_errors)
    correlation = covariance_array / denominator
    np.fill_diagonal(correlation, 1.0)
    contrast = f"{config.treated_cohort}-minus-{config.control_cohort}"
    config_fingerprint = sha256(config.model_dump_json().encode()).hexdigest()

    estimates: list[SignatureEstimate] = []
    for item_index, (scores, signature) in enumerate(inputs):
        item_warnings = list(scores.warnings)
        if excluded_by_signature[item_index]:
            item_warnings.append("incomplete_longitudinal_pairs_excluded")
        estimates.append(
            SignatureEstimate(
                signature_id=signature.signature_id,
                signature_version=signature.version,
                contrast=contrast,
                estimand_population=config.estimand_population or contrast,
                time_contrast=config.time_contrast,
                estimate=float(point_estimates[item_index]),
                standard_error=float(standard_errors[item_index]),
                confidence_level=config.confidence_level,
                confidence_interval=(
                    float(intervals[0, item_index]),
                    float(intervals[1, item_index]),
                ),
                target_name=signature.target_name,
                target_unit=signature.target_unit,
                direction=signature.direction,
                treated_subjects=len(treated_ids),
                control_subjects=len(control_ids),
                matched_feature_ids=scores.matched_feature_ids,
                missing_feature_ids=scores.missing_feature_ids,
                feature_coverage=scores.feature_coverage,
                uncertainty_method=(
                    f"joint_subject_cluster_bootstrap:{config.bootstrap_iterations}"
                ),
                provenance_id=(
                    f"{scores.matrix_provenance_id}:{scores.matrix_artifact_hash}:"
                    f"{scores.sample_assignment_hash}:{scores.signature_fingerprint}:"
                    f"config={config_fingerprint}"
                ),
                warnings=tuple(item_warnings),
                species_taxon_id=scores.species_taxon_id,
                tissue=scores.tissue,
                excluded_subject_ids=excluded_by_signature[item_index],
            )
        )
    evidence_ids = tuple(item.signature_id for item in estimates)
    input_fingerprint = sha256(
        "|".join(
            f"{scores.matrix_artifact_hash}:{scores.sample_assignment_hash}:"
            f"{scores.signature_fingerprint}"
            for scores, _ in inputs
        ).encode()
    ).hexdigest()
    covariance = EvidenceCovariance(
        evidence_ids=evidence_ids,
        covariance=tuple(tuple(float(value) for value in row) for row in covariance_array),
        source_id=(
            f"joint-subject-bootstrap:{contrast}:iterations={config.bootstrap_iterations}:"
            f"diagonal-shrinkage={config.covariance_shrinkage:g}:config={config_fingerprint}"
            f":inputs={input_fingerprint}"
        ),
        effective_sample_size=len(treated_ids) + len(control_ids),
    )
    return SignatureContrastBatch(
        estimates=tuple(estimates),
        covariance=covariance,
        correlation=tuple(tuple(float(value) for value in row) for row in correlation),
        bootstrap_iterations=config.bootstrap_iterations,
        random_seed=config.random_seed,
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
        excluded_subject_ids=excluded_by_signature[0],
    )


def aggregate_feature_effects(
    effects: tuple[FeatureEffect, ...] | FeatureEffectBatch,
    signature: GeneSignature,
    *,
    covariance: pd.DataFrame | None = None,
    missing_policy: MissingFeaturePolicy = MissingFeaturePolicy.DROP_AND_RENORMALIZE,
    minimum_feature_coverage: float = 0.7,
    confidence_level: float = 0.95,
) -> SignatureEstimate:
    """Aggregate upstream feature effects with optional correlation uncertainty."""
    if not 0 < minimum_feature_coverage <= 1:
        raise ValueError("minimum_feature_coverage must lie in (0, 1]")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    effect_batch: FeatureEffectBatch | None
    effect_values: tuple[FeatureEffect, ...]
    if isinstance(effects, FeatureEffectBatch):
        effect_batch = effects
        effect_values = effects.effects
    else:
        effect_batch = None
        effect_values = effects
    if not effect_values:
        raise ValueError("feature effects cannot be empty")
    identifiers = [item.feature_id for item in effect_values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("feature-effect identifiers must be unique")
    reference = effect_values[0]
    compatibility = {
        (
            item.feature_type,
            item.namespace,
            item.modality,
            item.contrast,
            item.effect_unit,
            item.species_taxon_id,
            item.tissue,
            item.genome_assembly,
            item.provenance_id,
        )
        for item in effect_values
    }
    if len(compatibility) != 1:
        raise ValueError(
            "feature effects must share feature type, namespace, modality, contrast, unit, "
            "species, tissue, assembly, and provenance"
        )
    if reference.feature_type.value != "gene":
        raise ValueError("gene signatures require gene-level feature effects")
    if reference.namespace is not signature.namespace:
        raise ValueError("feature effects and signature namespaces must match")
    if reference.species_taxon_id != signature.species_taxon_id:
        raise ValueError("feature effects and signature species must match")
    if signature.tissue is not None and reference.tissue != signature.tissue:
        raise ValueError("feature effects and signature tissues must match")
    lookup = {item.feature_id: item for item in effect_values}
    matched = tuple(item for item in signature.features if item.feature_id in lookup)
    missing = tuple(item.feature_id for item in signature.features if item.feature_id not in lookup)
    total_weight = sum(abs(item.weight) for item in signature.features)
    matched_weight = sum(abs(item.weight) for item in matched)
    coverage = matched_weight / total_weight
    if missing and missing_policy is MissingFeaturePolicy.ERROR:
        raise ValueError(f"signature features are absent from effects: {list(missing)}")
    if not matched or coverage < minimum_feature_coverage:
        raise ValueError("insufficient signature feature coverage")
    normalization = (
        total_weight if missing_policy is MissingFeaturePolicy.IMPUTE_ZERO else matched_weight
    )
    weights = np.asarray([item.weight for item in matched], dtype=float) / normalization
    matched_effects = tuple(lookup[item.feature_id] for item in matched)
    observed = np.asarray([item.effect for item in matched_effects])
    if covariance is None:
        matrix = np.diag([item.standard_error**2 for item in matched_effects])
        uncertainty_method = "independent_feature_delta_method"
        warnings = ["feature_covariance_not_supplied"]
    else:
        expected = [item.feature_id for item in matched]
        if set(covariance.index) != set(expected) or set(covariance.columns) != set(expected):
            raise ValueError(
                "feature covariance labels must exactly match included signature features"
            )
        matrix = covariance.loc[expected, expected].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("feature covariance must be finite")
        marginal_variances = np.diag(matrix)
        if np.any(marginal_variances <= 0):
            raise ValueError("feature covariance diagonal must be positive")
        marginal_errors = np.sqrt(marginal_variances)
        standardized = matrix / np.outer(marginal_errors, marginal_errors)
        if not np.allclose(standardized, standardized.T, rtol=1e-10, atol=1e-12):
            raise ValueError("feature covariance must be symmetric relative to its marginal scales")
        if float(np.linalg.eigvalsh(standardized)[0]) < -1e-10:
            raise ValueError("feature covariance must be positive semidefinite")
        reported = np.asarray([item.standard_error**2 for item in matched_effects])
        if np.any(np.abs(marginal_variances - reported) / reported > 1e-6):
            raise ValueError("feature covariance diagonal must match reported standard errors")
        uncertainty_method = "correlation_aware_delta_method"
        warnings = []
    if effect_batch is None:
        warnings.append("feature_effect_context_not_supplied")
    estimate = float(weights @ observed)
    variance = float(weights @ matrix @ weights)
    if variance <= 0:
        raise ValueError("aggregated signature variance must be positive")
    standard_error = sqrt(variance)
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    if missing:
        warnings.append("signature_features_missing")
    return SignatureEstimate(
        signature_id=signature.signature_id,
        signature_version=signature.version,
        contrast=reference.contrast,
        estimand_population=(
            effect_batch.estimand_population if effect_batch is not None else reference.contrast
        ),
        time_contrast=effect_batch.time_contrast if effect_batch is not None else None,
        estimate=estimate,
        standard_error=standard_error,
        confidence_level=confidence_level,
        confidence_interval=(
            estimate - z_value * standard_error,
            estimate + z_value * standard_error,
        ),
        target_name=signature.target_name,
        target_unit=signature.target_unit,
        direction=signature.direction,
        treated_subjects=effect_batch.treated_subjects if effect_batch is not None else 0,
        control_subjects=effect_batch.control_subjects if effect_batch is not None else 0,
        matched_feature_ids=tuple(item.feature_id for item in matched),
        missing_feature_ids=missing,
        feature_coverage=coverage,
        uncertainty_method=uncertainty_method,
        provenance_id=(
            f"{effect_batch.provenance_id}:{effect_batch.artifact_hash}:"
            f"signature={signature.fingerprint}"
            if effect_batch is not None
            else f"{reference.provenance_id}:signature={signature.fingerprint}"
        ),
        warnings=tuple(warnings),
        species_taxon_id=signature.species_taxon_id,
        tissue=reference.tissue,
    )


def _subject_values(
    frame: pd.DataFrame,
    config: SignatureContrastConfig,
) -> tuple[pd.Series, tuple[str, ...]]:
    """Resolve exactly one cross-sectional value or paired change per subject."""
    if frame.empty:
        return pd.Series(dtype=float), ()
    if config.mode is SignatureContrastMode.CROSS_SECTIONAL:
        for subject_id, rows in frame.groupby("subject_id", sort=True):
            timestamps = rows["timestamp"]
            if timestamps.isna().any() and timestamps.notna().any():
                raise ValueError(f"subject {subject_id} mixes dated and undated signature samples")
            if timestamps.dropna().nunique() > 1:
                raise ValueError(
                    "repeated longitudinal signature samples require paired_change mode; "
                    f"subject {subject_id} has multiple timestamps"
                )
        return frame.groupby("subject_id", sort=True)["score"].mean(), ()

    baseline = config.baseline_window
    followup = config.followup_window
    if baseline is None or followup is None:
        raise RuntimeError("paired-change windows were not validated")
    if frame["timestamp"].isna().any():
        raise ValueError("paired_change mode requires a timestamp for every signature sample")
    values: dict[str, float] = {}
    excluded: list[str] = []
    for subject_id, rows in frame.groupby("subject_id", sort=True):
        timestamps = pd.to_datetime(rows["timestamp"], utc=True)
        baseline_rows = rows.loc[
            (timestamps >= baseline.start) & (timestamps <= baseline.end), "score"
        ]
        followup_rows = rows.loc[
            (timestamps >= followup.start) & (timestamps <= followup.end), "score"
        ]
        if baseline_rows.empty or followup_rows.empty:
            if config.incomplete_pair_policy is IncompletePairPolicy.ERROR:
                raise ValueError(
                    f"subject {subject_id} lacks a sample in both longitudinal windows"
                )
            excluded.append(str(subject_id))
            continue
        values[str(subject_id)] = float(followup_rows.mean() - baseline_rows.mean())
    return pd.Series(values, dtype=float).sort_index(), tuple(sorted(excluded))


def _reject_cohort_subject_overlap(treated: pd.Series, controls: pd.Series) -> None:
    overlapping = sorted(set(treated.index).intersection(controls.index), key=str)
    if overlapping:
        raise ValueError(
            f"subjects cannot appear in both treated and control cohorts: {overlapping}"
        )
