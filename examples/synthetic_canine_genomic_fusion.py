"""Demonstrate covariance-aware genomic and multimodal fusion in a canine trial.

All values are synthetic. The workflow illustrates what RejuvenationKit can
measure and diagnose; it does not estimate an observed rapamycin effect in dogs.
"""

from __future__ import annotations

import numpy as np

from rejuvenationkit import (
    EffectDirection,
    EvidenceCovariance,
    EvidenceEstimate,
    EvidenceFusionConfig,
    EvidenceWeightConstraint,
    FusionModel,
    GeneralizedLeastSquaresFusion,
    HierarchicalEvidenceFusion,
    Modality,
)
from rejuvenationkit.genomics import (
    FeatureNamespace,
    GeneSignature,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
    SignatureContrastBatch,
    SignatureContrastConfig,
    SignatureFeature,
    estimate_signature_contrasts,
    score_weighted_signature,
)


def build_synthetic_expression() -> GenomicMatrix:
    """Create correlated mTOR/autophagy/inflammation expression channels."""
    generator = np.random.default_rng(42)
    samples: list[GenomicSample] = []
    rows: list[np.ndarray] = []
    features = tuple(
        GenomicFeature(
            feature_id=feature_id,
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
        )
        for feature_id in (
            "ENSCAFG_MTOR1",
            "ENSCAFG_MTOR2",
            "ENSCAFG_AUTO1",
            "ENSCAFG_AUTO2",
            "ENSCAFG_INFLAM1",
            "ENSCAFG_INFLAM2",
        )
    )
    for cohort, mean_shift in (("control", 0.0), ("rapamycin", 1.0)):
        for index in range(24):
            shared = generator.normal()
            rows.append(
                np.asarray(
                    [
                        shared - mean_shift + generator.normal(scale=0.35),
                        shared - 0.8 * mean_shift + generator.normal(scale=0.35),
                        -shared + mean_shift + generator.normal(scale=0.35),
                        -shared + 0.7 * mean_shift + generator.normal(scale=0.35),
                        0.4 * shared + 0.5 * mean_shift + generator.normal(scale=0.35),
                        0.4 * shared + 0.4 * mean_shift + generator.normal(scale=0.35),
                    ]
                )
            )
            samples.append(
                GenomicSample(
                    sample_id=f"{cohort}-{index}",
                    subject_id=f"dog-{cohort}-{index}",
                    tissue="blood",
                    species_taxon_id=9615,
                    cohort=cohort,
                    assay_id="synthetic-rna-seq",
                    batch_id=f"balanced-batch-{index % 4}",
                )
            )
    return GenomicMatrix(
        values=np.vstack(rows),
        samples=tuple(samples),
        features=features,
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=GenomicMatrixProvenance(
            source_id="synthetic-canine-genomic-fusion-v1",
            preprocessing=("synthetic-balanced-expression",),
        ),
    )


def signatures() -> tuple[GeneSignature, ...]:
    """Return primary-response signatures plus a separate safety-signal target."""
    common = {
        "version": "1.0",
        "namespace": FeatureNamespace.ENSEMBL,
        "species_taxon_id": 9615,
        "tissue": "blood",
        "target_unit": "arbitrary_score_units",
        "direction": EffectDirection.LOWER_IS_BETTER,
        "resource_id": "synthetic-score-signatures-v1-not-a-biological-age-calibration",
    }
    return (
        GeneSignature(
            signature_id="mtorc1-response",
            name="mTORC1 suppression",
            features=(
                SignatureFeature(feature_id="ENSCAFG_MTOR1", weight=1),
                SignatureFeature(feature_id="ENSCAFG_MTOR2", weight=1),
            ),
            target_name="synthetic_primary_response",
            **common,
        ),
        GeneSignature(
            signature_id="autophagy-response",
            name="Autophagy activation",
            features=(
                SignatureFeature(feature_id="ENSCAFG_AUTO1", weight=-1),
                SignatureFeature(feature_id="ENSCAFG_AUTO2", weight=-1),
            ),
            target_name="synthetic_primary_response",
            **common,
        ),
        GeneSignature(
            signature_id="inflammation-putative-safety-signal",
            name="Inflammatory response (putative safety signal)",
            features=(
                SignatureFeature(feature_id="ENSCAFG_INFLAM1", weight=1),
                SignatureFeature(feature_id="ENSCAFG_INFLAM2", weight=1),
            ),
            target_name="synthetic_inflammatory_safety_signal",
            **common,
        ),
    )


def genomic_evidence(
    matrix: GenomicMatrix,
) -> tuple[tuple[EvidenceEstimate, ...], EvidenceCovariance, SignatureContrastBatch]:
    """Score signatures and jointly estimate subject-bootstrap covariance."""
    contrast = SignatureContrastConfig(
        treated_cohort="rapamycin",
        control_cohort="control",
        bootstrap_iterations=1_000,
        random_seed=17,
        covariance_shrinkage=0.2,
        estimand_population="synthetic randomized canine rapamycin cohort",
        time_contrast="synthetic endpoint versus matched control",
    )
    definitions = signatures()
    batch = estimate_signature_contrasts(
        tuple(
            (score_weighted_signature(matrix, signature), signature) for signature in definitions
        ),
        contrast,
    )
    evidence, covariance = batch.to_evidence(
        modality=Modality.TRANSCRIPTOMICS,
        calibration_id="synthetic-score-identity-not-biological-age",
        assay_id="synthetic-rna-seq",
        correlation_group="shared-rna-subjects",
    )
    return evidence, covariance, batch


def main() -> None:
    """Compare naive, covariance-aware, and hierarchical conclusions."""
    expression = build_synthetic_expression()
    estimates, covariance, batch = genomic_evidence(expression)
    primary = estimates[:2]
    safety = estimates[2]
    covariance_array = np.asarray(covariance.covariance, dtype=float)
    primary_covariance = EvidenceCovariance(
        evidence_ids=tuple(item.evidence_id for item in primary),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance_array[:2, :2]),
        source_id=f"{covariance.source_id}:primary-response-block",
        effective_sample_size=covariance.effective_sample_size,
    )
    naive = GeneralizedLeastSquaresFusion().fuse(primary)
    covariance_aware = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(weight_constraint=EvidenceWeightConstraint.NONNEGATIVE)
    ).fuse(primary, primary_covariance)

    clinical = EvidenceEstimate(
        evidence_id="clinical-frailty",
        modality=Modality.CLINICAL,
        estimand=estimates[0].estimand,
        estimate=-0.6,
        standard_error=0.45,
        calibration_id="synthetic-clinical-score-calibration-v1",
        provenance_id="synthetic-canine-clinical-v1",
        tissue="whole-animal",
        species_taxon_id=9615,
    )
    combined = (*primary, clinical)
    full_covariance_array = np.zeros((3, 3), dtype=float)
    full_covariance_array[:2, :2] = np.asarray(primary_covariance.covariance, dtype=float)
    full_covariance_array[2, 2] = clinical.standard_error**2
    full_covariance = EvidenceCovariance(
        evidence_ids=tuple(item.evidence_id for item in combined),
        covariance=tuple(tuple(float(value) for value in row) for row in full_covariance_array),
        source_id="joint-subject-bootstrap-genomics-plus-independent-synthetic-clinical",
        effective_sample_size=covariance.effective_sample_size,
    )
    hierarchical = HierarchicalEvidenceFusion(
        across_modality_model=FusionModel.RANDOM_EFFECTS
    ).fuse(combined, full_covariance)

    print("Synthetic canine rapamycin genomic fusion")
    print(f"Matrix: {expression.shape[0]} samples x {expression.shape[1]} genes")
    for item in estimates:
        print(
            f"  {item.evidence_id:27s} {item.estimate:+.2f} ± {item.standard_error:.2f} score units"
        )
    print(f"Maximum genomic panel correlation: {np.max(np.abs(np.triu(batch.correlation, 1))):.2f}")
    print(
        f"Putative safety signal kept separate: {safety.estimate:+.2f} ± "
        f"{safety.standard_error:.2f} score units"
    )
    print(
        f"Naive independent genomic fusion: {naive.estimate:+.2f} ± "
        f"{naive.standard_error:.2f} score units"
    )
    print(
        f"Covariance-aware genomic fusion: {covariance_aware.estimate:+.2f} ± "
        f"{covariance_aware.standard_error:.2f} score units"
    )
    print(
        "Genomic weight-concentration count: "
        f"{covariance_aware.weight_concentration_effective_count:.2f} "
        f"of {len(primary)} primary signatures (not an independent-information count)"
    )
    print(
        f"Hierarchical multimodal estimate: {hierarchical.across_modality.estimate:+.2f} ± "
        f"{hierarchical.across_modality.standard_error:.2f} score units"
    )
    print(
        "Interpretation: mTOR/autophagy signatures share samples and features, so their covariance "
        "is propagated. The inflammatory safety signal has a different estimand and is never "
        "averaged into the primary response."
    )
    print("These data are synthetic and do not establish canine rapamycin efficacy.")


if __name__ == "__main__":
    main()
