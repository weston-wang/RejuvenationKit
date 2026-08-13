"""Demonstrate Phase 2 fusion on synthetic canine treatment-response estimates."""

from rejuvenationkit import (
    FusionConfig,
    MissingModalityPolicy,
    Modality,
    ModalityCalibration,
    ModalityEstimate,
    PrecisionWeightedFusion,
)

TARGET = "biological_age_delta_years"


def response(modality: Modality, estimate: float, standard_error: float) -> ModalityEstimate:
    """Create one synthetic, assay-specific treatment-response estimate."""
    return ModalityEstimate(
        modality=modality,
        estimate=estimate,
        standard_error=standard_error,
        target=TARGET,
    )


def print_result(name: str, result: object) -> None:
    """Print a compact human-readable result without adding a table dependency."""
    from rejuvenationkit.fusion import FusionResult

    if not isinstance(result, FusionResult):
        raise TypeError("result must be a FusionResult")
    lower, upper = result.confidence_interval
    missing = ", ".join(item.value for item in result.missing_modalities) or "none"
    print(f"\n{name}")
    print(f"  fused change: {result.estimate:.2f} years")
    print(f"  {result.confidence_level:.0%} CI: [{lower:.2f}, {upper:.2f}]")
    print(f"  I² disagreement: {result.heterogeneity_i2:.1%}")
    print(f"  maximum leave-one-out shift: {result.maximum_leave_one_out_shift:.2f} years")
    print(f"  missing modalities: {missing}")


def main() -> None:
    """Run coherent, conflicting, and missing-modality scenarios."""
    expected = (
        Modality.METHYLATION,
        Modality.TRANSCRIPTOMICS,
        Modality.PROTEOMICS,
        Modality.CLINICAL,
    )
    fusion = PrecisionWeightedFusion(
        FusionConfig(
            expected_modalities=expected,
            missing_modality_policy=MissingModalityPolicy.ALLOW,
            minimum_modalities=2,
            calibrations=(
                ModalityCalibration(
                    modality=Modality.METHYLATION,
                    bias=0.4,
                    standard_error_scale=1.15,
                    calibration_id="synthetic-canine-clock-holdout-v1",
                ),
            ),
        )
    )

    coherent = (
        response(Modality.METHYLATION, -3.8, 0.9),
        response(Modality.TRANSCRIPTOMICS, -2.9, 1.0),
        response(Modality.PROTEOMICS, -2.5, 1.2),
        response(Modality.CLINICAL, -2.1, 0.8),
    )
    conflicting = (*coherent[:-1], response(Modality.CLINICAL, 1.2, 0.8))
    missing_proteomics = tuple(
        item for item in coherent if item.modality is not Modality.PROTEOMICS
    )

    print("Synthetic demonstration only; these are not measured canine treatment outcomes.")
    print_result("Coherent response", fusion.fuse(coherent))
    print_result("Conflicting clinical response", fusion.fuse(conflicting))
    print_result("Missing proteomics", fusion.fuse(missing_proteomics))


if __name__ == "__main__":
    main()
