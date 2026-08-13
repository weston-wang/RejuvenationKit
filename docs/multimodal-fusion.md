# Phase 2: uncertainty-aware multimodal fusion

Phase 2 combines commensurate estimates from assay-specific pipelines without discarding their
uncertainty or hiding conflicts between modalities. It is a decision layer above methylation
clocks, transcriptomic signatures, proteomics, imaging, histology, wearable, and clinical
models—not a replacement for those models.

## Statistical model

`PrecisionWeightedFusion` supports two transparent meta-analytic models:

- **Fixed effect** assumes every modality estimates one shared biological effect and weights each
  estimate by inverse variance.
- **Random effects** is the default. It uses a DerSimonian–Laird estimate of between-modality
  variance and widens uncertainty when modalities disagree.

Every input must describe the same target on the same scale. For example, all inputs might estimate
change in biological age in years. A clock-age change in years cannot be fused directly with a
unitless pathway score.

```python
from rejuvenationkit import (
    FusionConfig,
    Modality,
    ModalityCalibration,
    ModalityEstimate,
    PrecisionWeightedFusion,
)

fusion = PrecisionWeightedFusion(
    FusionConfig(
        expected_modalities=(
            Modality.METHYLATION,
            Modality.TRANSCRIPTOMICS,
            Modality.CLINICAL,
        ),
        calibrations=(
            ModalityCalibration(
                modality=Modality.METHYLATION,
                bias=0.4,
                standard_error_scale=1.15,
                calibration_id="canine-clock-holdout-v1",
            ),
        ),
    )
)

result = fusion.fuse(
    (
        ModalityEstimate(
            modality=Modality.METHYLATION,
            estimate=-3.8,
            standard_error=0.9,
            target="biological_age_delta_years",
        ),
        ModalityEstimate(
            modality=Modality.TRANSCRIPTOMICS,
            estimate=-2.6,
            standard_error=1.1,
            target="biological_age_delta_years",
        ),
        ModalityEstimate(
            modality=Modality.CLINICAL,
            estimate=-1.4,
            standard_error=0.8,
            target="biological_age_delta_years",
        ),
    )
)
```

The result contains the fused estimate and confidence interval, normalized modality weights,
Cochran's Q disagreement score, between-modality variance, I² heterogeneity, present and missing
modalities, calibration identifiers, and a refit after omitting each modality.

## Calibration contract

`ModalityCalibration` subtracts a known bias and scales reported uncertainty. These values should
come from a held-out external or cross-validated calibration study with a prespecified reference
target. `calibration_id` should identify the exact validation artifact or analysis version.

Calling `fit(study)` records the study identifier and observation count for each modality. It does
not estimate biological truth from the same unlabeled observations being fused. This avoids a
plausible but circular calibration step and preserves an auditable boundary between assay-specific
validation and multimodal integration.

An identity calibration is explicit in the output when no profile is supplied. Identity does not
mean independently validated; it means that no bias or uncertainty correction was applied.

## Missing modalities

Missing modalities are never imputed. Configure the expected modalities, then choose:

- `ALLOW`: fuse what is present and record all absent modalities;
- `ERROR`: reject an incomplete set; or
- `minimum_modalities`: require a minimum evidence count regardless of modality identity.

This makes an analysis with missing tissue, failed sequencing, or absent imaging distinguishable
from a complete multimodal assessment.

## Sensitivity and disagreement

Random-effects uncertainty increases when estimates conflict, but it does not explain the source
of conflict. Leave-one-modality-out results show how much the fused estimate moves when each assay
is removed. A large shift identifies an influential modality that needs scientific review; it is
not a reason to exclude that modality automatically.

I² is descriptive with few modalities and should not be treated as a definitive hypothesis test.
The toolkit reports the statistic rather than converting it into a pass/fail judgment.

## Interpretation limits

- Inputs must estimate one target on one meaningful scale.
- Standard errors must include the important uncertainty from each upstream pipeline.
- The baseline treats modality estimates as independent. Correlated clocks or assays can make the
  fused standard error too small; covariance-aware fusion requires an externally estimated
  cross-modality covariance matrix and remains future work.
- DerSimonian–Laird variance is a transparent baseline, not an optimal estimator with only two or
  three modalities.
- Calibration and fusion must not reuse evaluation outcomes in ways that leak treatment effects.
- A fused biological-age change is not a clinical endpoint or evidence of improved survival.

See the
[`canine_multimodal_fusion.py` example](https://github.com/weston-wang/RejuvenationKit/blob/main/examples/canine_multimodal_fusion.py)
for a
complete synthetic canine example including disagreement and missing-assay scenarios.
