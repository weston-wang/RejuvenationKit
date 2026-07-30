# Case study: public Dog Aging Project audit

This case study exercises the complete Phase 1 audit on public longitudinal blood-chemistry data
from the [Dog Aging Project](https://doi.org/10.7910/DVN/8C63KB). It tests realistic ingestion,
visit attrition, incomplete multichannel trajectories, held-out calibration, visualization, and
reproducible export.

The data are observational and contain no rapamycin treatment assignment. Nothing in this case
study estimates treatment efficacy.

## Reproduce the audit

Install the visualization extra and run:

```bash
python examples/public_dog_phase1_audit.py \
  --output-dir artifacts/public-dog-phase1-audit
```

The adapter downloads and caches the source archive, selects the first two Precision waves, builds
typed subjects and observations, applies the expected-visit protocol, and makes a deterministic
calibration/evaluation split.

## Results

| Audit result | Value |
|---|---:|
| Subjects | 972 dogs |
| Long-form observations | 6,808 |
| Features | 4 chemistry channels |
| QC errors | 0 |
| QC warnings | 1 |
| Dogs with no required measurements at wave 2 | 237 |
| Lowest wave-2 feature coverage | 75.6% |
| Complete at wave 1 | 967 |
| Complete at both waves | 730 |
| Complete-case retention | 75.5% |
| Reference subjects with complete pairs | 517 |
| Held-out complete trajectories scored | 213 |
| Held-out incomplete trajectories | 76 |
| Held-out detections | 11 |

The observed detection fraction was approximately 5.2%, close to the configured nominal 5%
false-alarm rate. This is a useful calibration check, not evidence that those dogs improved or
deteriorated.

## What the audit revealed

The headline cohort size of 972 overstates the sample available for a complete four-channel paired
analysis. Only 730 dogs were complete at both waves. A workflow based only on enrollment counts
would obscure that reduction.

The missing-follow-up warning is also scientifically meaningful: attrition can bias longitudinal
aging estimates when dogs missing later samples differ systematically at baseline. The audit
therefore exports feature-specific attrition diagnostics rather than treating missingness as a
file-cleaning detail.

Finally, the held-out detector fits its reference mean, covariance, and threshold on a disjoint
calibration group. Subjects used for evaluation do not train the detector. That separation makes
the output more credible than scoring the same animals used to construct the null distribution.

## Reproducible bundle

The workflow exports:

- the complete typed audit and configuration as JSON;
- QC findings and readiness tables as CSV;
- a decision-oriented Markdown summary;
- an overview figure and four DSP diagnostic figures; and
- a manifest containing the byte size and SHA-256 digest of every other artifact.

The validated study model receives a canonical SHA-256 input fingerprint. Re-running with changed
subjects, observations, ordering, or metadata changes that fingerprint.

## What this case does not validate

- Randomized treatment-effect inference
- Rapamycin efficacy
- Assay-specific laboratory QC
- Survival or competing-risk endpoints
- Clinical or regulatory suitability

Those require a dataset with the corresponding design and endpoints.
