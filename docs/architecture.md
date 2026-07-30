# Architecture

The package separates validated data contracts from estimation algorithms:

1. `schemas` validates study metadata and long-form observations.
2. `qc` defines immutable feature and expected-visit policies, then produces structured findings
   without mutating input data.
3. `profiling` reuses those visit policies to quantify coverage, retention, and paired-analysis
   readiness.
4. `detection` whitens correlated longitudinal changes and applies empirically calibrated
   multivariate detection thresholds.
5. `sequential` learns reference aging dynamics and monitors onset and persistence across repeated
   multimodal visits.
6. `treatment_effect` uses out-of-fold control calibration and randomized-label inference to
   estimate longitudinal group effects.
7. `audit` orchestrates Phase 1 checks and inference into a reproducible human- and
   machine-readable report bundle.
8. `fusion` converts modality-specific evidence into an uncertain joint estimate.
9. `state` tracks latent biological state through time.
10. `combinations` estimates interaction effects for multi-intervention experiments.

All estimators follow `fit`/`predict`-style protocols and return typed results. Implementations
should remain assay-neutral; modality adapters can be added separately.

Expected visits are QC policies rather than stored observations. This keeps recorded facts in
`Study` separate from protocol expectations in `QCConfig` and allows the same study to be checked
against revised or alternative schedules. Named event anchors remain subject facts and allow one
relative policy, such as “28 days after first dose,” to resolve to different calendar dates.
