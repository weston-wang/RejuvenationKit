# Architecture

The package separates validated data contracts from estimation algorithms:

1. `schemas` validates study metadata and long-form observations.
2. `qc` defines immutable feature and expected-visit policies, then produces structured findings
   without mutating input data.
3. `profiling` reuses those visit policies to quantify coverage, retention, paired-analysis
   readiness, differential attrition, and robust distribution anomalies.
4. `longitudinal` resolves exact modality/feature/unit channels against protocol visits, records
   selected source rows and effective timestamps, and emits structured exclusions.
5. `detection` whitens correlated longitudinal changes and applies empirically calibrated
   multivariate detection thresholds.
6. `sequential` learns reference aging dynamics and monitors onset and persistence across repeated
   multimodal visits.
7. `treatment_effect` uses out-of-fold control calibration and randomized-label inference to
   estimate longitudinal group effects.
8. `audit` orchestrates Phase 1 checks and inference into a reproducible human- and
   machine-readable report bundle.
9. `fusion` provides a lightweight fixed/random-effects baseline for one scalar per modality.
10. `evidence` combines complete, calibrated estimands through covariance-aware generalized least
    squares and hierarchical modality balancing, with expected-evidence and influence diagnostics.
11. `genomics` keeps high-dimensional dense/sparse measurements aligned to sample, feature, scale,
    species, tissue, assembly, and provenance metadata; it constructs longitudinal signatures,
    directional ranked-set results, or held-out target predictions before fusion.
12. `genomics.resources` freezes external database releases or acquisition snapshots, exact
    feature domains, query inputs, parameters, response checksums, completeness, and warnings.
13. `genomics.annotations`, `genomics.enrichment`, `genomics.networks`, `genomics.variants`, and
    `genomics.sequencing` import archived biological context without converting database scores,
    p-values, consequences, or run counts into efficacy evidence.
14. `genomics.exports` and `genomics.chunked` preserve provider-export and bounded-memory import
    provenance for reproducible offline analysis.
15. `bridges` performs explicit, hash-bound conversions from timestamped calibrated evidence to
    state observations and from raw or latent-state trajectories to subject endpoints.
16. `state` performs irregular-time linear-Gaussian filtering, RTS smoothing, forecasting,
    held-out calibration, and innovation change detection.
17. `endpoints` defines one declared outcome per independent subject for downstream treatment
    comparisons.
18. `combinations` estimates factorial main effects and departures from additivity, audits design
    cells and exclusions, and provides bounded two-by-two design helpers.
19. `workflow` executes any configured Phase 1-to-4 path behind the serialized Phase 1 QC gate,
    requires explicit boundary inputs, and publishes a checksummed manifest-last bundle.

Estimators expose typed, task-appropriate methods such as `fit`, `score`, `fuse`, `estimate_report`,
or `analyze`, and return typed results. Implementations remain assay-neutral; modality adapters can
be added separately.

Genome-scale matrices are deliberately separate from scalar `Observation` rows. They remain in
matrix form through assay validation, signature scoring, sequence embedding, and held-out target
calibration. Only calibrated scalar targets with subject-level uncertainty cross into the evidence
fusion layer.

External biological resources form a separate descriptive branch. They may define a prespecified
signature, moderator, assay plan, or independent validation hypothesis. Their result types are
explicitly `not_fusible`; only subject-level contrasts or held-out calibrated predictions with a
declared estimand and uncertainty can enter evidence fusion. See
[external biology resources](external-biology-resources.md) and the
[Phase 2 architecture](phase-2-architecture-and-use-cases.md).

Expected visits are QC policies rather than stored observations. This keeps recorded facts in
`Study` separate from protocol expectations in `QCConfig` and allows the same study to be checked
against revised or alternative schedules. Named event anchors remain subject facts and allow one
relative policy, such as “28 days after first dose,” to resolve to different calendar dates.
