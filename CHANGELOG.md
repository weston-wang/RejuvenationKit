# Changelog

All notable changes will be documented here. This project follows Semantic Versioning and the
Keep a Changelog format.

## [Unreleased]

## [0.4.0a1] - 2026-08-21

### Added

- Completed the Phase 2 evidence layer with typed estimands and calibration references,
  covariance-aware generalized least-squares and hierarchical fusion, expected-evidence policies,
  influence diagnostics, and genome-scale expression, methylation, protein, variant, and signature
  contracts.
- Added provider-neutral, release-pinned external biology artifacts for annotations, gene sets,
  overrepresentation and directional ranked-set analysis, interaction networks, variants,
  ortholog translation, sequencing manifests, provider exports, and bounded-memory NDJSON or
  optional Parquet archives. Descriptive database outputs remain explicitly outside efficacy
  fusion.
- Added Phase 3 prespecified continuous-time linear-Gaussian state estimation with exact
  irregular-time discretization, partial-channel Kalman updates, RTS smoothing, forecasts,
  held-out forecast calibration, innovation change-point detection, and hash-bound model and study
  provenance.
- Added Phase 4 subject-endpoint contracts, explicit raw-study and latent-state bridges, factorial
  main-effect and departure-from-additivity analysis, classical and HC3 uncertainty, endpoint
  weighting, multiplicity control, cell/rank diagnostics, and two-by-two design helpers.
- Added a fail-closed four-phase workflow with explicit boundary inputs, a serialized Phase 1 QC
  gate, phase dispositions, cross-validated input/configuration/result identities, manifest-last
  publishing, no-clobber behavior, and checksum-verified loading.
- Added deterministic Phase 3, Phase 4, external-biology, and complete four-phase examples plus
  architecture and stakeholder use-case documentation.

### Changed

- Advanced the Phase 1 audit-report schema to version 3: profiler-origin longitudinal exclusions
  now participate in the serialized exclusion ledger, and reports strictly bind QC disposition,
  analysis plans and results, exclusion counts, and artifact inventory. Version 2 reports must be
  regenerated under the stronger contract.
- Hardened Phase 1 serialization against non-finite outputs, ambiguous wildcard/exact visit
  requirements, duplicate or drifting longitudinal axes, nonchronological selected visits,
  inconsistent readiness-table arithmetic, and detached detection or randomized-inference
  provenance.
- Made covariance symmetry, reported-variance agreement, positive-semidefinite checks, and
  hierarchical cross-modality detection invariant to marginal numerical scale; completed fusion
  result mappings are now immutable.
- Bound every paginated provider-export receipt to its exact ordered raw-page bytes (archive format
  v2), made interaction-network row accounting exhaustive, rejected secret-bearing resource URIs,
  and rejected symlink redirection in chunked-store integrity paths.
- Enforced shared feature type/namespace compatibility for upstream feature effects and made
  Hugging Face embedding provenance immutable and input-aligned; overlapping chunk pooling is now
  explicitly reported as a nonlinear context heuristic rather than unbiased per-base averaging.
- Phase 1 longitudinal extraction now resolves exact modality/feature/unit channels, records source
  rows and effective timestamps, rejects non-finite contamination, and prevents one observation
  from satisfying more than one expected visit.
- Phase 1 change, sequential, and randomized-treatment reports now retain structured exclusion,
  calibration, inference, and reconstruction provenance; serialized fitted detectors are
  integrity-checked before reuse.
- Phase 1 randomized treatment-effect inference now requires an explicit randomized-assignment
  declaration and rejects observational exposure groups before unrestricted label permutation.
- Study artifact hashing is now order-invariant for subjects, interventions, and observations while
  continuing to bind logical study content and metadata. Persisted study-derived identities from
  earlier alpha versions must be regenerated.
- Factorial analyses now require an explicit randomized or observational assignment declaration;
  interaction coefficients are documented as departures from additivity on the declared scale,
  not automatic synergy or efficacy claims.
- Phase 3 workflow and endpoint bridges now retain the canonical state-report artifact identity;
  state and combination reports reject contradictory serialized partitions, factorial cells,
  assignments, coefficients, and inferential summaries. Combination reports additionally bind the
  complete study used for intervention assignments and covariates.
- Phase 3 innovation reports now state explicitly that their empirical false-alarm rate is
  per-innovation and does not control familywise error across a subject trajectory or cohort;
  serialized held-out partitions and threshold decisions are cross-validated on load.
- Timed Phase 2-to-3 evidence is now row-order invariant, retains complete estimand, species,
  tissue, assay, and calibration provenance as observation attributes, rejects mixed semantics
  within one state channel, and prevents reuse of one evidence record at multiple times. Phase 4
  rejects unknown endpoint exclusions and requires cell and diagnostic exclusion partitions to
  agree exactly, while binding the endpoint source artifact into the combination-report identity.
- Workflow publication and loading now reject symbolic-link roots, managed artifacts, and managed
  ancestor paths so a bundle cannot read or write outside its selected directory.

## [0.3.0a3] - 2026-08-15

### Added

- Provider-neutral, immutable feature domains, external-resource snapshots, exact feature queries,
  secret-free query provenance, and separate raw-response versus normalized-artifact checksums.
- Offline functional-annotation and versioned gene/protein-set imports plus validated,
  explicit-background overrepresentation analysis with audited local multiplicity correction.
- Offline interaction-network imports with canonical endpoints, evidence channels, confidence and
  truncation semantics, seed coverage, and a hard non-fusible evidence boundary.
- Allele- and assembly-exact variant annotation imports and dosage joins with strict integral
  coordinates, reference-span validation, and an explicit no-equivalence-normalization policy.
- Integrity-bound public sequencing manifests preserving study, sample, experiment, and run
  hierarchy, with verified-only independent-subject counts.
- Typed protein-abundance matrices, explicit HGNC/STRING identifier semantics, a complete Phase 2
  structure/use-case guide, and an offline GO/STRING-shaped example.

### Changed

- Hardened ortholog translation with exact source/target domains, source-query and resource
  provenance, optional declared confidence semantics, target-collision and translated-weight
  audits, tissue preservation, and a non-fusible boundary.
- Bound feature-effect batches to their contrast and parent analysis, and included the full
  signature fingerprint in downstream provenance.
- Replaced genomic matrix hashing with the tagged, length-delimited v2 schema. Dense and canonical
  CSR representations of the same logical matrix now hash identically; v2 digest values are not
  compatible with earlier alpha hashes, so persisted calibration/provenance artifacts must be
  rebuilt when migrating.

## [0.3.0a2] - 2026-08-13

### Added

- Evidence-level estimands and provenance, covariance-aware generalized least-squares fusion,
  numerical conditioning diagnostics, and evidence/modality influence analysis.
- Hierarchical fusion that combines correlated clocks or signatures within modalities before
  balancing evidence across modalities.
- Dense and sparse genome-scale data contracts with expression, methylation, AnnData-like,
  VCF/BCF, and genomic-interval adapters.
- Prespecified genomic signature scoring, subject-clustered contrast uncertainty, correlation-aware
  feature-effect aggregation, and leakage-aware genomic target calibration.
- An optional, immutable-revision Hugging Face sequence-embedding interface with explicit model
  license, layer, pooling, strand, chunking, and no-silent-truncation policies.
- A synthetic canine workflow comparing naive, covariance-aware, and hierarchical genomic fusion.
- A public GSE131754 genome-scale benchmark with joint subject-bootstrap covariance, explicit
  covariance shrinkage, pathway-specific estimands, and age/sex/dose separation.
- Versioned cross-species signature translation with explicit one-to-many policies and retained
  feature-weight diagnostics.

## [0.3.0a1] - 2026-08-12

### Added

- Phase 2 fixed- and random-effects multimodal fusion with explicit external calibration,
  missing-modality policies, heterogeneity diagnostics, confidence intervals, and
  leave-one-modality-out influence analysis.
- A documented synthetic canine example covering coherent, conflicting, and missing-assay
  scenarios.

## [0.2.0a2] - 2026-08-02

### Added

- Citation metadata and a documented scientific-software release process.
- Trusted Publishing automation for token-free PyPI releases from GitHub.
- PyPI, changelog, and release-discovery metadata.

## [0.2.0a1] - 2026-07-30

### Added

- A one-command Phase 1 study audit with canonical input and artifact hashes, standardized
  CSV/JSON/Markdown outputs, overview and DSP visualizations, optional held-out detection and
  randomized inference, and public canine validation.
- An explicit RejuvenationKit client identity for reliable public Harvard Dataverse downloads.
- Out-of-fold control calibration, randomized multivariate treatment-effect tests, bootstrap
  channel intervals, and a synthetic canine rapamycin example completing the Phase 1 roadmap.
- Configurable Phase 1 QC for values, units, required-feature missingness, temporal ordering,
  technical-replicate disagreement, and batch mean shifts.
- Absolute expected-visit schedules with inclusive tolerance windows, subject/cohort selectors,
  visit-level completeness, and out-of-window measurement findings.
- Subject-relative schedules resolved from timezone-aware enrollment or dosing anchors.
- Cohort and intervention versus batch association screening using Cramér's V.
- A tested GSE131754 adapter and end-to-end public rapamycin RNA-seq QC example.
- A tested Dog Aging Project adapter and real canine longitudinal chemistry QC example.
- Typed visit-coverage, complete-case retention, and paired-analysis readiness profiles.
- Visit-level distribution summaries, Tukey-IQR outliers, and attrition-bias diagnostics.
- Shrinkage-covariance multivariate change detection with whitening and empirical false-alarm
  calibration.
- A public canine multimodal example combining longitudinal clinical chemistry and metabolomics.
- Optional DSP diagnostic figures for covariance, detection thresholds, whitened innovations, and
  subject-level component energy.
- Reference-conditioned sequential multimodal detection with irregular-time normalization,
  subject-level false-alarm calibration, onset, persistence, and modality evidence.
- A three-wave public canine demonstration and sequential trajectory, classification, and
  modality-evidence figures.
- A synthetic TRIAD-like rapamycin example demonstrating individual onset, persistence,
  transient response, and dominant evidence modality without implying unreleased trial results.
- Structured report metrics, severity counts, summaries, documentation, and synthetic tests.
- Generalized experimental-confounding diagnostics for site, clinic, visit/timepoint, plate,
  assay run, operator, vector lot, manufacturing batch, sequencing lane, and custom factors.

## [0.1.0] - 2026-07-25

### Added

- Initial pre-alpha architecture, schemas, phase interfaces, documentation, examples, and CI.
