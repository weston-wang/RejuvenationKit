# Changelog

All notable changes will be documented here. This project follows Semantic Versioning and the
Keep a Changelog format.

## [Unreleased]

### Added

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

## [0.1.0] - 2026-07-25

### Added

- Initial pre-alpha architecture, schemas, phase interfaces, documentation, examples, and CI.
