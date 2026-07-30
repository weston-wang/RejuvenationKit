# Roadmap

## Phase 1: aging-qc

- [x] Define range, missingness, temporal, replicate, and batch checks.
- [x] Produce machine-readable findings and human-readable summaries.
- [x] Validate on synthetic longitudinal studies with injected faults.
- [x] Add expected-visit schedules and visit-level missingness.
- [x] Add subject-relative schedules anchored to enrollment or intervention time.
- [x] Add treatment/batch confounding diagnostics.
- [x] Add visit coverage, complete-case retention, and paired-analysis readiness profiles.
- [x] Add visit/timepoint, site, clinic, plate, run, operator, lot, manufacturing-batch, and
  sequencing-lane confounding diagnostics.
- [x] Add distribution, robust-outlier, and attrition-bias diagnostics.
- [x] Add covariance-aware multivariate change detection with empirical false-alarm calibration.
- [x] Validate one detector across aligned longitudinal clinical and metabolomic channels.
- [x] Add covariance, threshold, whitened-innovation, and component-energy visual diagnostics.
- [x] Exercise calibrated thresholds on held-out public longitudinal aging data.
- [x] Add sequential detection across three or more visits with persistence requirements.
- [x] Add cross-validated calibration and randomized group-level treatment-effect inference.
- [x] Package Phase 1 into a reproducible one-command study-audit workflow.
- [x] Validate the audit workflow against public longitudinal canine data.

## Phase 2: aging-fusion

- Define calibrated modality estimates and missing-modality behavior.
- Implement baseline uncertainty-aware fusion.
- Quantify modality disagreement and leave-one-modality-out sensitivity.

## Phase 3: aging-state

- Implement linear-Gaussian filtering and smoothing baseline.
- Add irregular visit spacing and missing observations.
- Validate coverage, change detection, and forecast calibration.

## Phase 4: full SDK

- Add combination-therapy estimands and efficient-design helpers.
- Add workflow orchestration, adapters, reports, and benchmark datasets.
- Stabilize public APIs and publish versioned documentation.
