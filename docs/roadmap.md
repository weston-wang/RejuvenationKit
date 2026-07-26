# Roadmap

## Phase 1: aging-qc

- [x] Define range, missingness, temporal, replicate, and batch checks.
- [x] Produce machine-readable findings and human-readable summaries.
- [x] Validate on synthetic longitudinal studies with injected faults.
- [x] Add expected-visit schedules and visit-level missingness.
- [ ] Add subject-relative schedules anchored to enrollment or intervention time.
- [ ] Add treatment/batch confounding diagnostics.
- [ ] Validate thresholds on a public longitudinal aging dataset.

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
