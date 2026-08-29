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
- [x] Exercise the audit workflow against public longitudinal canine data and document the boundary
  between the prior online reference run and bounded adapter fixtures used in CI.

## Phase 2: aging-fusion

- [x] Define calibrated modality estimates and missing-modality behavior.
- [x] Implement baseline uncertainty-aware fusion.
- [x] Quantify modality disagreement and leave-one-modality-out sensitivity.
- [x] Add evidence-level estimands, provenance, and multiple estimates per modality.
- [x] Implement covariance-aware generalized least-squares fusion and conditioning diagnostics.
- [x] Add evidence- and modality-level influence plus hierarchical modality balancing.
- [x] Define dense/sparse genome-scale sample and feature matrix contracts.
- [x] Add expression, methylation, AnnData-like, VCF/BCF, and genomic-interval adapters.
- [x] Add prespecified signature scoring, subject-clustered uncertainty, and feature-effect
  aggregation.
- [x] Add leakage-aware genomic target calibration with domain and overlap checks.
- [x] Add a revision-pinned, optional Hugging Face genome-embedding interface.
- [x] Demonstrate correlated genomic and clinical evidence in a synthetic canine workflow.
- [x] Exercise signature and covariance inference on a prespecified external public benchmark,
  with its small-sample and no-ground-truth limits stated explicitly.
- [x] Add a versioned, ambiguity-reporting cross-species ortholog adapter.
- [x] Add provider-neutral feature domains, immutable resource snapshots, secret-free query
  provenance, and separate canonical raw-resource/query/normalized-result hashes.
- [x] Add archived functional-annotation imports and versioned gene/protein-set contracts.
- [x] Add directionless overrepresentation analysis with an explicit measured background,
  term-size audit, and local multiple-testing correction.
- [x] Add archived interaction-network imports with provider mappings, confidence semantics,
  evidence channels, seed coverage, thresholding, and truncation audits.
- [x] Add allele- and assembly-specific variant annotation requests, consequence imports, and
  strict joins to dosage matrices without pretending to normalize equivalent indels.
- [x] Add public sequencing study/sample/experiment/run manifests without assuming runs or
  BioSamples are independent subjects; report subject counts only for verified mappings.
- [x] Add linear/log protein-abundance matrices and provenance-bound Ensembl/HCOP-like ortholog
  imports with explicit confidence semantics and translation-loss audits.
- [x] Validate the external-context boundary with archived GO- and STRING-shaped fixtures.
- [x] Add a separate, statistically explicit and reference-tested directional ranked-set analysis
  contract, bounded to association with a fixed pre-ranked universe.
- [x] Add provider-specific export helpers only where exact upstream release, licensing,
  pagination, and response-checksum semantics can be preserved.
- [x] Add chunked/Arrow import paths for full-scale annotation, network, and variant dumps, with
  resumable bounded-memory ingestion and deterministic manifests.

## Phase 3: aging-state

- [x] Define exact, typed observation channels and continuous-time linear-Gaussian state models.
- [x] Implement irregular-time Kalman filtering, Joseph covariance updates, and RTS smoothing.
- [x] Support partially observed channel vectors without inventing missing measurements.
- [x] Preserve coverage, exclusion, covariance, and source-study provenance in typed reports.
- [x] Add model-conditional future forecasts with uncertainty propagation.
- [x] Add disjoint held-out one-step forecast calibration and innovation diagnostics.
- [x] Add empirically calibrated innovation change-point detection.
- [x] Add explicit Phase 2 evidence-to-observation and Phase 3 state-to-endpoint bridges.
- [x] Document and exercise the complete state-estimation workflow on deterministic synthetic data.

## Phase 4: full SDK

- [x] Define one-endpoint-per-independent-subject contracts with exact estimands and artifact hashes.
- [x] Add explicit raw-study and latent-state endpoint construction with missing-subject audits.
- [x] Implement prespecified factorial main-effect and interaction estimands for randomized or
  explicitly observational multi-intervention studies.
- [x] Add classical and HC3 uncertainty, endpoint-precision policies, covariate adjustment,
  identifiability diagnostics, and interaction-family multiplicity correction.
- [x] Add balanced factorial allocation and an approximate two-by-two interaction-design helper.
- [x] Document and exercise a deterministic synthetic canine combination-therapy workflow.
- [x] Complete the fail-closed end-to-end Phase 1-to-4 workflow, result manifest, and public exports.
- [x] Stabilize the integrated public APIs and pass the full local and CI release-validation suite.

## Release step (maintainer approval required)

- [ ] Commit the reviewed diff, push it, publish the versioned documentation, tag the release, and
  verify the resulting PyPI and archival artifacts.
