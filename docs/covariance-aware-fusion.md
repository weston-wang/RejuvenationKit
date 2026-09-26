# Covariance-aware and hierarchical fusion

The original `PrecisionWeightedFusion` API remains the simple one-estimate-per-modality baseline.
The evidence-level API is for several clocks, pathways, tissues, or assays that share samples,
features, or training data.

## Evidence contracts

`Estimand` defines a name, unit, direction, population, time contrast, and transform. Every field
must match before estimates are combined. `fuse_by_estimand(...)` separates unrelated targets
instead of averaging biological-age years, inflammation scores, and pathway activity.

`EvidenceEstimate` adds:

- a unique evidence ID independent of modality;
- a typed, hash-bound `CalibrationReference` plus analysis provenance;
- subject, sample, tissue, species, and assay metadata;
- a correlation group; and
- quality flags.

Multiple transcriptomic signatures are therefore visible as distinct estimates without pretending
they came from independent modalities.

Fusion is fail-closed by default: each input needs a held-out-validated or externally validated
calibration reference whose exact `Estimand` matches the evidence. Internal cross-validation and
legacy free-string calibration IDs can still be retained as exploratory evidence, but require the
caller to disable the eligibility gate explicitly. `expected_evidence_ids` can prespecify a panel;
missing members either raise or produce an explicit warning under the configured policy.

Mixed species and distinct subject-level estimates are rejected unless the configuration explicitly
allows them. Mixed tissues are permitted for organ-level fusion but produce a visible warning.

## Generalized least squares

For estimate vector \(y\) and externally estimated covariance \(\Sigma\),
`GeneralizedLeastSquaresFusion` calculates

\[
\hat\theta =
\frac{\mathbf{1}^T\Sigma^{-1}y}{\mathbf{1}^T\Sigma^{-1}\mathbf{1}},
\qquad
\operatorname{SE}(\hat\theta)=
\left(\mathbf{1}^T\Sigma^{-1}\mathbf{1}\right)^{-1/2}.
\]

That standard error trusts \(\Sigma\) completely. If the estimates disagree more than
\(\Sigma\) allows, for example because of calibration bias or unmodeled between-assay
heterogeneity, the interval would stay narrow anyway. The default
`dispersion_policy=EvidenceDispersionPolicy.MULTIPLICATIVE` therefore uses the disagreement
statistic \(Q = r^T\Sigma^{-1}r\), which is \(\chi^2_{k-1}\) when \(\Sigma\) is complete.
With \(\varphi = Q/(k-1)\), the result reports:

- `standard_error` \(= \operatorname{SE}\sqrt{\max(1,\varphi)}\);
- an interval half-width of \(\max\left(z,\; t_{k-1}\sqrt{\varphi}\right)\operatorname{SE}\);
- `dispersion_factor` \(= \max(1,\varphi)\); and
- an `evidence_overdispersed` warning when \(\varphi > 1\).

The \(t_{k-1}\sqrt{\varphi}\) term is exact when \(\Sigma\) is correct up to scale. Taking
the maximum with the fixed-effect \(z\) interval stops it from collapsing when a few estimates
agree by chance. In simulations with correlated evidence, the fixed-effect interval covered 83% at
nominal 95% once between-evidence heterogeneity equalled about one standard error. The adjusted
interval covered 97–99%. It is conservative for two or three estimates, because two or three
numbers carry little information about heterogeneity.
`EvidenceDispersionPolicy.FIXED` restores the fully trusted-covariance interval.

The covariance artifact must name exactly the supplied evidence IDs. When multiple inputs share a
`correlation_group`, omitting covariance raises by default; assuming independence requires an
explicit warning policy. The implementation reorders
it by ID and verifies symmetry, positive diagonals, reported variances, finiteness, and positive
semidefiniteness. Symmetry, diagonal agreement, and positive semidefiniteness are
checked in marginal-standard-error (correlation) coordinates so changing the numerical scale of
the common estimand cannot hide an invalid covariance. The result reports:

- evidence and summed modality weights;
- standardized residuals and a covariance-weighted disagreement score;
- weight-concentration count \(1/\sum_i w_i^2\), which describes weight balance but is not an
  independent-information count;
- matrix condition number and applied diagonal ridge;
- negative-weight and quality warnings; and
- leave-one-evidence and leave-one-modality influence.

Completed baseline, evidence-level, and hierarchical result mappings are exposed as immutable
views, so downstream presentation code cannot silently rewrite validated weights, residuals,
calibration identifiers, or within-modality results.

Near-singular covariance is automatically regularized to a configurable maximum condition number
or rejected in strict mode. Negative GLS weights can be mathematically valid under correlation,
but they are always surfaced for scientific review.

`EvidenceWeightConstraint.NONNEGATIVE` provides a constrained minimum-variance solution whose
weights lie in `[0, 1]` and sum to one. This prevents extrapolative cancellation when covariance is
estimated from a small cohort. The constraint is reported in result warnings and may increase
uncertainty; it is a robustness policy, not a way to select favorable assays.

## Hierarchical fusion

`HierarchicalEvidenceFusion` first fuses correlated estimates within each modality, then combines
one estimate per modality through the compatible fixed- or random-effects baseline. This prevents
five correlated RNA signatures from automatically receiving five times the representation of one
clinical endpoint.

The current hierarchical model does not propagate covariance between modality-level summaries. If
the supplied matrix contains cross-modality covariance, hierarchical fusion raises by default.
An explicit `WARN_IGNORE` policy retains the earlier
`cross_modality_covariance_not_propagated_by_hierarchy` result warning. Use one-stage GLS when that
dependence is central; use hierarchy when modality balance and interpretability are the main design
goal.

Covariance should come from held-out subjects, repeated calibration cohorts, or subject-level
bootstrap replicates. It should not be estimated by resampling genes or embedding dimensions.
