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
- calibration and analysis provenance;
- subject, sample, tissue, species, and assay metadata;
- a correlation group; and
- quality flags.

Multiple transcriptomic signatures are therefore visible as distinct estimates without pretending
they came from independent modalities.

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

The covariance artifact must name exactly the supplied evidence IDs. The implementation reorders
it by ID, verifies symmetry, positive diagonals, reported variances, finiteness, and positive
semidefiniteness, then reports:

- evidence and summed modality weights;
- standardized residuals and a covariance-weighted disagreement score;
- weight-concentration count \(1/\sum_i w_i^2\), which describes weight balance but is not an
  independent-information count;
- matrix condition number and applied diagonal ridge;
- negative-weight and quality warnings; and
- leave-one-evidence and leave-one-modality influence.

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
the supplied matrix contains cross-modality covariance, the result reports
`cross_modality_covariance_not_propagated_by_hierarchy`. Use one-stage GLS when that dependence is
central; use hierarchy when modality balance and interpretability are the main design goal.

Covariance should come from held-out subjects, repeated calibration cohorts, or subject-level
bootstrap replicates. It should not be estimated by resampling genes or embedding dimensions.
