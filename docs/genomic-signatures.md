# Genomic signatures and target calibration

Genome-level measurements become Phase 2 evidence through a declared biological estimand and
held-out uncertainty—not by counting significant genes or averaging embedding dimensions.

## Weighted signatures

`GeneSignature` records a version, full-definition fingerprint, species, namespace, tissue,
allowed numerical scales, feature weights, biological target, and source resource.
`score_weighted_signature(...)` accepts only a declared scale, calculates a signed weighted mean
per sample, and reports both gene coverage and sample-level observed-weight coverage.

Missing features have three explicit policies:

- `ERROR`: require every prespecified feature;
- `DROP_AND_RENORMALIZE`: normalize to observed absolute weight; or
- `IMPUTE_ZERO`: retain the original signature denominator and treat absent features as zero.

All policies still enforce `minimum_feature_coverage`. Missingness produces warnings and is carried
into downstream evidence. `minimum_sample_coverage` prevents a single observed gene from being
renormalized into a complete score.

For a cross-sectional contrast, `estimate_signature_contrast(...)` permits technical replicates
only at one timestamp per subject, aggregates them, and bootstraps independent subjects within
treated and control groups. Before resampling, within-group deviations are inflated by
\(\sqrt{n/(n-1)}\), so the bootstrap variance matches the unbiased \(s^2/n\). An uncorrected
bootstrap understates the standard error by 18% at three subjects per group. Intervals are
estimate ± Student t × SE with Welch–Satterthwaite degrees of freedom. Percentile intervals covered
only about 80% at a nominal 95% with three subjects per arm. If a subject has multiple biological timepoints, the analysis fails
closed rather than silently averaging baseline and follow-up.

Longitudinal change requires `SignatureContrastMode.PAIRED_CHANGE`, timezone-aware non-overlapping
baseline and follow-up windows, and an explicit `time_contrast`. Each subject contributes its
follow-up-minus-baseline score. Incomplete pairs either raise or are excluded under an explicit
policy, with subject IDs and warnings retained in the result. Matrix artifact identity—not merely
the numerical values—enters contrast and covariance provenance. This avoids treating cells, genes,
repeated aliquots, timepoints, or embedding dimensions as independent animals.

`estimate_signature_contrasts(...)` resamples the same subject indices across several signatures,
returns their joint covariance, and converts estimates plus covariance into the evidence API in one
operation. `covariance_shrinkage` blends off-diagonal bootstrap covariance toward zero while
preserving marginal standard errors. The shrinkage value is recorded in covariance provenance; it
should be prespecified and sensitivity-tested rather than chosen for a desired conclusion.

Joint covariance does not imply that different pathways are interchangeable estimators. Distinct
targets can remain a multivariate response vector. GLS fusion is reserved for estimates that share
a genuinely common estimand after calibration.

## Feature-effect aggregation

`read_feature_effects(...)` ingests DESeq2/edgeR/limma-style tables through explicit column
mappings. `aggregate_feature_effects(...)` applies signature weights to upstream feature estimates.
For decision-grade imports, wrap the rows in `FeatureEffectBatch`: it binds the exact contrast,
parent effect provenance, independent-subject counts, tested feature universe, design formula,
normalization, inference method, multiplicity policy, and time/population estimand. The batch
rejects effects from a different contrast or upstream analysis. Downstream provenance includes the
batch artifact hash and the complete signature fingerprint, so changing weights without changing a
display ID cannot silently reuse an old result identity.
When a labeled covariance matrix is supplied, uncertainty is

\[
\operatorname{Var}(w^T\hat\beta)=w^T\Sigma w.
\]

If covariance is absent, diagonal propagation is used and
`feature_covariance_not_supplied` is reported. P-values do not become weights automatically, and
feature selection based on the same evaluation contrast is a leakage risk.

## Leakage-safe genomic target calibration

`GenomicTargetCalibrator` uses a standardized ridge model and out-of-fold residuals to map a fixed
feature set onto a declared scalar target. Repeated samples from one subject remain in the same
fold. Repeated-sample residuals are aggregated so each independent subject contributes equal
weight to cross-validated error calibration. The reported standard error is not variance across
features.

At evaluation time the calibrator requires:

- no reused training sample identifiers;
- no reused training subjects unless the policy is explicitly relaxed;
- the same species, tissue, scale, feature type, namespace, assembly, full feature definitions,
  preprocessing steps, software versions, reference resources, and exact feature set; and
- finite values with no hidden imputation.

The calibration fingerprint binds the complete training matrix artifact, target values, subject
grouping, and configuration. Prediction batches retain both training and evaluation artifact hashes
and structured domains. The result includes an empirical error interval, training-calibrated
centered domain-distance threshold, immutable domain metadata, and conversion to an
`EvidenceEstimate` with an `INTERNAL_CROSS_VALIDATED` typed calibration reference. The `genomics`
extra supplies scikit-learn:

```bash
python -m pip install "rejuvenationkit[genomics]"
```

Cross-validation is an internal calibration estimate, not proof of external validity. A signature
trained and tested inside one cohort needs independent study validation before being used as a
general biological-age measurement. Evidence-level fusion therefore rejects internal-only
calibration by default; an exploratory caller must opt out explicitly, or an independent validation
workflow must issue a held-out/external calibration artifact.
