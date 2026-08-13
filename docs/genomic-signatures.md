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

`estimate_signature_contrast(...)` aggregates repeated samples within subjects and bootstraps
subjects within treated and control groups. This avoids treating cells, genes, repeated aliquots,
or embedding dimensions as independent animals.

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
- the same species, tissue, scale, feature namespace, and exact feature set; and
- finite values with no hidden imputation.

The result includes an empirical error interval, training-calibrated centered domain-distance
threshold, immutable species/tissue/scale/namespace metadata, calibration fingerprint, and
conversion to an `EvidenceEstimate`. The `genomics` extra supplies scikit-learn:

```bash
python -m pip install "rejuvenationkit[genomics]"
```

Cross-validation is an internal calibration estimate, not proof of external validity. A signature
trained and tested inside one cohort needs independent study validation before being used as a
general biological-age measurement.
