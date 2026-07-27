# Public rapamycin data example

The `public_gse131754_rapamycin.py` example runs RejuvenationKit against the public
[GSE131754 RNA-seq dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131754).
The study contains liver expression profiles from several lifespan-extending interventions.
The example selects rapamycin and age/sex-matched control samples at 6 and 12 months.

## Run it

```bash
python examples/public_gse131754_rapamycin.py \
  --genes 250 \
  --output-dir analysis/gse131754
```

The script:

1. downloads and caches the 2.4 MB processed assigned-read matrix from NCBI;
2. validates its gene and sample identifiers, numeric values, completeness, and non-negativity;
3. parses intervention, age, sex, and replicate from the published sample names;
4. creates typed `Subject`, `Observation`, and `Study` objects;
5. runs Phase 1 QC over the most variable genes;
6. profiles feature completeness within each reported-age sampling group;
7. summarizes visit-level expression distributions and robust sample outliers;
8. explicitly identifies that different mice—not repeated measurements—represent each age;
9. computes descriptive RAP-minus-control log2 CPM differences within each age/sex stratum; and
10. optionally writes a sample manifest, profiling tables, QC report, and descriptive results.

Network access is required only for the first run. The downloaded file is cached under
`examples/data/cache/`, which is excluded from version control.

## Why the timeline is normalized

GSE131754 reports mouse age but not calendar collection timestamps. The adapter therefore stores a
clearly labeled normalized birth anchor and derives observation times from reported age. It does
not invent clinic dates.

## What this can establish

This example demonstrates that RejuvenationKit can ingest a real public intervention dataset and
produce an analysis-ready, typed, quality-checked representation. The descriptive contrasts can
identify candidate treatment-responsive genes for later analysis.

It cannot establish that rapamycin extends lifespan or rejuvenates an organism:

- the data are cross-sectional liver RNA-seq, not longitudinal health or survival records;
- each experimental group has only three biological replicates;
- the descriptive comparison does not model count dispersion or multiple testing;
- the processed matrix does not expose sequencing-batch identifiers, so the batch-confounding
  diagnostic cannot be evaluated from this file; and
- gene selection and exploratory contrasts are not an independently validated Phase 2 estimator.

A formal follow-up should use a count-aware model such as DESeq2 or edgeR, pre-register contrasts,
control false discovery, and validate signatures in an independent dataset. RejuvenationKit Phase
2 can later consume those calibrated effects rather than replace assay-specific statistics.
