# Reanalysis: shared controls inflate cross-intervention concordance

A common argument in geroscience is that different lifespan-extending interventions converge on
a shared molecular response. The usual evidence is a positive correlation between the
fold-change vectors of two interventions. This page reanalyzes a public mouse liver RNA-seq
dataset and shows that the naive version of that analysis changes conclusions for a purely
statistical reason.

This is a methods demonstration on one public dataset. It does **not** reproduce the source
publication's multi-dataset signature pipeline and makes no claim about that publication's
conclusions.

## Data

[GSE131754](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131754) contains 78 liver
transcriptomes, with three mice per group, from
[Tyshkovskiy et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6907080/):

| Interventions | Control | Strata |
|---|---|---|
| Acarbose (ACA), caloric restriction (CR), rapamycin (RAP) | shared `CON` | 6 and 12 months × female/male |
| 17-α-estradiol (EST), Protandim (PROT) | shared `CON` | 6 months × female/male |
| Growth-hormone-receptor knockout (GHRKO) | own `GHRCON` | 5 months male |
| Snell dwarf (SNELL) | own `SNELLCON` | 5 months male |
| Methionine restriction (MR) | own `MRCON` | 14 months male |

Genes with CPM > 1 in at least half of samples are kept (11,619 genes, chosen without group
labels). Each intervention's effect vector is its mean treated-minus-control log2 CPM difference,
averaged over its strata.

## The artifact

For two interventions A and B measured against the same control mice C,

\[
\widehat{\Delta}_A = \bar A - \bar C, \qquad \widehat{\Delta}_B = \bar B - \bar C,
\qquad \operatorname{Cov}(\widehat{\Delta}_A, \widehat{\Delta}_B) =
\operatorname{Cov}(\Delta_A, \Delta_B) + \operatorname{Var}(\bar C).
\]

The \(\operatorname{Var}(\bar C)\) term is control-sampling noise shared by both vectors. With no
real effects and equal group sizes it produces a correlation of about 0.5 across genes. With
three mice per group, most of each fold-change vector's variance is noise, so the artifact is
large.

## Estimator

`rejuvenationkit.genomics.concordance.intervention_concordance` reports three things for each
pair.

1. **Naive correlation:** the Pearson correlation of the two fold-change vectors.
2. **Corrected correlation:** the covariance minus the per-gene shared-control noise
   \(s_C^2/n_C\), divided by the noise-corrected signal standard deviations. This estimates the
   correlation of the *true* effects.
3. **Disjoint-control cross-check:** the same quantity when A uses one control mouse and B uses
   the others, so the two contrasts never share an animal. It is averaged over splits and needs
   no noise model.

The corrected covariance is tested with a within-stratum label permutation, and p-values are
Holm-adjusted over all 28 pairs. In simulation the corrected estimate recovered the true effect
correlation (0.01, 0.61 and −0.39 for truths of 0, 0.6 and −0.4). The naive correlation for the
same data was 0.24, 0.56 and 0.04. The permutation test held 5% size under the complete null.

## Results

Run with 999 permutations per pair:

```bash
python examples/gse131754_shared_control_reanalysis.py --permutations 999
```

**Pairs with a shared control** (10 pairs):

| Pair | Naive r | Corrected r | Disjoint-control r | Holm p |
|---|---:|---:|---:|---:|
| ACA–CR | +0.409 | +0.422 | +0.428 | 0.23 |
| ACA–RAP | +0.329 | +0.073 | +0.104 | 1.00 |
| EST–PROT | +0.266 | −0.030 | −0.039 | 1.00 |
| PROT–RAP | +0.251 | +0.100 | +0.088 | 1.00 |
| EST–RAP | +0.238 | +0.015 | +0.002 | 1.00 |
| ACA–PROT | +0.207 | +0.074 | +0.069 | 1.00 |
| ACA–EST | +0.180 | +0.006 | +0.000 | 1.00 |
| CR–PROT | +0.175 | +0.116 | +0.114 | 1.00 |
| CR–EST | +0.161 | +0.092 | +0.089 | 1.00 |
| CR–RAP | +0.135 | −0.144 | −0.130 | 1.00 |

**Pairs with independent controls that survive Holm adjustment:**

| Pair | Naive r | Corrected r | Holm p |
|---|---:|---:|---:|
| ACA–MR | +0.335 | +0.545 | 0.028 |
| CR–GHRKO | +0.354 | +0.416 | 0.028 |
| CR–SNELL | +0.385 | +0.444 | 0.028 |
| CR–MR | +0.219 | +0.275 | 0.028 |
| ACA–SNELL | +0.226 | +0.338 | 0.048 |

GHRKO–SNELL has the largest corrected concordance (+0.871, raw p = 0.008, Holm p = 0.18).

## What changes

| Question | Naive fold-change correlation | Shared-control-corrected analysis |
|---|---|---|
| Do the five shared-control interventions share a hepatic response? | Yes: all 10 pairs positive (r 0.14–0.41) | Mostly no: 6 of 10 pairs are within ±0.1 of zero, and none survives family-wise adjustment |
| Does rapamycin resemble acarbose, CR, EST or Protandim? | Moderately (r 0.14–0.33) | No detectable concordance (corrected r −0.14 to +0.10) |
| Which concordances are best supported? | Similar-looking values everywhere | CR and acarbose with the GH-axis mutants and methionine restriction, which were measured against independent controls |

The two correction routes agree to within 0.035 on every shared-control pair. One subtracts an
estimated noise term. The other never lets the two contrasts share an animal. That agreement is
the main evidence that the naive correlations were inflated by shared-control noise rather than
by shared biology.

For pairs with independent controls the corrected r is *larger* than the naive r. There is no
shared noise to remove, and correcting for each vector's own noise disattenuates the correlation.

## Limitations

- Three animals per group leave each effect vector mostly noise. The rapamycin vector averaged
  over four strata is about 14% signal, partly because the 6-month (42 ppm, 2 months) and 12-month
  (14 ppm, 8 months) cohorts differ in dose and duration and respond in opposite directions on some
  pathways. A null rapamycin concordance means "not detectable here", not "absent".
- Pairs with independent controls can still share batch or processing effects. The correction
  removes only the shared-*animal* noise that the design makes explicit. For example, GHRKO and
  Snell dwarf controls may have been processed together.
- The permutation test is exact under the complete null and conservative when real but
  uncorrelated effects exist, so Holm-adjusted p-values here are cautious.
- Correlation across genes treats genes as exchangeable. It is a descriptive similarity measure,
  not a test that specific pathways are shared.
- Nothing here concerns lifespan. Transcriptomic concordance is a mechanism hypothesis, not
  evidence that interventions extend life through the same route.
