# Public GSE131754 joint-response benchmark

The genome-level Phase 2 workflow was exercised on the public mouse-liver RNA-seq count matrix
[GSE131754](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131754). The source study contains
age-, sex-, and strain-matched controls for several lifespan-extending interventions. The
rapamycin subset used here has 24 samples: three rapamycin and three control mice in each of four
age/sex strata.

Run the complete download and analysis:

```bash
python examples/public_gse131754_genomic_fusion.py \
  --bootstrap-iterations 2000
```

The adapter validates and fingerprints a 24-sample by 43,629-gene raw-count matrix, then computes
library-size log2 CPM only for 23 prespecified panel genes. It scores four small, inspectable
mechanism panels:

Pinned source-file SHA-256:
`dbd9c37015a17729fc800dfda537b58f64e694af60e4aaeaece748d9d97c7e90`.
The complete aligned matrix content hash printed by the reference run was
`56c21c5630ec176fa85d584a60e106987ca030baeed77fc9e1106b4b0ba87c26`.

- mTORC1 and hepatic lipogenesis;
- autophagy and lysosomal function;
- NRF2 cytoprotection; and
- inflammatory/interferon response as a putative safety signal.

The panels use canonical genes from biological themes described with the dataset and an embedded,
auditable GRCm38 Ensembl ID-to-symbol map. Their signs are oriented so lower values are the
hypothesized favorable direction. They are software-validation fixtures—not the publication's full
models, externally calibrated biological-age clocks, or a claim that each transcript is causal.

## Observed engineering results

The example keeps each age, sex, dose, and duration stratum separate. It uses one joint
subject-cluster bootstrap and 35% diagonal covariance shrinkage to estimate the pathway effect
vector and covariance. The panels are not externally calibrated as interchangeable measurements
of one scalar, so the example deliberately does not collapse them into an efficacy score.

| Age/sex | Design | mTORC1/lipogenesis | Autophagy/lysosome | NRF2 | Inflammatory response | Max. \|correlation\| |
|---|---|---:|---:|---:|---:|---:|
| 6 months, female | 42 ppm, 2 months | +0.589 ± 0.328 | −0.057 ± 0.038 | −0.112 ± 0.087 | +0.309 ± 0.068 | 0.62 |
| 6 months, male | 42 ppm, 2 months | +0.645 ± 0.406 | −0.021 ± 0.023 | −0.097 ± 0.150 | +0.121 ± 0.172 | 0.51 |
| 12 months, female | 14 ppm, 8 months | −0.547 ± 0.438 | +0.087 ± 0.074 | +0.262 ± 0.151 | −0.244 ± 0.142 | 0.52 |
| 12 months, male | 14 ppm, 8 months | −0.569 ± 0.326 | +0.109 ± 0.050 | −0.010 ± 0.167 | −0.124 ± 0.105 | 0.55 |

Values are treated-minus-control mean signed log2 CPM ± bootstrap SE, not years of biological age.
The bootstrap resamples variance-corrected deviations, so each SE matches the unbiased
\(\sqrt{s_t^2/3 + s_c^2/3}\). An uncorrected bootstrap of three animals understated every SE by a
factor of \(\sqrt{2/3}\); earlier versions of this table reported those smaller values. With four
degrees of freedom or fewer, the 95% Welch t interval is roughly ±2.8 SE or wider, not ±1.96 SE.
Every estimate retains its age, sex, dose, duration, and pathway-specific estimand. The covariance
shows why the panels cannot be treated as four independent confirmations.

The result is deliberately more detailed than a list of differentially expressed genes: it keeps
mechanistic directions visible, quantifies dependence, propagates small-sample uncertainty, and
prevents an unjustified cross-pathway average from hiding disagreement. It does not demonstrate a
coherent favorable effect in every pathway or stratum.

## Why the robust policy matters

With only three mice per arm, raw off-diagonal bootstrap covariance is unstable. The example uses
a recorded 35% shrinkage toward diagonal covariance and reports the resulting condition number.
Shrinkage stabilizes the joint uncertainty artifact; it does not create biological
commensurability or turn the response vector into one outcome.

## Interpretation limits

- Three animals per arm per stratum are inadequate for stable covariance estimation or strong
  biological conclusions.
- Six- and 12-month cohorts received different doses and durations, so the example does not pool
  them.
- The public processed file does not expose every experimental batch factor needed for a full
  confounding audit.
- Hepatic transcriptional response is not equivalent to healthspan, survival, or rejuvenation.
- The example validates data handling and inference behavior on external genome-scale data; it
  does not independently validate the mechanism panels.

The source paper describes the broader intervention study and hepatic longevity signatures:
[Tyshkovskiy et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6907080/).
