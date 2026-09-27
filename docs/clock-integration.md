# Using biolearn and pyaging clocks

RejuvenationKit does not reimplement aging clocks. Two maintained libraries already compute
dozens of published clocks:

- [`biolearn`](https://github.com/bio-learn/biolearn), from the Biomarkers of Aging Consortium;
- [`pyaging`](https://github.com/rsinghlab/pyaging).

`rejuvenationkit.clocks` takes their outputs and adds what they do not provide:

1. typed, provenance-carrying observations that flow into Phase 1 QC, randomized treatment
   effects, Phase 3 state estimation and [surrogate validation](surrogate-validation.md); and
2. a leakage-safe definition of age acceleration.

Neither library is a RejuvenationKit dependency. Install the one you use separately, for example
`pip install biolearn` (it also needs `torch`, `seaborn` and `lifelines` at import time).

## From biolearn

```python
from biolearn.data_library import GeoData
from biolearn.model_gallery import ModelGallery
from rejuvenationkit.clocks import clock_observations, clock_table_from_biolearn

geo = GeoData(metadata, methylation_betas)  # samples as columns
gallery = ModelGallery()
table = clock_table_from_biolearn(
    {name: gallery.get(name).predict(geo) for name in ("Horvathv1", "DunedinPACE")}
)
observations, missing = clock_observations(
    table,
    samples,  # index: sample ID; columns: subject_id, timestamp (tz-aware), optional batch_id
    units={"Horvathv1": "years", "DunedinPACE": "years per year"},
    source="biolearn 0.9.1",
)
```

`clock_table_from_biolearn` also accepts the samples-by-models frame returned by
`biolearn.mortality.run_predictions`. The test suite runs a real biolearn Horvath clock through
this path when biolearn and torch are installed.

## From pyaging

```python
import pyaging as pya
from rejuvenationkit.clocks import clock_table_from_pyaging

pya.pred.predict_age(adata, ["Horvath2013", "DunedinPACE"])
table = clock_table_from_pyaging(adata, ["Horvath2013", "DunedinPACE"])
```

pyaging stores lowercase column names in `adata.obs`, so names are matched case-insensitively.

## What the observations carry

Each observation records the following, so every downstream result traces back to one clock run
on one sample:

- the subject;
- the measurement time;
- the clock name as the feature;
- the declared unit;
- the plate or batch;
- the sample ID; and
- the clock software and version.

Missing predictions are returned explicitly, not silently dropped.

## Leakage-safe age acceleration

Age acceleration is usually the residual from regressing clock age on chronological age. Studies
often fit that regression on **all** samples, treated follow-up samples included. The treatment
effect then partly moves into the fitted line.

`age_acceleration(table, chronological_age, reference_samples=...)` fits the line only on
declared reference samples, such as controls or pre-intervention baselines. Every other sample is
scored against that line, and each reference sample is scored leave-one-out.

In simulation the true effect was −2 clock-years. The dogs were 8 ± 2 years old, clock noise
was 1.5 years, and there were 300 replicates.

| Design | All-sample regression | Reference-only (this module) |
|---|---:|---:|
| Randomized, both arms measured at baseline and follow-up, difference-in-differences | −2.00 | −2.00 |
| Single-arm pre-post, reference = baselines | −1.24 (38% understated) | −2.02 |
| Cross-sectional, treated 4 years older than controls | −1.00 (50% understated) | −2.07 |

In the balanced randomized design, differencing cancels the slope, so the choice does not matter.
The designs common in early human and companion-animal rejuvenation studies are single-arm
pre-post studies and age-imbalanced comparisons. In those, all-sample age acceleration
systematically **understates** the true effect, and the bias grows with follow-up length and age
imbalance. It can also manufacture spurious effects when groups differ in age and the clock's
true slope is not 1.

## Limitations

- The reference line is linear. Clocks with nonlinear age dependence need a nonlinear reference
  model fitted under the same rule.
- Reference samples must come from the same assay platform, tissue and preprocessing as the
  samples being scored. Clock predictions shift across array versions and normalization choices.
  Record the batch with `batch_id` and check it in Phase 1 QC.
- A clock's own training cohort is not a valid reference population for a new study.
