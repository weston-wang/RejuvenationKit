# Private DSP experiments

This directory is excluded through `.git/info/exclude`. Its contents remain local to this clone
and will not appear in normal Git status, commits, or pushes.

The first experiment is a second-order discrete Volterra model for nonlinear, history-dependent
epigenetic-marker responses. It is exploratory research code, not part of RejuvenationKit's
public API.

Run its validation and example directly:

```bash
.venv/bin/pytest -o addopts="" private_experiments/test_volterra_epigenetics.py
.venv/bin/python private_experiments/volterra_epigenetics_example.py
```

The model treats each animal trajectory independently when constructing lagged regressors, so
history never leaks across subjects. It estimates:

- baseline marker level;
- first-order exposure-memory response;
- second-order exposure interactions and saturation;
- fitted linear and nonlinear contributions; and
- full first- and symmetric second-order kernels.

The current implementation assumes equally spaced samples. Irregular sampling, hierarchical
animal effects, uncertainty intervals, sparsity, and external validation remain future work.

## Human rapamycin off-target work

`tensor_off_target.py` factors a subject × visit × marker response tensor relative to placebo.
Marker annotations turn otherwise opaque loadings into pathway energy, intended-target energy,
and off-target candidates. `human_rapamycin_off_target_example.py` then models each factor's
exposure history with the Volterra model.

The example is deliberately synthetic and shaped like the published PEARL trial. PEARL's
participant-level molecular and bloodwork matrix is not public. Real human expression datasets
such as GSE62375, GSE124020, and GSE154401 are suitable for validating gene/pathway direction,
but cannot be coupled to PEARL participants or presented as longitudinal healthy-aging results.

`human_rapamycin_geo.py` downloads GEO series matrices, maps platform probes to gene symbols,
constructs cell-context-specific rapamycin-control contrasts, scores seven aging pathways, and
calculates cross-context gene-effect concordance. `validate_human_rapamycin_pathways.py` runs the
real-data validation. GPL17586 is mapped through Bioconductor's
`hta20transcriptcluster.db` and NCBI's human gene-info table because GEO does not publish a compact
annotation for that platform.
