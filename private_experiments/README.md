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
