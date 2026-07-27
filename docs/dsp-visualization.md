# DSP visualization

Install the optional plotting dependency:

```bash
python -m pip install -e ".[visualization]"
```

The visualization module turns a `ChangeDetectionReport` into four engineering diagnostics:

| Function | Diagnostic purpose |
|---|---|
| `plot_covariance_structure` | Shows which reference-change channels co-vary before whitening |
| `plot_detection_scores` | Shows every held-out score against the calibrated false-alarm threshold |
| `plot_whitened_innovations` | Shows two normalized innovation components and projected threshold radius |
| `plot_subject_decomposition` | Contrasts original channel innovations with whitened component energy |

```python
from rejuvenationkit.visualization import save_detection_figures

paths = save_detection_figures(
    report,
    output_dir,
    subject_id="dog-001",
    prefix="rapamycin-month-12",
)
```

The standard figure set is deterministic for a fixed report. When no subject is selected, the
decomposition uses the highest-scoring subject.

## Interpretation

- Covariance and correlation plots describe the fitted reference changes, not biological
  causality.
- In more than two dimensions, the circle in the whitened scatter is a projected radius. A point
  can fall inside that two-component circle and still be detected because of energy in other
  whitened components.
- Squared whitened amplitudes sum to the Mahalanobis score, but Cholesky-whitened components depend
  on feature ordering. They should be called components, not independent biological pathways.
- A threshold crossing is a detection under the reference model, not a significance test or
  treatment response.
