# Architecture

The package separates validated data contracts from estimation algorithms:

1. `schemas` validates study metadata and long-form observations.
2. `qc` produces structured findings without mutating input data.
3. `fusion` converts modality-specific evidence into an uncertain joint estimate.
4. `state` tracks latent biological state through time.
5. `combinations` estimates interaction effects for multi-intervention experiments.

All estimators follow `fit`/`predict`-style protocols and return typed results. Implementations
should remain assay-neutral; modality adapters can be added separately.

