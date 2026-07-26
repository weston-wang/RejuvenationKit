# Contributing

RejuvenationKit welcomes small, testable contributions grounded in a real research decision.

## Development

1. Use Python 3.11 or newer.
2. Install `python -m pip install -e ".[dev,docs]"`.
3. Install hooks with `pre-commit install`.
4. Add or update tests for every behavior change.
5. Run the local checks listed in the README.

## Scientific expectations

- State the estimand and assumptions.
- Report uncertainty; do not rely on significance alone.
- Avoid data leakage across subjects, batches, and time points.
- Validate against synthetic truth or a suitable public benchmark.
- Document failure modes, missing-data behavior, and determinism.

Public APIs require type hints and docstrings. Experimental APIs belong under
`rejuvenationkit.experimental` until their validation contract is stable.

