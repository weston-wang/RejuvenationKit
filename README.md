# RejuvenationKit

[![CI](https://github.com/weston-wang/RejuvenationKit/actions/workflows/ci.yml/badge.svg)](https://github.com/weston-wang/RejuvenationKit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

RejuvenationKit is a planned Python SDK for reproducible, uncertainty-aware analysis of
longitudinal rejuvenation studies. It will sit above established assay-specific pipelines and
help researchers answer a systems-level question: **what changed, how confidently, and for how
long?**

> **Status:** pre-alpha. Phase 1 longitudinal QC is implemented; later phases retain small
> placeholder interfaces. This project is for research use and is not medical software.

## Architecture

```text
Study / subject / observation schemas
                 │
        ┌────────┴────────┐
        │ Phase 1: QC     │  longitudinal integrity, drift, outliers
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │ Phase 2: Fusion │  multimodal estimates + uncertainty
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │ Phase 3: State  │  latent biological-state tracking
        └────────┬────────┘
                 │
        ┌────────┴──────────────┐
        │ Phase 4: Full SDK     │  workflows, combinations, reporting
        └───────────────────────┘
```

The core schemas are assay-neutral. An observation identifies a subject, time point, modality,
feature, value, unit, and optional uncertainty. Algorithms consume validated `Study` objects and
return typed result objects rather than unstructured tables.

## Roadmap

- **Phase 1 — `aging-qc` (baseline implemented):** subject- and visit-level missingness,
  absolute and subject-relative visit windows, input ordering, range and unit checks, batch mean
  shifts, treatment/batch confounding, and replicate consistency.
- **Phase 2 — `aging-fusion`:** fuse clocks, omics, pathology, imaging, and clinical
  biomarkers while preserving modality-level uncertainty and disagreement.
- **Phase 3 — `aging-state`:** longitudinal latent-state estimation, smoothing, change-point
  detection, and forecast validation.
- **Phase 4 — full SDK:** stable workflows, combination-therapy interaction analysis,
  reporting, adapters, documentation, and public benchmark datasets.

Milestones and acceptance criteria live in [docs/roadmap.md](docs/roadmap.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
pre-commit install
pytest
```

```python
from datetime import datetime, timezone

from rejuvenationkit.schemas import Modality, Observation, Study, Subject

study = Study(
    study_id="demo",
    subjects=[Subject(subject_id="mouse-001", cohort="treated")],
    observations=[
        Observation(
            subject_id="mouse-001",
            timestamp=datetime.now(timezone.utc),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=31.2,
            unit="g",
        )
    ],
)
```

See [`examples/minimal_study.py`](examples/minimal_study.py) and
[`examples/phase_1_qc.py`](examples/phase_1_qc.py), with sample data in
[`examples/data/longitudinal_observations.csv`](examples/data/longitudinal_observations.csv).
The [`examples/rapamycin_phase_1_qc.py`](examples/rapamycin_phase_1_qc.py) example demonstrates
staggered dosing anchors and balanced treatment batches.
The [`examples/public_gse131754_rapamycin.py`](examples/public_gse131754_rapamycin.py) workflow
downloads a real public mouse RNA-seq dataset, converts it into typed study data, runs QC, and
produces explicitly exploratory rapamycin-versus-control expression contrasts.
The [`examples/public_dog_aging_project.py`](examples/public_dog_aging_project.py) workflow
downloads real longitudinal blood chemistry measurements from pet dogs, checks expected-visit
missingness, and summarizes paired changes. It has no treatment assignment and is not a
rapamycin-effectiveness analysis.

## Contribution workflow

1. Open an issue describing the scientific decision and expected validation.
2. Create a focused branch and add tests before or with the implementation.
3. Run `ruff check .`, `ruff format --check .`, `mypy`, and `pytest`.
4. Open a pull request using the template and document assumptions, validation data, and limits.

New algorithms should expose uncertainty, accept deterministic random seeds when applicable, and
include a synthetic or public-data validation case. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Scope boundaries

RejuvenationKit will not replace base calling, alignment, differential-expression software,
clinical judgment, or regulatory validation. It is designed as a transparent decision layer over
curated assay outputs. No therapy recommendations are produced.

## License

Apache License 2.0. See [LICENSE](LICENSE).
