# RejuvenationKit

[![CI](https://github.com/weston-wang/RejuvenationKit/actions/workflows/ci.yml/badge.svg)](https://github.com/weston-wang/RejuvenationKit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rejuvenationkit.svg)](https://pypi.org/project/rejuvenationkit/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11–3.13-blue.svg)](pyproject.toml)

RejuvenationKit is a typed Python toolkit for auditing and analyzing longitudinal preclinical
rejuvenation studies. It sits above established assay-specific pipelines and helps research teams
answer:

> **Is this study trustworthy, did the intervention produce a coherent response, when did it
> appear, and which biological systems drove it?**

Phase 1 is complete and available as an alpha. It includes protocol-aware quality control,
analysis-readiness profiling, experimental-confounding checks, held-out DSP change detection,
sequential response monitoring, randomized longitudinal inference, and reproducible report
bundles.

This project is for research use. It is not medical software and does not produce treatment
recommendations.

## Who this is for

RejuvenationKit is designed for:

- preclinical gene-therapy and longevity teams comparing constructs, doses, or vector lots;
- canine-aging and veterinary-trial researchers with repeated visits and heterogeneous endpoints;
- computational biologists who need a decision layer above RNA-seq, methylation, proteomics,
  histology, imaging, or clinical-assay pipelines; and
- collaborators reviewing whether a study is ready to support an efficacy claim.

It is especially useful when an experiment combines several noisy readouts and risks confusing
biology with site, plate, assay run, operator, manufacturing lot, or visit timing.

## What it catches

- Missing visits and missing features at expected visits
- Treatment, cohort, or timepoint confounding with experimental handling
- Batch shifts, replicate disagreement, distribution anomalies, and attrition bias
- Weak paired-analysis sample sizes hidden by apparently large enrollment
- Multichannel responses that emerge gradually or persist across visits
- Individual responders and dominant evidence modalities
- Randomized treatment effects calibrated without fitting the null model on treated subjects

## Public canine validation

The one-command audit was run end to end on public Dog Aging Project longitudinal chemistry data:

| Result | Observed |
|---|---:|
| Dogs | 972 |
| Long-form observations | 6,808 |
| Complete-case retention to the second wave | 75.5% |
| Held-out complete trajectories scored | 213 |
| Held-out trajectories crossing the nominal 5% threshold | 11 |

The cohort is observational and contains no rapamycin assignment, so detections are not treatment
effects. The case demonstrates real ingestion, missingness, retention, held-out calibration,
reporting, and artifact integrity. See [the DAP audit case study](docs/dap-audit-case-study.md).

## Evaluate it with a study

The most valuable feedback is a de-identified, simulated, or public dataset shaped like a real
preclinical workflow. Open a
[study-evaluation request](https://github.com/weston-wang/RejuvenationKit/issues/new?template=study-evaluation.yml)
with the decision, visit schedule, modalities, and known complications. Do not attach confidential
or identifiable data to a public issue.

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
  shifts, treatment/batch/site/plate/lot/timepoint confounding, replicate consistency, visit
  coverage, longitudinal
  retention, paired-analysis readiness, robust outliers, attrition-bias diagnostics, and
  covariance-aware multivariate and sequential change detection, leakage-safe control
  calibration, and randomized longitudinal treatment-effect inference.
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
python -m pip install "rejuvenationkit[visualization]"
```

Until `0.2.0a2` is published to PyPI, install the current source release from GitHub as described
in [the publishing guide](docs/publishing.md).

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

Run `run_phase1_audit(...)` with a `QCConfig` and output directory to create the JSON, CSV,
Markdown, manifest, and visualization bundle. The complete workflow is documented in
[the study-audit guide](docs/study-audit.md).

For development:

```bash
python -m pip install -e ".[dev,docs,visualization]"
pre-commit install
pytest
```

See [`examples/minimal_study.py`](examples/minimal_study.py) and
[`examples/phase_1_qc.py`](examples/phase_1_qc.py), with sample data in
[`examples/data/longitudinal_observations.csv`](examples/data/longitudinal_observations.csv).
For a complete report bundle, call `run_phase1_audit(...)` or run
[`examples/public_dog_phase1_audit.py`](examples/public_dog_phase1_audit.py). The audit writes its
configuration, input fingerprint, QC findings, readiness tables, summary, and overview figure in
one operation. Optional plans add held-out multivariate detection or randomized treatment
inference without changing the underlying study.

The [`examples/rapamycin_phase_1_qc.py`](examples/rapamycin_phase_1_qc.py) example demonstrates
staggered dosing anchors and balanced treatment batches.
The [`examples/public_gse131754_rapamycin.py`](examples/public_gse131754_rapamycin.py) workflow
downloads a real public mouse RNA-seq dataset, converts it into typed study data, runs QC, and
produces explicitly exploratory rapamycin-versus-control expression contrasts.
The [`examples/public_dog_aging_project.py`](examples/public_dog_aging_project.py) workflow
downloads real longitudinal blood chemistry measurements from pet dogs, checks expected-visit
missingness, and summarizes paired changes. It has no treatment assignment and is not a
rapamycin-effectiveness analysis.
The [`examples/public_dog_multimodal.py`](examples/public_dog_multimodal.py) workflow combines
aligned clinical chemistry and metabolomics, then applies held-out covariance-aware change
detection across both modalities. Install `.[public-data]` to read the source R-data object and
`.[visualization]` to export covariance, detection-score, whitening, and decomposition figures.
The [`examples/gene_therapy_confounding.py`](examples/gene_therapy_confounding.py) workflow
demonstrates how a superficially complete canine gene-therapy study can still be unusable because
site, vector lot, and visit-specific plates overlap the biological contrasts.
The [`examples/randomized_rapamycin_effect.py`](examples/randomized_rapamycin_effect.py) workflow
demonstrates out-of-fold control calibration, covariance-aware randomization testing, and
feature-level longitudinal effect intervals in a synthetic 60-dog trial.
The [`examples/public_dog_sequential.py`](examples/public_dog_sequential.py) workflow learns
reference dynamics over three Precision waves, reports onset and persistence of unusual held-out
trajectories, and exports sequential evidence and modality plots. It is an observational
monitoring demonstration, not a treatment-effect analysis.
The [`examples/triad_like_rapamycin_sequential.py`](examples/triad_like_rapamycin_sequential.py)
workflow generates a clearly labeled synthetic 580-dog, seven-visit trial shaped like the public
TRIAD protocol. It demonstrates responder onset, persistence, transient effects, and
modality-localized evidence; it contains no DAP treatment outcomes.

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

## Citation

If RejuvenationKit contributes to research, cite the archived software release used in the
analysis. Citation metadata are available in [CITATION.cff](CITATION.cff); a DOI will be added
after Zenodo archives the release.
