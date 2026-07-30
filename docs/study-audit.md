# One-command Phase 1 study audit

`run_phase1_audit` turns the individual Phase 1 components into one reproducible workflow. It
runs longitudinal QC, analysis-readiness profiling, optional randomized treatment inference, and
visual reporting, then writes a consistent artifact bundle.

```python
from pathlib import Path

from rejuvenationkit import Phase1AuditConfig, run_phase1_audit

report = run_phase1_audit(
    study,
    config=Phase1AuditConfig(qc=qc_config),
    output_dir=Path("artifacts/my-study"),
)
print(report.summary_markdown())
```

## Bundle contents

Every audit contains:

| Artifact | Purpose |
|---|---|
| `audit.json` | Complete typed result, configuration, source metadata, input fingerprint, and manifest |
| `manifest.json` | Byte size and SHA-256 digest for every other exported artifact |
| `summary.md` | Short decision-oriented human report |
| `findings.csv` | Stable QC codes, severities, affected subjects, and serialized context |
| `visit_coverage.csv` | Feature availability at every expected visit |
| `visit_retention.csv` | Complete-case retention between visits |
| `paired_readiness.csv` | Feature-specific paired-analysis sample sizes |
| `feature_distributions.csv` | Visit-level distributions and robust outliers |
| `attrition_bias.csv` | Baseline differences between retained and attrited subjects |
| `audit_overview.png` | Findings, coverage, retention, and outlier overview |

The overview requires the `visualization` installation extra. Set
`include_visualizations=False` for a dependency-light machine-readable audit.

## Optional held-out change detection

Add a `ChangeDetectionAuditPlan` to fit covariance and thresholds only on prespecified reference
subjects, then score a disjoint evaluation set. The audit adds the subject scores and standard
DSP diagnostic figures to the same bundle. Reference/evaluation overlap and unknown visit names
are rejected.

## Optional randomized comparison

Add a `TreatmentAuditPlan` to run the leakage-safe Phase 1 randomized evaluator using visits
already defined in the audit QC configuration:

```python
from rejuvenationkit import TreatmentAuditPlan

plan = TreatmentAuditPlan(
    config=treatment_effect_config,
    baseline_visit_id="baseline",
    follow_up_visit_ids=("month-1", "month-3", "month-6"),
    treated_subject_ids=treated_ids,
    control_subject_ids=control_ids,
    treated_label="rapamycin",
    control_label="placebo",
)

report = run_phase1_audit(
    study,
    config=audit_config,
    output_dir=Path("artifacts/rapamycin-study"),
    treatment_plan=plan,
)
```

This adds `treatment_effects.csv` and `treatment_subject_scores.csv`. Group membership is always
explicit; the audit does not infer a causal comparison from cohort names.

## Public canine validation

`examples/public_dog_phase1_audit.py` downloads the public Dog Aging Project phenotype archive,
adapts two longitudinal chemistry waves, creates a deterministic calibration/evaluation split,
and runs the full audit bundle with held-out multivariate change detection:

```bash
python examples/public_dog_phase1_audit.py
```

The dataset is an observational cohort without rapamycin assignment. The example therefore omits
the treatment plan and makes no efficacy claim. It validates ingestion, expected-visit policies,
missingness, retention, outlier profiling, held-out DSP scoring, serialization, and reporting on
external canine data.

## Reproducibility and interpretation

The JSON report contains the complete audit configuration, study metadata, and a canonical
SHA-256 fingerprint of the validated input model. The separate manifest records byte sizes and
SHA-256 digests for every other artifact. The input fingerprint detects changes to subjects,
observations, ordering, or metadata; it is not a substitute for archiving the source dataset.

A `PASS` means only that the configured QC checks found no errors. Warnings, analysis readiness,
attrition, study design, endpoint definitions, and the prespecified statistical analysis plan
still determine whether an efficacy claim is supportable.
