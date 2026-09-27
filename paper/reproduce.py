"""Regenerate every table in the RejuvenationKit methods note.

Usage::

    python paper/reproduce.py                 # full simulations (several minutes)
    python paper/reproduce.py --quick         # smoke test with few replicates
    python paper/reproduce.py --gse131754 PATH  # also rerun the public reanalysis

Each table is written as CSV to ``--output`` (default ``paper/results``) and
printed. Seeds are fixed, so a full run reproduces the note's numbers exactly.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from rejuvenationkit import (
    FusionConfig,
    Modality,
    ModalityEstimate,
    PrecisionWeightedFusion,
    RandomEffectsInterval,
    age_acceleration,
    simulate_trial_level_precision,
)
from rejuvenationkit._small_sample import variance_corrected, welch_critical_values
from rejuvenationkit.detection import _leave_one_out_scores, _regularized_covariance
from rejuvenationkit.genomics.concordance import InterventionContrast, intervention_concordance


def detector_calibration(replicates: int) -> pd.DataFrame:
    """Table 1: held-out false-alarm rate of in-sample vs leave-one-out thresholds."""
    random = np.random.default_rng(101)
    rows = []
    for reference, channels in ((20, 2), (20, 4), (20, 8), (50, 10), (517, 4)):
        rates: dict[str, list[float]] = {"in_sample": [], "leave_one_out": []}
        for _ in range(replicates):
            data = random.standard_normal((reference, channels))
            covariance = _regularized_covariance(data, shrinkage=0.2, ridge=1e-9)
            inverse = np.linalg.inv(covariance)
            centered = data - data.mean(axis=0)
            in_sample = np.einsum("ij,jk,ik->i", centered, inverse, centered)
            loo = _leave_one_out_scores(data, shrinkage=0.2, ridge=1e-9)
            held_out = random.standard_normal((200, channels)) - data.mean(axis=0)
            scores = np.einsum("ij,jk,ik->i", held_out, inverse, held_out)
            for name, reference_scores in (("in_sample", in_sample), ("leave_one_out", loo)):
                threshold = np.quantile(reference_scores, 0.95, method="higher")
                rates[name].append(float(np.mean(scores > threshold)))
        rows.append(
            {
                "reference_subjects": reference,
                "channels": channels,
                "in_sample_false_alarm": np.mean(rates["in_sample"]),
                "leave_one_out_false_alarm": np.mean(rates["leave_one_out"]),
            }
        )
    return pd.DataFrame(rows)


def fusion_coverage(replicates: int) -> pd.DataFrame:
    """Table 2: 95% coverage of random-effects fusion with few modalities."""
    random = np.random.default_rng(202)
    modalities = (
        Modality.METHYLATION,
        Modality.TRANSCRIPTOMICS,
        Modality.PROTEOMICS,
        Modality.CLINICAL,
    )
    wald = PrecisionWeightedFusion(FusionConfig(random_effects_interval=RandomEffectsInterval.WALD))
    hartung_knapp = PrecisionWeightedFusion()
    rows = []
    for count in (2, 3, 4):
        for heterogeneity in (0.0, 0.5, 1.0):
            covered = {"wald": 0, "hartung_knapp": 0}
            for _ in range(replicates):
                errors = random.uniform(0.2, 0.6, count)
                values = random.normal(0.0, heterogeneity, count) + random.normal(0.0, errors)
                estimates = tuple(
                    ModalityEstimate(
                        modality=m, estimate=float(v), standard_error=float(e), target="t"
                    )
                    for m, v, e in zip(modalities, values, errors, strict=False)
                )
                for name, fusion in (("wald", wald), ("hartung_knapp", hartung_knapp)):
                    low, high = fusion.fuse(estimates).confidence_interval
                    covered[name] += low <= 0 <= high
            rows.append(
                {
                    "modalities": count,
                    "between_modality_sd": heterogeneity,
                    "dersimonian_laird_wald": covered["wald"] / replicates,
                    "hartung_knapp": covered["hartung_knapp"] / replicates,
                }
            )
    return pd.DataFrame(rows)


def small_sample_intervals(replicates: int) -> pd.DataFrame:
    """Table 3: coverage of percentile vs corrected Welch t bootstrap intervals."""
    random = np.random.default_rng(303)
    rows = []
    for per_arm in (3, 8, 20):
        percentile = corrected = 0
        for _ in range(replicates):
            treated = random.standard_normal(per_arm)
            control = random.standard_normal(per_arm)
            indices_t = random.integers(0, per_arm, (1_000, per_arm))
            indices_c = random.integers(0, per_arm, (1_000, per_arm))
            raw = treated[indices_t].mean(axis=1) - control[indices_c].mean(axis=1)
            low, high = np.quantile(raw, [0.025, 0.975])
            percentile += low <= 0 <= high
            t_c, c_c = variance_corrected(treated), variance_corrected(control)
            boot = t_c[indices_t].mean(axis=1) - c_c[indices_c].mean(axis=1)
            critical = welch_critical_values(t_c[None, :], c_c[None, :], 0.95)[0]
            estimate = treated.mean() - control.mean()
            corrected += abs(estimate) <= critical * boot.std(ddof=1)
        rows.append(
            {
                "subjects_per_arm": per_arm,
                "percentile_bootstrap": percentile / replicates,
                "corrected_welch_t": corrected / replicates,
            }
        )
    return pd.DataFrame(rows)


def shared_control_concordance(replicates: int) -> pd.DataFrame:
    """Table 4: naive vs corrected concordance when two arms share controls."""
    rows = []
    for true_correlation, signal_sd in ((0.0, 0.3), (0.6, 0.3), (-0.4, 0.3), (0.0, 0.0)):
        naive: list[float] = []
        corrected: list[float] = []
        rejections = 0
        for replicate in range(replicates):
            random = np.random.default_rng(400 + replicate)
            effects = random.multivariate_normal(
                [0, 0],
                np.array([[1, true_correlation], [true_correlation, 1]]) * signal_sd**2,
                2_000,
            )
            columns, groups, strata, data = [], [], [], []
            for stratum in ("s1", "s2"):
                baseline = random.normal(5, 1, 2_000)
                for group, shift in (("A", effects[:, 0]), ("B", effects[:, 1]), ("C", 0.0)):
                    for index in range(3):
                        data.append(baseline + shift + random.normal(0, 0.5, 2_000))
                        columns.append(f"{group}{stratum}{index}")
                        groups.append(group)
                        strata.append(stratum)
            result = intervention_concordance(
                pd.DataFrame(np.asarray(data).T, columns=columns),
                pd.DataFrame({"group": groups, "stratum": strata}, index=columns),
                InterventionContrast(treated_group="A", control_group="C"),
                InterventionContrast(treated_group="B", control_group="C"),
                permutations=99,
                random_seed=replicate,
            )
            naive.append(result.naive_correlation)
            if result.corrected_correlation is not None:
                corrected.append(result.corrected_correlation)
            rejections += result.permutation_p_value <= 0.05
        rows.append(
            {
                "true_correlation": true_correlation if signal_sd > 0 else float("nan"),
                "signal_sd": signal_sd,
                "mean_naive": np.mean(naive),
                "mean_corrected": np.mean(corrected) if corrected else float("nan"),
                "rejection_rate": rejections / replicates,
            }
        )
    return pd.DataFrame(rows)


def age_acceleration_leakage(replicates: int) -> pd.DataFrame:
    """Table 5: bias of all-sample vs reference-only age acceleration."""
    random = np.random.default_rng(505)

    def pooled_residual(frame: pd.DataFrame) -> pd.Series:
        slope, intercept = np.polyfit(frame["age"], frame["clock"], 1)
        return frame["clock"] - (intercept + slope * frame["age"])

    designs: dict[str, Callable[[], tuple[pd.DataFrame, list[str], pd.Series, pd.Series]]] = {}

    def pre_post() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
        rows = []
        for dog in range(30):
            age0 = random.normal(8, 2)
            for visit, age in (("base", age0), ("follow", age0 + 3)):
                rows.append(
                    (
                        f"d{dog}{visit}",
                        visit,
                        age,
                        age + random.normal(0, 1.5) - 2 * (visit == "follow"),
                    )
                )
        frame = pd.DataFrame(rows, columns=["id", "group", "age", "clock"]).set_index("id")
        return (
            frame,
            list(frame.index[frame["group"] == "base"]),
            frame["group"] == "follow",
            frame["group"] == "base",
        )

    def imbalanced() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
        rows = []
        for group, mean_age in (("treated", 12), ("control", 8)):
            for dog in range(40):
                age = random.normal(mean_age, 2)
                rows.append(
                    (
                        f"{group}{dog}",
                        group,
                        age,
                        age + random.normal(0, 1.5) - 2 * (group == "treated"),
                    )
                )
        frame = pd.DataFrame(rows, columns=["id", "group", "age", "clock"]).set_index("id")
        return (
            frame,
            list(frame.index[frame["group"] == "control"]),
            frame["group"] == "treated",
            frame["group"] == "control",
        )

    def randomized() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
        rows = []
        for arm in ("treated", "control"):
            for dog in range(30):
                age0 = random.normal(8, 2)
                for visit, age in (("base", age0), ("follow", age0 + 3)):
                    effect = -2 * (arm == "treated" and visit == "follow")
                    rows.append(
                        (
                            f"{arm}{dog}{visit}",
                            f"{arm}-{visit}",
                            age,
                            age + random.normal(0, 1.5) + effect,
                        )
                    )
        frame = pd.DataFrame(rows, columns=["id", "group", "age", "clock"]).set_index("id")
        return (
            frame,
            list(frame.index[frame["group"].str.startswith("control")]),
            frame["group"],
            frame["group"],
        )

    designs["randomized difference-in-differences"] = randomized
    designs["single-arm pre-post"] = pre_post
    designs["cross-sectional, treated 4 y older"] = imbalanced
    rows = []
    for name, build in designs.items():
        pooled, reference = [], []
        for _ in range(replicates):
            frame, reference_ids, exposed, comparison = build()
            all_sample = pooled_residual(frame)
            safe = age_acceleration(frame[["clock"]], frame["age"], reference_samples=reference_ids)
            values = safe.to_frame()["clock_acceleration"]
            if exposed.dtype == bool:
                pooled.append(all_sample[exposed].mean() - all_sample[comparison].mean())
                reference.append(values[exposed].mean() - values[comparison].mean())
            else:
                pooled.append(_difference_in_differences(all_sample, frame["group"]))
                reference.append(_difference_in_differences(values, frame["group"]))
        rows.append(
            {
                "design": name,
                "true_effect": -2.0,
                "all_sample_regression": np.mean(pooled),
                "reference_only": np.mean(reference),
            }
        )
    return pd.DataFrame(rows)


def _difference_in_differences(values: pd.Series, groups: pd.Series) -> float:
    means = values.groupby(groups).mean()
    return float(
        (means["treated-follow"] - means["treated-base"])
        - (means["control-follow"] - means["control-base"])
    )


def surrogate_precision(replicates: int) -> pd.DataFrame:
    """Table 6: how many independent units trial-level surrogate validation needs."""
    rows = []
    for correlation in (0.9, 0.7):
        for units in (5, 10, 20, 40):
            table = simulate_trial_level_precision(
                unit_count=units,
                trial_correlation=correlation,
                surrogate_reliability=0.8,
                simulations=replicates,
                bootstrap_samples=200,
                random_seed=units,
            )
            table = table[table["estimable"]]
            rows.append(
                {
                    "units": units,
                    "true_r_squared_trial": correlation**2,
                    "median_interval_width": table["interval_width"].median(),
                    "probability_lower_bound_above_0_5": (table["interval_low"] > 0.5).mean(),
                    "coverage": table["covers_truth"].mean(),
                }
            )
    return pd.DataFrame(rows)


def public_reanalysis(path: Path, permutations: int) -> pd.DataFrame:
    """Table 7: GSE131754 cross-intervention concordance (downloads if absent)."""
    from rejuvenationkit.datasets import (
        INTERVENTION_CONTROLS,
        download_counts,
        filtered_log2_cpm,
        intervention_sample_table,
        read_counts,
    )
    from rejuvenationkit.genomics.concordance import StrataPolicy, concordance_family

    values = filtered_log2_cpm(read_counts(download_counts(path)))
    family = concordance_family(
        values,
        intervention_sample_table(values),
        tuple(
            InterventionContrast(treated_group=treated, control_group=control)
            for treated, control in INTERVENTION_CONTROLS.items()
        ),
        strata_policy=StrataPolicy.OWN,
        permutations=permutations,
    )
    return family.to_frame()


def main() -> None:
    """Run every table and write CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="few replicates, for smoke tests")
    parser.add_argument("--output", type=Path, default=Path("paper/results"))
    parser.add_argument("--gse131754", type=Path, default=None, help="count-matrix cache path")
    args = parser.parse_args()
    scale = 0.02 if args.quick else 1.0
    replicates = lambda full: max(int(full * scale), 3)  # noqa: E731
    tables: dict[str, Callable[[], pd.DataFrame]] = {
        "table1_detector_calibration": lambda: detector_calibration(replicates(2_000)),
        "table2_fusion_coverage": lambda: fusion_coverage(replicates(10_000)),
        "table3_small_sample_intervals": lambda: small_sample_intervals(replicates(4_000)),
        "table4_shared_control_concordance": lambda: shared_control_concordance(replicates(100)),
        "table5_age_acceleration_leakage": lambda: age_acceleration_leakage(replicates(300)),
        "table6_surrogate_precision": lambda: surrogate_precision(replicates(150)),
    }
    if args.gse131754 is not None:
        tables["table7_gse131754_concordance"] = lambda: public_reanalysis(
            args.gse131754, 19 if args.quick else 999
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, build in tables.items():
        table = build()
        table.to_csv(args.output / f"{name}.csv", index=False)
        print(f"\n{name}")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
