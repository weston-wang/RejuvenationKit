from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.datasets import filtered_log2_cpm, intervention_sample_table
from rejuvenationkit.genomics.concordance import (
    ConcordanceFamily,
    InterventionContrast,
    StrataPolicy,
    concordance_family,
    holm_adjust,
    intervention_concordance,
)

A = InterventionContrast(treated_group="A", control_group="C")
B = InterventionContrast(treated_group="B", control_group="C")


def shared_control_data(
    *,
    true_correlation: float,
    signal_sd: float,
    genes: int = 2_000,
    replicates: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random = np.random.default_rng(seed)
    covariance = np.array([[1.0, true_correlation], [true_correlation, 1.0]]) * signal_sd**2
    effects = random.multivariate_normal([0.0, 0.0], covariance, genes)
    columns: list[str] = []
    groups: list[str] = []
    strata: list[str] = []
    data: list[np.ndarray] = []
    for stratum in ("young", "old"):
        baseline = random.normal(5.0, 1.0, genes)
        for group, shift in (("A", effects[:, 0]), ("B", effects[:, 1]), ("C", 0.0)):
            for replicate in range(replicates):
                data.append(baseline + shift + random.normal(0.0, 0.5, genes))
                columns.append(f"{group}-{stratum}-{replicate}")
                groups.append(group)
                strata.append(stratum)
    values = pd.DataFrame(np.asarray(data).T, columns=columns)
    samples = pd.DataFrame({"group": groups, "stratum": strata}, index=columns)
    return values, samples


def test_shared_control_artifact_is_removed_for_unrelated_effects() -> None:
    values, samples = shared_control_data(true_correlation=0.0, signal_sd=0.3)
    result = intervention_concordance(values, samples, A, B, permutations=49)

    assert result.shared_control_strata == ("old", "young")
    assert result.naive_correlation > 0.15
    assert result.corrected_correlation is not None
    assert abs(result.corrected_correlation) < 0.06
    assert result.disjoint_control_correlation == pytest.approx(
        result.corrected_correlation, abs=0.05
    )
    assert result.artifact_correlation == pytest.approx(
        result.naive_correlation - result.corrected_correlation
    )


@pytest.mark.parametrize("true_correlation", [0.6, -0.4])
def test_corrected_correlation_recovers_true_effect_correlation(true_correlation: float) -> None:
    values, samples = shared_control_data(true_correlation=true_correlation, signal_sd=0.3)
    result = intervention_concordance(values, samples, A, B, permutations=99)

    assert result.corrected_correlation == pytest.approx(true_correlation, abs=0.08)
    assert result.disjoint_control_correlation == pytest.approx(true_correlation, abs=0.08)
    assert result.permutation_p_value <= 0.05


def test_pure_noise_reports_no_corrected_correlation_and_valid_p_value() -> None:
    values, samples = shared_control_data(true_correlation=0.0, signal_sd=0.0)
    result = intervention_concordance(values, samples, A, B, permutations=49)

    # The shared control alone makes two noise vectors correlate near 0.5.
    assert result.naive_correlation == pytest.approx(0.5, abs=0.08)
    assert result.corrected_correlation is None
    assert result.disjoint_control_correlation is None

    rejections = [
        intervention_concordance(
            *shared_control_data(true_correlation=0.0, signal_sd=0.0, genes=300, seed=seed),
            A,
            B,
            permutations=39,
            random_seed=seed,
        ).permutation_p_value
        <= 0.05
        for seed in range(60)
    ]
    # Exact under the complete null: about 5% of replicates reject.
    assert np.mean(rejections) <= 0.12


def test_independent_controls_need_no_correction() -> None:
    values, samples = shared_control_data(true_correlation=0.0, signal_sd=0.3)
    samples = samples.copy()
    b_old = (samples["group"] == "B") & (samples["stratum"] == "old")
    c_old = (samples["group"] == "C") & (samples["stratum"] == "old")
    samples.loc[b_old, "stratum"] = "b-only"
    extra = values.loc[:, c_old].copy()
    extra.columns = [f"D-{name}" for name in extra.columns]
    values = pd.concat([values, extra], axis=1)
    samples = pd.concat(
        [samples, pd.DataFrame({"group": "D", "stratum": "b-only"}, index=extra.columns)]
    )
    result = intervention_concordance(
        values,
        samples,
        A,
        InterventionContrast(treated_group="B", control_group="D"),
        strata_policy=StrataPolicy.OWN,
        permutations=19,
    )
    assert result.shared_control_strata == ()
    assert result.shared_control_noise_covariance == 0.0
    assert result.second_strata == ("b-only",)


def test_common_policy_requires_a_shared_stratum() -> None:
    values, samples = shared_control_data(true_correlation=0.0, signal_sd=0.3)
    samples = samples.copy()
    samples.loc[samples["group"] == "B", "stratum"] = "elsewhere"
    samples.loc[(samples["group"] == "C") & (samples["stratum"] == "old"), "stratum"] = "elsewhere"
    with pytest.raises(ValueError, match="share no stratum"):
        intervention_concordance(values, samples, A, B, permutations=9)


def test_inputs_are_validated() -> None:
    values, samples = shared_control_data(true_correlation=0.0, signal_sd=0.3, genes=50)
    with pytest.raises(ValueError, match="same samples"):
        intervention_concordance(values, samples.iloc[::-1], A, B, permutations=9)
    with pytest.raises(ValueError, match="missing columns"):
        intervention_concordance(values, samples.drop(columns="stratum"), A, B)
    with pytest.raises(ValueError, match="permutations"):
        intervention_concordance(values, samples, A, B, permutations=0)
    with pytest.raises(ValidationError, match="must differ"):
        InterventionContrast(treated_group="C", control_group="C")
    with pytest.raises(ValueError, match="at least two treated"):
        intervention_concordance(
            values,
            samples,
            A,
            InterventionContrast(treated_group="Z", control_group="C"),
        )


def test_family_holm_adjusts_and_rejects_forged_adjustment() -> None:
    values, samples = shared_control_data(true_correlation=0.6, signal_sd=0.3, genes=500)
    samples = samples.copy()
    samples.loc[samples.index[:3], "group"] = "A"
    family = concordance_family(
        values,
        samples,
        (A, B, InterventionContrast(treated_group="A", control_group="B")),
        permutations=19,
    )
    assert len(family.results) == 3
    raw = tuple(item.permutation_p_value for item in family.results)
    assert family.holm_adjusted_p_values == pytest.approx(holm_adjust(raw))
    assert list(family.to_frame().columns)[:2] == ["first", "second"]

    with pytest.raises(ValidationError, match="Holm"):
        ConcordanceFamily(
            results=family.results,
            holm_adjusted_p_values=tuple(0.5 for _ in raw),
        )
    with pytest.raises(ValueError, match="at least two"):
        concordance_family(values, samples, (A,))
    with pytest.raises(ValueError, match="unique"):
        concordance_family(values, samples, (A, A))


def test_holm_adjust_matches_step_down_definition() -> None:
    assert holm_adjust((0.01, 0.04, 0.03)) == pytest.approx((0.03, 0.06, 0.06))
    assert holm_adjust((0.5, 0.9)) == pytest.approx((1.0, 1.0))


def test_gse131754_design_helpers() -> None:
    counts = pd.DataFrame(
        {
            "RAP_6m_F_1": [100, 0, 50],
            "CON_6m_F_1": [120, 1, 40],
            "GHRKO_5m_M_1": [90, 0, 60],
        },
        index=pd.Index(["g1", "g2", "g3"], name="GENE_ID"),
    )
    values = filtered_log2_cpm(counts, minimum_cpm=1.0, minimum_sample_fraction=0.5)
    assert list(values.index) == ["g1", "g3"]
    samples = intervention_sample_table(values)
    assert samples.loc["RAP_6m_F_1"].tolist() == ["RAP", "6m-F"]
    assert samples.loc["GHRKO_5m_M_1"].tolist() == ["GHRKO", "5m-M"]
    with pytest.raises(ValueError, match="fraction"):
        filtered_log2_cpm(counts, minimum_sample_fraction=0.0)
    with pytest.raises(ValueError, match="no genes"):
        filtered_log2_cpm(counts, minimum_cpm=1e9)
