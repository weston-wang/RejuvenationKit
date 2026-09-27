from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rejuvenationkit.clocks import (
    age_acceleration,
    clock_observations,
    clock_table_from_biolearn,
    clock_table_from_pyaging,
)
from rejuvenationkit.schemas import Modality

START = datetime(2026, 1, 1, tzinfo=UTC)


def sample_table(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [item.split("-")[0] for item in ids],
            "timestamp": [START + timedelta(days=365 * (item.endswith("follow"))) for item in ids],
            "batch_id": ["plate-1"] * len(ids),
        },
        index=ids,
    )


def test_biolearn_single_model_and_run_predictions_formats() -> None:
    first = pd.DataFrame({"Predicted": [50.0, 60.0]}, index=["s1", "s2"])
    second = pd.DataFrame({"Predicted": [0.9, 1.1], "Other": [1, 2]}, index=["s1", "s2"])
    table = clock_table_from_biolearn({"Horvathv1": first, "DunedinPACE": second})
    assert list(table.columns) == ["Horvathv1", "DunedinPACE"]
    assert table.loc["s2", "DunedinPACE"] == 1.1

    gallery = pd.DataFrame({"Horvathv1": [50.0, np.nan]}, index=["s1", "s2"])
    assert clock_table_from_biolearn(gallery).isna().sum().sum() == 1

    with pytest.raises(ValueError, match="no 'Predicted'"):
        clock_table_from_biolearn({"Bad": pd.DataFrame({"x": [1.0]})})
    with pytest.raises(ValueError, match="non-numeric"):
        clock_table_from_biolearn(pd.DataFrame({"c": ["old"]}, index=["s1"]))
    with pytest.raises(ValueError, match="unique"):
        clock_table_from_biolearn(pd.DataFrame({"c": [1.0, 2.0]}, index=["s", "s"]))
    with pytest.raises(ValueError, match="empty"):
        clock_table_from_biolearn(pd.DataFrame())


def test_pyaging_obs_columns_match_case_insensitively() -> None:
    obs = pd.DataFrame(
        {"horvath2013": [40.0, 45.0], "tissue": ["blood", "blood"]}, index=["a", "b"]
    )
    table = clock_table_from_pyaging(SimpleNamespace(obs=obs), ["Horvath2013"])
    assert list(table.columns) == ["Horvath2013"]
    assert clock_table_from_pyaging(obs, ["horvath2013"]).shape == (2, 1)
    with pytest.raises(ValueError, match="no column"):
        clock_table_from_pyaging(obs, ["PhenoAge"])
    with pytest.raises(TypeError):
        clock_table_from_pyaging(object(), ["x"])
    with pytest.raises(ValueError, match="at least one"):
        clock_table_from_pyaging(obs, [])


def test_clock_observations_preserve_provenance_and_report_missing() -> None:
    ids = ["dog1-base", "dog1-follow", "dog2-base"]
    table = pd.DataFrame(
        {"Horvathv1": [5.0, 6.5, 7.0], "DunedinPACE": [1.0, np.nan, 0.9]}, index=ids
    )
    observations, missing = clock_observations(
        table,
        sample_table(ids),
        units={"Horvathv1": "years", "DunedinPACE": "years per year"},
        source="biolearn 0.9.1",
    )
    assert missing == (("dog1-follow", "DunedinPACE"),)
    assert len(observations) == 5
    follow = next(item for item in observations if item.replicate_id == "dog1-follow")
    assert follow.subject_id == "dog1"
    assert follow.modality is Modality.METHYLATION
    assert follow.timestamp == START + timedelta(days=365)
    assert follow.source_uri == "biolearn 0.9.1"
    assert follow.batch_id == "plate-1"
    assert follow.attributes["sample_id"] == "dog1-follow"

    with pytest.raises(ValueError, match="unknown samples"):
        clock_observations(table, sample_table(ids[:2]), source="biolearn")
    with pytest.raises(ValueError, match="no unit"):
        clock_observations(table, sample_table(ids), units={"Horvathv1": "y"}, source="x")
    with pytest.raises(ValueError, match="source"):
        clock_observations(table, sample_table(ids), source=" ")
    naive = sample_table(ids).assign(timestamp=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone"):
        clock_observations(table, naive, source="x")
    with pytest.raises(ValueError, match="missing columns"):
        clock_observations(table, sample_table(ids).drop(columns="subject_id"), source="x")


def test_age_acceleration_uses_reference_line_and_leave_one_out() -> None:
    ages = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 3.0], index=list("abcdef"))
    clock = pd.DataFrame({"clock": [1.1, 1.9, 3.2, 3.9, 5.1, 1.0]}, index=list("abcdef"))
    result = age_acceleration(clock, ages, reference_samples=list("abcde"))
    frame = result.to_frame()
    slope, intercept = np.polyfit(ages[:5], clock["clock"][:5], 1)
    assert result.fits[0].slope == pytest.approx(slope)
    assert frame.loc["f", "clock_acceleration"] == pytest.approx(1.0 - (intercept + slope * 3))
    loo_slope, loo_intercept = np.polyfit(ages[1:5], clock["clock"][1:5], 1)
    assert frame.loc["a", "clock_acceleration"] == pytest.approx(
        1.1 - (loo_intercept + loo_slope * 1.0)
    )

    with pytest.raises(ValueError, match="three reference"):
        age_acceleration(clock, ages, reference_samples=["a", "b"])
    with pytest.raises(ValueError, match="lack clock"):
        age_acceleration(clock, ages, reference_samples=["a", "b", "z"])
    with pytest.raises(ValueError, match="unknown clock"):
        age_acceleration(clock, ages, reference_samples=list("abc"), clocks=["other"])
    flat = pd.Series(3.0, index=list("abcdef"))
    with pytest.raises(ValueError, match="more than one"):
        age_acceleration(clock, flat, reference_samples=list("abc"))


def test_reference_line_avoids_treatment_leakage_in_pre_post_designs() -> None:
    random = np.random.default_rng(0)
    all_sample: list[float] = []
    reference_only: list[float] = []
    for _ in range(100):
        rows = []
        for dog in range(30):
            baseline_age = random.normal(8, 2)
            for visit, age in (("base", baseline_age), ("follow", baseline_age + 3)):
                effect = -2.0 if visit == "follow" else 0.0
                rows.append((f"d{dog}-{visit}", visit, age, age + effect + random.normal(0, 1.5)))
        frame = pd.DataFrame(rows, columns=["sample", "visit", "age", "clock"]).set_index("sample")
        slope, intercept = np.polyfit(frame["age"], frame["clock"], 1)
        pooled = frame["clock"] - (intercept + slope * frame["age"])
        all_sample.append(
            pooled[frame["visit"] == "follow"].mean() - pooled[frame["visit"] == "base"].mean()
        )
        accelerated = age_acceleration(
            frame[["clock"]],
            frame["age"],
            reference_samples=list(frame.index[frame["visit"] == "base"]),
        ).to_frame()["clock_acceleration"]
        reference_only.append(
            accelerated[frame["visit"] == "follow"].mean()
            - accelerated[frame["visit"] == "base"].mean()
        )
    # Fitting the age line on all samples absorbs part of the -2 effect into the slope.
    assert np.mean(all_sample) > -1.5
    assert np.mean(reference_only) == pytest.approx(-2.0, abs=0.15)


def test_real_biolearn_clock_flows_into_observations() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("biolearn")
    from biolearn.data_library import GeoData
    from biolearn.model_gallery import ModelGallery

    model = ModelGallery().get("Horvathv1")
    cpgs = [name for name in model.coefficients.index if str(name).startswith("cg")]
    random = np.random.default_rng(0)
    ids = ["dog1-base", "dog1-follow", "dog2-base"]
    methylation = pd.DataFrame(random.uniform(0.2, 0.8, (len(cpgs), 3)), index=cpgs, columns=ids)
    metadata = pd.DataFrame({"age": [5, 6, 7], "sex": [0, 0, 1]}, index=ids)
    predictions = model.predict(GeoData(metadata, methylation))

    table = clock_table_from_biolearn({"Horvathv1": predictions})
    observations, missing = clock_observations(table, sample_table(ids), source="biolearn")
    assert missing == ()
    assert [item.value for item in observations] == pytest.approx(predictions["Predicted"].tolist())
