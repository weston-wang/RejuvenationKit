import gzip
import io
from pathlib import Path

import pandas as pd
import pytest

from rejuvenationkit.datasets import gse131754
from rejuvenationkit.datasets.gse131754 import (
    build_study,
    descriptive_log2_fold_changes,
    download_counts,
    parse_sample_name,
    read_counts,
    select_rapamycin_and_controls,
)


def matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RAP_6m_F_1": [10, 20],
            "RAP_6m_M_1": [12, 18],
            "CON_6m_F_1": [5, 25],
            "CON_6m_M_1": [6, 24],
            "ACA_6m_F_1": [8, 22],
        },
        index=pd.Index(["gene-a", "gene-b"], name="GENE_ID"),
    )


def write_counts(path: Path, counts: pd.DataFrame) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        counts.to_csv(output, sep="\t")


def test_parse_sample_name() -> None:
    sample = parse_sample_name("RAP_12m_F_3")
    assert sample.intervention_code == "RAP"
    assert sample.age_months == 12
    assert sample.sex == "F"
    assert sample.replicate == 3
    assert sample.is_rapamycin
    assert not sample.is_control

    control = parse_sample_name("CON_6m_M_1")
    assert control.is_control
    with pytest.raises(ValueError, match="unrecognized"):
        parse_sample_name("not-a-sample")


def test_read_and_select_counts(tmp_path: Path) -> None:
    path = tmp_path / "counts.tsv.gz"
    write_counts(path, matrix())
    counts = read_counts(path)
    selected = select_rapamycin_and_controls(counts, ages_months=(6,))
    assert selected.columns.tolist() == [
        "RAP_6m_F_1",
        "RAP_6m_M_1",
        "CON_6m_F_1",
        "CON_6m_M_1",
    ]
    with pytest.raises(ValueError, match="no matching"):
        select_rapamycin_and_controls(counts, ages_months=(99,))


def test_read_counts_rejects_invalid_matrices(tmp_path: Path) -> None:
    negative = matrix()
    negative.iloc[0, 0] = -1
    negative_path = tmp_path / "negative.tsv.gz"
    write_counts(negative_path, negative)
    with pytest.raises(ValueError, match="negative"):
        read_counts(negative_path)

    invalid_name = matrix().rename(columns={"RAP_6m_F_1": "invalid"})
    invalid_path = tmp_path / "invalid.tsv.gz"
    write_counts(invalid_path, invalid_name)
    with pytest.raises(ValueError, match="unrecognized"):
        read_counts(invalid_path)


def test_build_study_maps_public_metadata() -> None:
    selected = select_rapamycin_and_controls(matrix())
    study = build_study(selected, gene_ids=("gene-a", "gene-b"), source_uri="fixture")
    assert len(study.subjects) == 4
    assert len(study.observations) == 8
    treated = next(subject for subject in study.subjects if subject.subject_id == "RAP_6m_F_1")
    assert treated.interventions == ("rapamycin",)
    assert treated.attributes["sex"] == "F"
    assert study.observations[0].source_uri == "fixture"

    with pytest.raises(ValueError, match="absent"):
        build_study(selected, gene_ids=("missing",))


def test_descriptive_log2_fold_changes() -> None:
    selected = select_rapamycin_and_controls(matrix())
    result = descriptive_log2_fold_changes(selected)
    assert result.columns.tolist() == [
        "6m_F",
        "6m_M",
        "mean_log2_cpm_difference",
        "max_absolute_difference",
    ]
    assert result["max_absolute_difference"].is_monotonic_decreasing

    with pytest.raises(ValueError, match="matched"):
        descriptive_log2_fold_changes(selected.filter(like="RAP"))

    zero_library = selected.copy()
    zero_library.iloc[:, 0] = 0
    with pytest.raises(ValueError, match="positive library"):
        descriptive_log2_fold_changes(zero_library)


def test_download_counts_uses_cache_and_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = gzip.compress(b'"GENE_ID"\\t"RAP_6m_F_1"\\n"gene-a"\\t1\\n')
    calls = 0

    def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
        nonlocal calls
        assert url == gse131754.GSE131754_URL
        assert timeout == 60
        calls += 1
        return io.BytesIO(payload)

    monkeypatch.setattr(gse131754, "urlopen", fake_urlopen)
    destination = tmp_path / "counts.tsv.gz"
    assert download_counts(destination) == destination
    assert destination.read_bytes() == payload
    assert download_counts(destination) == destination
    assert calls == 1
