from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from rejuvenationkit.datasets.dog_aging_project import (
    DAP_CHEMISTRY_MEMBER,
    build_study,
    paired_changes,
    parse_visit_number,
    read_chemistry,
    select_visits,
)


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "dap.zip"
    data = (
        "dog_id,Sample_Year,krt_cp_albumin_value,krt_cp_creatinine_value\n"
        "10,precision_1,3.2,0.8\n"
        "10,precision_2,3.4,0.9\n"
        "20,precision_1,3.1,0.7\n"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr(DAP_CHEMISTRY_MEMBER, data)
    return path


def test_read_select_and_build_longitudinal_dog_study(tmp_path: Path) -> None:
    chemistry = read_chemistry(_archive(tmp_path))
    selected = select_visits(chemistry)
    study = build_study(selected, features=("krt_cp_albumin_value",))

    assert len(study.subjects) == 2
    assert len(study.observations) == 3
    assert study.metadata["organism"] == "Canis lupus familiaris"
    first, second = study.observations[:2]
    assert second.timestamp - first.timestamp == timedelta(days=365)
    assert second.replicate_id == "precision_2"


def test_paired_changes_excludes_dogs_without_follow_up(tmp_path: Path) -> None:
    chemistry = read_chemistry(_archive(tmp_path))
    changes = paired_changes(chemistry, features=("krt_cp_albumin_value",))

    assert changes.at["krt_cp_albumin_value", "paired_dogs"] == 1
    assert changes.at["krt_cp_albumin_value", "mean_change"] == pytest.approx(0.2)


@pytest.mark.parametrize("visit", ["baseline", "precision_0", "precision_x"])
def test_parse_visit_number_rejects_invalid_labels(visit: str) -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        parse_visit_number(visit)


def test_read_chemistry_rejects_duplicate_dog_visits(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.zip"
    frame = pd.DataFrame({"dog_id": [1, 1], "Sample_Year": ["precision_1", "precision_1"]})
    with ZipFile(path, "w") as archive:
        archive.writestr(DAP_CHEMISTRY_MEMBER, frame.to_csv(index=False))

    with pytest.raises(ValueError, match="must be unique"):
        read_chemistry(path)
