"""Adapter for public longitudinal Dog Aging Project blood chemistry data."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from rejuvenationkit.schemas import Modality, Observation, Study, Subject

DAP_DATASET_DOI: Final = "https://doi.org/10.7910/DVN/8C63KB"
DAP_ARCHIVE_URL: Final = "https://dataverse.harvard.edu/api/access/datafile/13986068"
DAP_CHEMISTRY_MEMBER: Final = (
    "baseline_phenotype_datasets/DAP_2024_SamplesResults_ChemistryPanel_analyzed.csv"
)
_NORMALIZED_FIRST_VISIT = datetime(2000, 1, 1, tzinfo=UTC)
_VISIT_PATTERN = "precision_"


def download_archive(destination: Path, *, overwrite: bool = False) -> Path:
    """Download the public phenotype archive, reusing a local cache by default."""
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urlopen(DAP_ARCHIVE_URL, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_chemistry(path: Path) -> pd.DataFrame:
    """Read and validate the canine chemistry table inside the public archive."""
    with ZipFile(path) as archive:
        if DAP_CHEMISTRY_MEMBER not in archive.namelist():
            raise ValueError(f"archive does not contain {DAP_CHEMISTRY_MEMBER!r}")
        with archive.open(DAP_CHEMISTRY_MEMBER) as source:
            chemistry = pd.read_csv(source)

    required = {"dog_id", "Sample_Year"}
    missing = required.difference(chemistry.columns)
    if missing:
        raise ValueError(f"chemistry table is missing columns: {sorted(missing)}")
    if chemistry.empty:
        raise ValueError("chemistry table is empty")
    if chemistry.duplicated(["dog_id", "Sample_Year"]).any():
        raise ValueError("dog_id and Sample_Year pairs must be unique")
    if chemistry["dog_id"].isna().any() or chemistry["Sample_Year"].isna().any():
        raise ValueError("dog_id and Sample_Year cannot be missing")
    for visit in chemistry["Sample_Year"].astype(str):
        parse_visit_number(visit)
    return chemistry


def parse_visit_number(visit: str) -> int:
    """Return the positive wave number encoded by ``precision_N``."""
    if not visit.startswith(_VISIT_PATTERN):
        raise ValueError(f"unrecognized Dog Aging Project visit: {visit!r}")
    suffix = visit.removeprefix(_VISIT_PATTERN)
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError(f"unrecognized Dog Aging Project visit: {visit!r}")
    return int(suffix)


def select_visits(
    chemistry: pd.DataFrame,
    *,
    visits: tuple[str, ...] = ("precision_1", "precision_2"),
) -> pd.DataFrame:
    """Select requested visit waves while retaining dogs missing later waves."""
    if not visits:
        raise ValueError("at least one visit is required")
    for visit in visits:
        parse_visit_number(visit)
    selected = chemistry.loc[chemistry["Sample_Year"].isin(visits)].copy()
    if selected.empty:
        raise ValueError("no requested Dog Aging Project visits were found")
    return selected


def build_study(
    chemistry: pd.DataFrame,
    *,
    features: tuple[str, ...],
    source_uri: str = DAP_DATASET_DOI,
) -> Study:
    """Convert public canine chemistry waves into a typed longitudinal study.

    Visit-wave timestamps are normalized because the released table provides
    ``precision_N`` waves rather than actual specimen collection dates.
    """
    missing_features = set(features).difference(chemistry.columns)
    if missing_features:
        raise ValueError(f"requested features are absent: {sorted(missing_features)}")

    dog_ids = tuple(sorted({str(value) for value in chemistry["dog_id"]}))
    subjects = tuple(
        Subject(
            subject_id=dog_id,
            cohort="dog_aging_project_precision",
            anchors={"normalized_precision_1": _NORMALIZED_FIRST_VISIT},
            attributes={"species": "Canis lupus familiaris", "source": "Dog Aging Project"},
        )
        for dog_id in dog_ids
    )
    observations: list[Observation] = []
    for _, row in chemistry.iterrows():
        dog_id = str(row["dog_id"])
        visit = str(row["Sample_Year"])
        visit_number = parse_visit_number(visit)
        timestamp = _NORMALIZED_FIRST_VISIT + timedelta(days=365 * (visit_number - 1))
        for feature in features:
            value = row[feature]
            if pd.isna(value):
                continue
            observations.append(
                Observation(
                    subject_id=dog_id,
                    timestamp=timestamp,
                    modality=Modality.CLINICAL,
                    feature=feature,
                    value=float(cast(float, value)),
                    unit="reported_value",
                    replicate_id=visit,
                    source_uri=source_uri,
                )
            )
    return Study(
        study_id="dog-aging-project-longitudinal-chemistry",
        subjects=subjects,
        observations=tuple(observations),
        metadata={
            "dataset_doi": DAP_DATASET_DOI,
            "organism": "Canis lupus familiaris",
            "design": "observational longitudinal cohort",
            "timeline": "normalized visit waves; not calendar collection dates",
        },
    )


def paired_changes(
    chemistry: pd.DataFrame,
    *,
    features: tuple[str, ...],
    baseline: str = "precision_1",
    follow_up: str = "precision_2",
) -> pd.DataFrame:
    """Calculate descriptive within-dog changes between two visit waves."""
    selected = chemistry.loc[chemistry["Sample_Year"].isin((baseline, follow_up))]
    rows: list[dict[str, str | int | float]] = []
    for feature in features:
        wide = selected.pivot(index="dog_id", columns="Sample_Year", values=feature)
        if baseline not in wide or follow_up not in wide:
            continue
        paired = wide[[baseline, follow_up]].dropna()
        changes = paired[follow_up] - paired[baseline]
        rows.append(
            {
                "feature": feature,
                "paired_dogs": len(paired),
                "mean_change": float(changes.mean()),
                "median_change": float(changes.median()),
            }
        )
    return pd.DataFrame(rows).set_index("feature")
