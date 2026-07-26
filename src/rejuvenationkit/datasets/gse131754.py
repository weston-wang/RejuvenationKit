"""Reproducible adapter for the public GSE131754 intervention dataset."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from rejuvenationkit.schemas import Modality, Observation, Study, Subject

GSE131754_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131754/suppl/"
    "GSE131754_Interventions_assigned_reads.txt.gz"
)
_SAMPLE_PATTERN = re.compile(
    r"^(?P<intervention>[A-Z0-9]+)_(?P<age_months>\d+)m_(?P<sex>[FM])_(?P<replicate>\d+)$"
)
_NORMALIZED_BIRTH = datetime(2000, 1, 1, tzinfo=UTC)
_DAYS_PER_MONTH = 365.2425 / 12


class SampleMetadata(BaseModel):
    """Metadata encoded in a GSE131754 count-matrix column name."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    intervention_code: str = Field(min_length=1)
    age_months: int = Field(gt=0)
    sex: str = Field(pattern="^[FM]$")
    replicate: int = Field(gt=0)

    @property
    def is_rapamycin(self) -> bool:
        """Return whether this is a rapamycin-treated sample."""
        return self.intervention_code == "RAP"

    @property
    def is_control(self) -> bool:
        """Return whether this is an untreated matched-control sample."""
        return self.intervention_code == "CON"


def parse_sample_name(sample_id: str) -> SampleMetadata:
    """Parse intervention, age, sex, and replicate from a sample name."""
    match = _SAMPLE_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"unrecognized GSE131754 sample name: {sample_id!r}")
    values = match.groupdict()
    return SampleMetadata(
        sample_id=sample_id,
        intervention_code=values["intervention"],
        age_months=int(values["age_months"]),
        sex=values["sex"],
        replicate=int(values["replicate"]),
    )


def download_counts(destination: Path, *, overwrite: bool = False) -> Path:
    """Download the processed count matrix, reusing a local cache by default."""
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urlopen(GSE131754_URL, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_counts(path: Path) -> pd.DataFrame:
    """Read and validate the public gene-by-sample count matrix."""
    counts = pd.read_csv(path, sep="\t", compression="infer", index_col="GENE_ID")
    if counts.empty:
        raise ValueError("GSE131754 count matrix is empty")
    if not counts.index.is_unique:
        raise ValueError("GSE131754 gene identifiers must be unique")
    if counts.columns.duplicated().any():
        raise ValueError("GSE131754 sample identifiers must be unique")
    for sample_id in counts.columns:
        parse_sample_name(str(sample_id))
    numeric = cast(pd.DataFrame, counts.apply(pd.to_numeric, errors="raise"))
    if numeric.isna().any().any():
        raise ValueError("GSE131754 count matrix contains missing values")
    if (numeric < 0).any().any():
        raise ValueError("GSE131754 count matrix contains negative counts")
    return numeric


def select_rapamycin_and_controls(
    counts: pd.DataFrame,
    *,
    ages_months: tuple[int, ...] = (6, 12),
) -> pd.DataFrame:
    """Select rapamycin and age/sex-matched control columns."""
    selected = [
        column
        for column in counts.columns
        if (
            (metadata := parse_sample_name(str(column))).age_months in ages_months
            and (metadata.is_rapamycin or metadata.is_control)
        )
    ]
    if not selected:
        raise ValueError("no matching rapamycin or control samples were found")
    return counts.loc[:, selected].copy()


def descriptive_log2_fold_changes(counts: pd.DataFrame) -> pd.DataFrame:
    """Estimate equal-weighted RAP-minus-control log2 CPM differences.

    This is an exploratory summary stratified by age and sex, not a
    count-dispersion model or statistical significance test.
    """
    library_sizes = counts.sum(axis=0)
    if (library_sizes <= 0).any():
        raise ValueError("every sample must have a positive library size")
    log_cpm = np.log2(counts.divide(library_sizes, axis=1) * 1_000_000 + 1)
    metadata = {str(column): parse_sample_name(str(column)) for column in counts.columns}
    contrasts: list[pd.Series] = []
    labels: list[str] = []
    for age_months in sorted({item.age_months for item in metadata.values()}):
        for sex in ("F", "M"):
            rapamycin = [
                sample_id
                for sample_id, item in metadata.items()
                if item.age_months == age_months and item.sex == sex and item.is_rapamycin
            ]
            controls = [
                sample_id
                for sample_id, item in metadata.items()
                if item.age_months == age_months and item.sex == sex and item.is_control
            ]
            if not rapamycin or not controls:
                continue
            contrasts.append(log_cpm[rapamycin].mean(axis=1) - log_cpm[controls].mean(axis=1))
            labels.append(f"{age_months}m_{sex}")
    if not contrasts:
        raise ValueError("no matched age/sex contrasts are available")
    result = pd.concat(contrasts, axis=1)
    result.columns = labels
    result["mean_log2_cpm_difference"] = result.mean(axis=1)
    result["max_absolute_difference"] = result[labels].abs().max(axis=1)
    return result.sort_values("max_absolute_difference", ascending=False)


def build_study(
    counts: pd.DataFrame,
    *,
    gene_ids: tuple[str, ...],
    source_uri: str = GSE131754_URL,
) -> Study:
    """Convert selected counts into a typed cross-sectional Study.

    Timestamps use a normalized study timeline derived from reported age rather
    than invented calendar collection dates.
    """
    missing_genes = set(gene_ids).difference(str(index) for index in counts.index)
    if missing_genes:
        raise ValueError(f"requested genes are absent: {sorted(missing_genes)}")
    metadata = [parse_sample_name(str(column)) for column in counts.columns]
    subjects = tuple(
        Subject(
            subject_id=item.sample_id,
            cohort=f"{item.intervention_code.lower()}_{item.age_months}m",
            interventions=("rapamycin",) if item.is_rapamycin else (),
            anchors={"normalized_birth": _NORMALIZED_BIRTH},
            attributes={
                "sex": item.sex,
                "age_months": item.age_months,
                "replicate": item.replicate,
                "geo_accession": "GSE131754",
            },
        )
        for item in metadata
    )
    observations = tuple(
        Observation(
            subject_id=item.sample_id,
            timestamp=_NORMALIZED_BIRTH + timedelta(days=item.age_months * _DAYS_PER_MONTH),
            modality=Modality.TRANSCRIPTOMICS,
            feature=gene_id,
            value=float(cast(int | float, counts.at[gene_id, item.sample_id])),
            unit="assigned_reads",
            source_uri=source_uri,
        )
        for item in metadata
        for gene_id in gene_ids
    )
    return Study(
        study_id="GSE131754-rapamycin-controls",
        subjects=subjects,
        observations=observations,
        metadata={
            "geo_accession": "GSE131754",
            "organism": "Mus musculus",
            "tissue": "liver",
            "design": "cross-sectional",
        },
    )
