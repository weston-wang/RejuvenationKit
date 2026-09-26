"""Reproducible adapter for the public GSE131754 intervention dataset."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from rejuvenationkit.evidence import EffectDirection
from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
)
from rejuvenationkit.genomics.signatures import (
    GeneSignature,
    SignatureFeature,
)
from rejuvenationkit.schemas import Modality, Observation, Study, Subject

GSE131754_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131754/suppl/"
    "GSE131754_Interventions_assigned_reads.txt.gz"
)
GSE131754_SHA256 = "dbd9c37015a17729fc800dfda537b58f64e694af60e4aaeaece748d9d97c7e90"
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


def download_counts(
    destination: Path,
    *,
    overwrite: bool = False,
    expected_sha256: str = GSE131754_SHA256,
) -> Path:
    """Download the pinned count matrix and verify cached or new bytes."""
    if destination.exists() and not overwrite:
        _verify_sha256(destination, expected_sha256)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urlopen(GSE131754_URL, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        _verify_sha256(temporary, expected_sha256)
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


INTERVENTION_CONTROLS: dict[str, str] = {
    "ACA": "CON",
    "CR": "CON",
    "EST": "CON",
    "PROT": "CON",
    "RAP": "CON",
    "GHRKO": "GHRCON",
    "SNELL": "SNELLCON",
    "MR": "MRCON",
}
"""Each GSE131754 intervention code mapped to its matched control group.

ACA (acarbose), CR (caloric restriction), EST (17-alpha-estradiol), PROT
(Protandim), and RAP (rapamycin) share the ``CON`` animals in each age/sex
stratum. The three genetic or dietary models have their own controls.
"""


def filtered_log2_cpm(
    counts: pd.DataFrame,
    *,
    minimum_cpm: float = 1.0,
    minimum_sample_fraction: float = 0.5,
) -> pd.DataFrame:
    """Return log2(CPM + 1) for genes expressed in enough samples.

    A gene is kept when its CPM exceeds ``minimum_cpm`` in at least
    ``minimum_sample_fraction`` of samples. Filtering uses only expression
    levels, never group labels, so it cannot select genes by treatment effect.
    """
    if not 0 < minimum_sample_fraction <= 1:
        raise ValueError("minimum_sample_fraction must lie in (0, 1]")
    library_sizes = counts.sum(axis=0)
    if (library_sizes <= 0).any():
        raise ValueError("every sample must have a positive library size")
    cpm = counts.divide(library_sizes, axis=1) * 1_000_000
    keep = (cpm > minimum_cpm).mean(axis=1) >= minimum_sample_fraction
    if not keep.any():
        raise ValueError("no genes pass the expression filter")
    return cast(pd.DataFrame, np.log2(cpm.loc[keep] + 1))


def intervention_sample_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Return a sample table with ``group`` and ``stratum`` columns.

    ``group`` is the intervention or control code and ``stratum`` is age and sex,
    for example ``"6m-F"``. The index matches the count-matrix columns.
    """
    rows = {}
    for column in counts.columns:
        metadata = parse_sample_name(str(column))
        rows[str(column)] = {
            "group": metadata.intervention_code,
            "stratum": f"{metadata.age_months}m-{metadata.sex}",
        }
    return pd.DataFrame.from_dict(rows, orient="index").loc[[str(item) for item in counts.columns]]


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


def build_genomic_matrix(
    counts: pd.DataFrame,
    *,
    source_uri: str = GSE131754_URL,
    source_checksum: str | None = None,
) -> GenomicMatrix:
    """Convert the complete gene-by-sample counts into a typed genomic matrix."""
    if counts.empty:
        raise ValueError("count matrix is empty")
    if not counts.index.is_unique or counts.columns.duplicated().any():
        raise ValueError("gene and sample identifiers must be unique")
    metadata = tuple(parse_sample_name(str(column)) for column in counts.columns)
    samples = tuple(
        GenomicSample(
            sample_id=item.sample_id,
            subject_id=item.sample_id,
            tissue="liver",
            species_taxon_id=10090,
            cohort=(
                "rapamycin"
                if item.is_rapamycin
                else "control"
                if item.is_control
                else item.intervention_code.lower()
            ),
            assay_id="polyA RNA-seq assigned reads",
            attributes={
                "age_months": item.age_months,
                "sex": item.sex,
                "replicate": item.replicate,
                "intervention_code": item.intervention_code,
            },
        )
        for item in metadata
    )
    features = tuple(
        GenomicFeature(
            feature_id=str(identifier),
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
            genome_assembly="GRCm38",
        )
        for identifier in counts.index
    )
    values = counts.to_numpy(dtype=float).T
    return GenomicMatrix(
        values=values,
        samples=samples,
        features=features,
        scale=MatrixScale.RAW_COUNTS,
        provenance=GenomicMatrixProvenance(
            source_id="GSE131754:Interventions_assigned_reads",
            source_checksum=source_checksum,
            preprocessing=(
                "STAR 2.5.2b alignment to mm10/GRCm38",
                "featureCounts 1.5 assigned reads",
            ),
            software_versions={
                "STAR": "2.5.2b",
                "featureCounts": "1.5",
            },
            reference_resource_ids=(source_uri,),
        ),
    )


def rapamycin_mechanism_signatures() -> tuple[GeneSignature, ...]:
    """Return fixed mouse-liver mechanism panels for engineering validation.

    These small, inspectable panels use canonical genes from biological themes
    reported with GSE131754: mTOR/lipid metabolism, autophagy, NRF2 response,
    and immune signaling. They are prespecified software fixtures, not validated
    biological-age clocks or a reproduction of the publication's full models.
    All weights are oriented so a negative score is the hypothesized favorable
    rapamycin direction.
    """

    def build_signature(
        signature_id: str,
        name: str,
        identifiers: tuple[str, ...],
        *,
        weight: float,
    ) -> GeneSignature:
        return GeneSignature(
            signature_id=signature_id,
            version="1.0",
            name=name,
            features=tuple(
                SignatureFeature(feature_id=identifier, weight=weight) for identifier in identifiers
            ),
            namespace=FeatureNamespace.ENSEMBL,
            species_taxon_id=10090,
            tissue="liver",
            target_name=f"{signature_id}_response",
            target_unit="mean_signed_log2_cpm",
            direction=EffectDirection.LOWER_IS_BETTER,
            resource_id=(
                "RejuvenationKit:GSE131754-canonical-mechanism-panels-v1;"
                "embedded-Ensembl-GRCm38-ID-map-v1"
            ),
            allowed_scales=(MatrixScale.LOG_CPM,),
        )

    return (
        build_signature(
            "mtorc1-lipogenesis",
            "mTORC1 and hepatic lipogenesis",
            (
                "ENSMUSG00000028991",  # Mtor
                "ENSMUSG00000025583",  # Rptor
                "ENSMUSG00000020538",  # Srebf1
                "ENSMUSG00000025153",  # Fasn
                "ENSMUSG00000020532",  # Acaca
                "ENSMUSG00000037071",  # Scd1
            ),
            weight=1,
        ),
        build_signature(
            "autophagy-lysosome",
            "Autophagy and lysosomal program",
            (
                "ENSMUSG00000023990",  # Tfeb
                "ENSMUSG00000038160",  # Atg5
                "ENSMUSG00000030314",  # Atg7
                "ENSMUSG00000035086",  # Becn1
                "ENSMUSG00000031812",  # Map1lc3b
                "ENSMUSG00000015837",  # Sqstm1
            ),
            weight=-1,
        ),
        build_signature(
            "nrf2-cytoprotection",
            "NRF2 cytoprotective response",
            (
                "ENSMUSG00000015839",  # Nfe2l2
                "ENSMUSG00000003849",  # Nqo1
                "ENSMUSG00000032350",  # Gclc
                "ENSMUSG00000028124",  # Gclm
                "ENSMUSG00000005413",  # Hmox1
            ),
            weight=-1,
        ),
        build_signature(
            "inflammatory-response",
            "Inflammatory and interferon response (putative safety signal)",
            (
                "ENSMUSG00000028163",  # Nfkb1
                "ENSMUSG00000026104",  # Stat1
                "ENSMUSG00000025746",  # Il6
                "ENSMUSG00000024401",  # Tnf
                "ENSMUSG00000035385",  # Ccl2
                "ENSMUSG00000034855",  # Cxcl10
            ),
            weight=1,
        ),
    )


def _verify_sha256(path: Path, expected: str) -> None:
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"GSE131754 checksum mismatch for {path}: expected {expected}, observed {observed}"
        )
