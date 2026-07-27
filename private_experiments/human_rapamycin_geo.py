"""Adapters and cross-context pathway validation for human rapamycin GEO data."""

from __future__ import annotations

import csv
import gzip
import io
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

SERIES = {
    "GSE62375": ("GSE62nnn", "GPL10558"),
    "GSE124020": ("GSE124nnn", "GPL17586"),
    "GSE154401": ("GSE154nnn", "GPL6480"),
}
HTA_ANNOTATION_URL = (
    "https://bioconductor.org/packages/release/data/annotation/src/contrib/"
    "hta20transcriptcluster.db_8.8.0.tar.gz"
)
HUMAN_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)

PATHWAYS = {
    "mTORC1": {"MTOR", "RPTOR", "RPS6KB1", "RPS6KB2", "EIF4EBP1", "TSC1", "TSC2"},
    "mTORC2": {"RICTOR", "MAPKAP1", "AKT1", "AKT2", "SGK1", "FOXO3"},
    "autophagy": {"ULK1", "ATG5", "ATG7", "BECN1", "MAP1LC3B", "SQSTM1", "TFEB"},
    "inflammaging": {"NFKB1", "RELA", "IL1B", "IL6", "CCL2", "CXCL10", "TNF", "STAT1"},
    "senescence": {"CDKN2A", "CDKN1A", "TP53", "LMNB1", "SERPINE1", "MMP3"},
    "mitochondrial": {"PPARGC1A", "TFAM", "NRF1", "SOD2", "NDUFS1", "COX4I1"},
    "proteostasis": {"HSPA1A", "HSP90AA1", "HSPA5", "XBP1", "ATF4", "DDIT3"},
}


@dataclass(frozen=True)
class GeoContrast:
    """One treatment-control gene-expression contrast."""

    accession: str
    context: str
    treated_samples: tuple[str, ...]
    control_samples: tuple[str, ...]
    gene_effect: pd.Series


def download_geo_files(cache: Path) -> dict[str, tuple[Path, Path | None]]:
    """Download series matrices and available GEO platform annotations."""
    cache.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Path, Path | None]] = {}
    for accession, (series_stem, platform) in SERIES.items():
        matrix = cache / f"{accession}_series_matrix.txt.gz"
        _download(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/"
            f"{series_stem}/{accession}/matrix/{accession}_series_matrix.txt.gz",
            matrix,
        )
        platform_stem = f"{platform[:-3]}nnn"
        annotation_path = cache / f"{platform}.annot.gz"
        annotation: Path | None = annotation_path
        try:
            _download(
                "https://ftp.ncbi.nlm.nih.gov/geo/platforms/"
                f"{platform_stem}/{platform}/annot/{platform}.annot.gz",
                annotation_path,
            )
        except OSError:
            if platform == "GPL17586":
                _build_hta_annotation(cache, annotation_path)
            else:
                annotation = None
        files[accession] = (matrix, annotation)
    return files


def load_contrasts(cache: Path) -> tuple[tuple[GeoContrast, ...], tuple[str, ...]]:
    """Load all gene-mappable treatment-control contrasts."""
    files = download_geo_files(cache)
    contrasts: list[GeoContrast] = []
    unsupported: list[str] = []
    for accession, (matrix_path, annotation_path) in files.items():
        if annotation_path is None:
            unsupported.append(
                f"{accession}: {SERIES[accession][1]} has no compact GEO gene annotation"
            )
            continue
        titles, expression = _read_series_matrix(matrix_path)
        mapping = _read_annotation(annotation_path)
        gene_expression = _collapse_to_genes(expression, mapping)
        contrasts.extend(_accession_contrasts(accession, titles, gene_expression))
    return tuple(contrasts), tuple(unsupported)


def pathway_scores(contrast: GeoContrast) -> pd.Series:
    """Return robust mean standardized effects for predefined aging pathways."""
    centered = contrast.gene_effect - contrast.gene_effect.median()
    scale = float(1.4826 * np.median(np.abs(centered)))
    standardized = centered / max(scale, 1e-8)
    scores = {}
    for pathway, genes in PATHWAYS.items():
        available = standardized.index.intersection(sorted(genes))
        scores[pathway] = (
            float(standardized.loc[available].mean()) if len(available) >= 2 else np.nan
        )
    return pd.Series(scores, name=f"{contrast.accession}:{contrast.context}")


def contrast_concordance(contrasts: tuple[GeoContrast, ...]) -> pd.DataFrame:
    """Calculate pairwise Spearman agreement across shared measured genes."""
    effects = pd.concat(
        {f"{item.accession}:{item.context}": item.gene_effect for item in contrasts},
        axis=1,
    )
    return effects.corr(method="spearman", min_periods=100)


def _accession_contrasts(
    accession: str,
    titles: pd.Series,
    expression: pd.DataFrame,
) -> list[GeoContrast]:
    specifications = {
        "GSE62375": (
            ("older_50nM", ("70 year old", "50nM rapamycin"), ("70 year old", "Control_37")),
            ("older_100nM", ("70 year old", "100nM rapamycin"), ("70 year old", "Control_37")),
            ("younger_50nM", ("30 year old", "50nM rapamycin"), ("30 year old", "Control_37")),
            ("younger_100nM", ("30 year old", "100nM rapamycin"), ("30 year old", "Control_37")),
        ),
        "GSE124020": (
            (
                "macrophage_IFNg",
                ("M(IFN-\u03b3) + Rapamycin",),
                ("M(IFN-\u03b3),",),
            ),
            ("macrophage_IL10", ("IL-10) + Rapamycin",), ("IL-10),",)),
        ),
        "GSE154401": (
            ("Tconv", ("Tconv_rapamycin",), ("Tconv_anti-CD3CD28",)),
            ("Treg", ("Treg_rapamycin",), ("Treg_anti-CD3CD28",)),
        ),
    }
    results = []
    for context, treated_terms, control_terms in specifications[accession]:
        treated = _matching_samples(titles, treated_terms)
        control = _matching_samples(titles, control_terms)
        if not treated or not control:
            raise ValueError(f"{accession} {context} sample selection was empty")
        effect = expression.loc[:, list(treated)].mean(axis=1) - expression.loc[
            :, list(control)
        ].mean(axis=1)
        results.append(
            GeoContrast(
                accession=accession,
                context=context,
                treated_samples=treated,
                control_samples=control,
                gene_effect=effect.sort_index(),
            )
        )
    return results


def _read_series_matrix(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        lines = source.readlines()
    title_line = next(line for line in lines if line.startswith("!Sample_title"))
    titles = next(csv.reader([title_line], delimiter="\t"))[1:]
    start = next(index for index, line in enumerate(lines) if line.startswith('"ID_REF"'))
    end = next(
        index
        for index, line in enumerate(lines[start:], start)
        if line.startswith("!series_matrix")
    )
    expression = pd.read_csv(
        io.StringIO("".join(lines[start:end])),
        sep="\t",
        index_col=0,
    )
    sample_titles = pd.Series(titles, index=expression.columns, dtype="string")
    return sample_titles, expression.apply(pd.to_numeric)


def _read_annotation(path: Path) -> pd.Series:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        lines = source.readlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("ID\t"))
    table = pd.read_csv(io.StringIO("".join(lines[start:])), sep="\t", low_memory=False)
    symbols = table.set_index("ID")["Gene symbol"].astype("string").str.split(" /// ").str[0]
    return symbols.loc[symbols.notna() & symbols.ne("") & symbols.ne("---")]


def _collapse_to_genes(expression: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    shared = expression.index.intersection(mapping.index)
    annotated = expression.loc[shared].copy()
    annotated["gene_symbol"] = mapping.loc[shared].to_numpy()
    return annotated.groupby("gene_symbol", sort=True).mean(numeric_only=True)


def _matching_samples(titles: pd.Series, terms: tuple[str, ...]) -> tuple[str, ...]:
    mask = pd.Series(True, index=titles.index)
    for term in terms:
        mask &= titles.str.contains(term, regex=False)
    return tuple(titles.index[mask])


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise OSError(f"unable to download {url}") from error


def _build_hta_annotation(cache: Path, destination: Path) -> None:
    archive = cache / "hta20transcriptcluster.db_8.8.0.tar.gz"
    gene_info_path = cache / "Homo_sapiens.gene_info.gz"
    database = cache / "hta20transcriptcluster.sqlite"
    _download(HTA_ANNOTATION_URL, archive)
    _download(HUMAN_GENE_INFO_URL, gene_info_path)
    if not database.exists():
        member_name = "hta20transcriptcluster.db/inst/extdata/hta20transcriptcluster.sqlite"
        with tarfile.open(archive, "r:gz") as package:
            member = package.getmember(member_name)
            source = package.extractfile(member)
            if source is None:
                raise ValueError("Bioconductor package lacks its annotation database")
            with database.open("wb") as output:
                shutil.copyfileobj(source, output)
    with sqlite3.connect(database) as connection:
        probes = pd.read_sql_query(
            "SELECT probe_id, gene_id FROM probes "
            "WHERE gene_id IS NOT NULL AND gene_id != '' AND is_multiple = 0",
            connection,
            dtype={"gene_id": "string"},
        )
    gene_info = pd.read_csv(
        gene_info_path,
        sep="\t",
        compression="gzip",
        dtype={"GeneID": "string", "Symbol": "string"},
    )
    gene_info.columns = [column.removeprefix("#") for column in gene_info.columns]
    symbols = probes.merge(
        gene_info.loc[:, ["GeneID", "Symbol"]],
        left_on="gene_id",
        right_on="GeneID",
        how="inner",
    ).drop_duplicates("probe_id")
    with gzip.open(destination, "wt", encoding="utf-8") as output:
        output.write("ID\tGene symbol\n")
        symbols.loc[:, ["probe_id", "Symbol"]].to_csv(
            output,
            sep="\t",
            header=False,
            index=False,
        )
