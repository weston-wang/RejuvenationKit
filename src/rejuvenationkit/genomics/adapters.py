"""Adapters from common genome-scale containers into canonical matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
    MatrixValues,
)


class AnnDataLike(Protocol):
    """Structural subset of AnnData used by the dependency-free adapter."""

    X: Any
    obs: pd.DataFrame
    var: pd.DataFrame
    obs_names: Sequence[str]
    var_names: Sequence[str]
    layers: Mapping[str, Any]


def from_expression_frame(
    frame: pd.DataFrame,
    *,
    samples: tuple[GenomicSample, ...],
    namespace: FeatureNamespace,
    scale: MatrixScale,
    provenance: GenomicMatrixProvenance,
    feature_type: GenomicFeatureType = GenomicFeatureType.GENE,
    genome_assembly: str | None = None,
) -> GenomicMatrix:
    """Convert a samples-by-features DataFrame into a validated matrix."""
    if tuple(str(value) for value in frame.index) != tuple(item.sample_id for item in samples):
        raise ValueError("DataFrame row order must exactly match sample metadata")
    if frame.columns.duplicated().any():
        raise ValueError("DataFrame feature identifiers must be unique")
    numeric = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    features = tuple(
        GenomicFeature(
            feature_id=str(identifier),
            feature_type=feature_type,
            namespace=namespace,
            genome_assembly=genome_assembly,
        )
        for identifier in frame.columns
    )
    return GenomicMatrix(
        values=numeric,
        samples=samples,
        features=features,
        scale=scale,
        provenance=provenance,
    )


def from_methylation_beta(
    frame: pd.DataFrame,
    *,
    samples: tuple[GenomicSample, ...],
    provenance: GenomicMatrixProvenance,
    genome_assembly: str,
    namespace: FeatureNamespace = FeatureNamespace.ILLUMINA_PROBE,
) -> GenomicMatrix:
    """Validate normalized samples-by-CpGs beta values on the closed unit interval."""
    return from_expression_frame(
        frame,
        samples=samples,
        namespace=namespace,
        scale=MatrixScale.METHYLATION_BETA,
        provenance=provenance,
        feature_type=GenomicFeatureType.CPG,
        genome_assembly=genome_assembly,
    )


def beta_to_m_values(
    matrix: GenomicMatrix,
    *,
    epsilon: float = 1e-6,
    maximum_output_cells: int = 100_000_000,
) -> GenomicMatrix:
    """Convert methylation beta values to log2 M-values with explicit clipping."""
    if matrix.scale is not MatrixScale.METHYLATION_BETA:
        raise ValueError("beta_to_m_values requires a methylation_beta matrix")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie between zero and 0.5")
    values = matrix.dense_values(maximum_cells=maximum_output_cells)
    clipped = np.clip(values, epsilon, 1 - epsilon)
    transformed = np.log2(clipped / (1 - clipped))
    provenance = matrix.provenance.model_copy(
        update={"preprocessing": (*matrix.provenance.preprocessing, f"beta_to_m:{epsilon:g}")}
    )
    return GenomicMatrix(
        values=transformed,
        samples=matrix.samples,
        features=matrix.features,
        scale=MatrixScale.METHYLATION_M,
        provenance=provenance,
    )


def normalize_counts_log_cpm(
    matrix: GenomicMatrix,
    *,
    feature_ids: tuple[str, ...] | None = None,
    pseudocount: float = 1.0,
    maximum_output_cells: int = 10_000_000,
) -> GenomicMatrix:
    """Library-size normalize raw counts and return selected log2-CPM features.

    Library sizes always use every matrix feature. ``feature_ids`` limits only
    the dense output, which allows pathway scoring from large sparse matrices
    without materializing every zero-valued gene.
    """
    if matrix.scale is not MatrixScale.RAW_COUNTS:
        raise ValueError("normalize_counts_log_cpm requires a raw_counts matrix")
    if pseudocount <= 0 or not np.isfinite(pseudocount):
        raise ValueError("pseudocount must be positive and finite")
    library_sizes: npt.NDArray[np.float64]
    if isinstance(matrix.values, sparse.csr_matrix):
        library_sizes = np.asarray(matrix.values.sum(axis=1), dtype=float).ravel()
    else:
        if np.isnan(matrix.values).any():
            raise ValueError("raw count normalization does not accept missing values")
        library_sizes = np.asarray(matrix.values.sum(axis=1), dtype=float)
    if np.any(library_sizes <= 0):
        raise ValueError("every sample must have a positive library size")
    selected = matrix.subset_features(feature_ids) if feature_ids is not None else matrix
    cells = selected.shape[0] * selected.shape[1]
    if cells > maximum_output_cells:
        raise ValueError(
            f"log-CPM output would contain {cells:,} cells; "
            f"maximum_output_cells={maximum_output_cells:,}"
        )
    counts = selected.dense_values(maximum_cells=maximum_output_cells)
    normalized = np.log2(counts / library_sizes[:, None] * 1_000_000 + pseudocount)
    provenance = matrix.provenance.model_copy(
        update={
            "preprocessing": (
                *matrix.provenance.preprocessing,
                f"library_size_log2_cpm:pseudocount={pseudocount:g}",
            )
        }
    )
    return GenomicMatrix(
        values=normalized,
        samples=matrix.samples,
        features=selected.features,
        scale=MatrixScale.LOG_CPM,
        provenance=provenance,
    )


def from_anndata(
    adata: AnnDataLike,
    *,
    layer: str | None,
    scale: MatrixScale,
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
    provenance: GenomicMatrixProvenance,
    tissue_column: str,
    species_taxon_id_column: str,
    subject_id_column: str,
    cohort_column: str | None = None,
    assay_id_column: str | None = None,
    batch_id_column: str | None = None,
    genome_assembly: str | None = None,
) -> GenomicMatrix:
    """Adapt an AnnData-like object without importing or requiring AnnData."""
    if layer is None:
        raw_values = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"AnnData layer is absent: {layer}")
        raw_values = adata.layers[layer]
    obs_names = tuple(str(value) for value in adata.obs_names)
    var_names = tuple(str(value) for value in adata.var_names)
    if tuple(str(value) for value in adata.var.index) != var_names:
        raise ValueError("AnnData var index must align with var_names")
    if tuple(str(value) for value in adata.obs.index) != obs_names:
        raise ValueError("AnnData obs index must align with obs_names")
    required_columns = {tissue_column, species_taxon_id_column, subject_id_column}
    optional_columns = {
        value for value in (cohort_column, assay_id_column, batch_id_column) if value is not None
    }
    missing_columns = sorted((required_columns | optional_columns).difference(adata.obs.columns))
    if missing_columns:
        raise ValueError(f"AnnData obs columns are absent: {missing_columns}")
    if adata.obs.loc[:, list(required_columns)].isna().any().any():
        raise ValueError("required AnnData sample metadata cannot be missing")

    def optional_text(sample_id: str, column: str | None) -> str | None:
        if column is None:
            return None
        value = adata.obs.at[sample_id, column]
        return None if pd.isna(value) else str(value)

    samples = tuple(
        GenomicSample(
            sample_id=sample_id,
            subject_id=str(adata.obs.at[sample_id, subject_id_column]),
            tissue=str(adata.obs.at[sample_id, tissue_column]),
            species_taxon_id=int(cast(Any, adata.obs.at[sample_id, species_taxon_id_column])),
            cohort=optional_text(sample_id, cohort_column),
            assay_id=optional_text(sample_id, assay_id_column),
            batch_id=optional_text(sample_id, batch_id_column),
        )
        for sample_id in obs_names
    )
    features = tuple(
        GenomicFeature(
            feature_id=feature_id,
            feature_type=feature_type,
            namespace=namespace,
            genome_assembly=genome_assembly,
        )
        for feature_id in var_names
    )
    values: MatrixValues
    if sparse.issparse(raw_values):
        values = sparse.csr_matrix(cast(Any, raw_values), dtype=np.float64)
    else:
        values = np.asarray(raw_values, dtype=float)
    return GenomicMatrix(
        values=values,
        samples=samples,
        features=features,
        scale=scale,
        provenance=provenance,
    )


def read_vcf_dosage(
    path: str | Path,
    *,
    species_taxon_id: int,
    tissue: str,
    genome_assembly: str,
    provenance: GenomicMatrixProvenance,
    cohort_by_sample: Mapping[str, str] | None = None,
    subject_by_sample: Mapping[str, str] | None = None,
) -> GenomicMatrix:
    """Read biallelic diploid VCF/BCF genotypes as alternate-allele dosages.

    This adapter uses :mod:`pysam` from the optional ``hts`` extra. Phasing is
    ignored. Multiallelic, non-diploid, and partially missing genotypes are
    rejected instead of being silently simplified.
    """
    try:
        pysam = import_module("pysam")
    except ModuleNotFoundError as error:
        raise ImportError(
            "read_vcf_dosage requires the optional dependency; install rejuvenationkit[hts]"
        ) from error
    variant_file = pysam.VariantFile(str(path))
    try:
        sample_ids = tuple(str(value) for value in variant_file.header.samples)
        rows: list[list[float]] = [[] for _ in sample_ids]
        features: list[GenomicFeature] = []
        for record in variant_file:
            alternatives = tuple(record.alts or ())
            if len(alternatives) != 1:
                raise ValueError(f"VCF record {record.contig}:{record.pos} must be biallelic")
            reference = str(record.ref)
            alternate = str(alternatives[0])
            identifier = str(record.id or f"{record.contig}:{record.pos}:{reference}:{alternate}")
            features.append(
                GenomicFeature(
                    feature_id=identifier,
                    feature_type=GenomicFeatureType.VARIANT,
                    namespace=FeatureNamespace.VCF,
                    genome_assembly=genome_assembly,
                    chromosome=str(record.contig),
                    start=int(record.pos) - 1,
                    end=int(record.pos) - 1 + len(reference),
                    attributes={"reference": reference, "alternate": alternate},
                )
            )
            for index, sample_id in enumerate(sample_ids):
                genotype = record.samples[sample_id].get("GT")
                if genotype is None or all(value is None for value in genotype):
                    rows[index].append(float("nan"))
                    continue
                if len(genotype) != 2 or any(value is None for value in genotype):
                    raise ValueError(
                        f"VCF genotype for {sample_id} at {identifier} is partially missing "
                        "or non-diploid"
                    )
                alleles = tuple(cast(int, value) for value in genotype)
                if any(value not in (0, 1) for value in alleles):
                    raise ValueError(
                        f"VCF genotype for {sample_id} at {identifier} is not diploid biallelic"
                    )
                rows[index].append(float(sum(alleles)))
    finally:
        variant_file.close()
    if not features:
        raise ValueError("VCF contains no variants")
    samples = tuple(
        GenomicSample(
            sample_id=sample_id,
            subject_id=(subject_by_sample or {}).get(sample_id, sample_id),
            tissue=tissue,
            species_taxon_id=species_taxon_id,
            cohort=(cohort_by_sample or {}).get(sample_id),
            assay_id="VCF/BCF genotype",
        )
        for sample_id in sample_ids
    )
    return GenomicMatrix(
        values=np.asarray(rows, dtype=float),
        samples=samples,
        features=tuple(features),
        scale=MatrixScale.VARIANT_DOSAGE,
        provenance=provenance,
    )


def annotate_feature_overlaps(
    matrix: GenomicMatrix,
    annotations: pd.DataFrame,
    *,
    annotation_id_column: str,
    annotation_genome_assembly: str,
) -> pd.DataFrame:
    """Map coordinate-bearing features to intervals using optional bioframe."""
    try:
        bioframe = import_module("bioframe")
    except ModuleNotFoundError as error:
        raise ImportError(
            "annotate_feature_overlaps requires the optional dependency; "
            "install rejuvenationkit[genomics]"
        ) from error
    required = {"chrom", "start", "end", annotation_id_column}
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(f"annotation columns are absent: {missing}")
    matrix_assemblies = {
        item.genome_assembly for item in matrix.features if item.genome_assembly is not None
    }
    if len(matrix_assemblies) > 1:
        raise ValueError("matrix features span multiple genome assemblies")
    if matrix_assemblies != {annotation_genome_assembly}:
        raise ValueError("annotation genome assembly must match every matrix feature")
    records = []
    for feature in matrix.features:
        if feature.chromosome is None or feature.start is None or feature.end is None:
            raise ValueError(f"feature lacks genomic coordinates: {feature.feature_id}")
        records.append(
            {
                "chrom": feature.chromosome,
                "start": feature.start,
                "end": feature.end,
                "feature_id": feature.feature_id,
            }
        )
    left = pd.DataFrame(records)
    selected = annotations.loc[:, ["chrom", "start", "end", annotation_id_column]].copy()
    overlap = bioframe.overlap(left, selected, how="left", suffixes=("_feature", "_annotation"))
    overlap = overlap.rename(
        columns={
            "feature_id_feature": "feature_id",
            f"{annotation_id_column}_annotation": annotation_id_column,
        }
    )
    return cast(pd.DataFrame, overlap)
