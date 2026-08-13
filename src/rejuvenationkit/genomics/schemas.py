"""Matrix-oriented contracts for genome-scale measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse


class GenomicFeatureType(StrEnum):
    """Biological entity represented by a matrix column."""

    GENE = "gene"
    TRANSCRIPT = "transcript"
    CPG = "cpg"
    REGION = "region"
    VARIANT = "variant"
    PROMOTER = "promoter"
    ENHANCER = "enhancer"
    EMBEDDING_DIMENSION = "embedding_dimension"


class FeatureNamespace(StrEnum):
    """Identifier namespace for genomic features."""

    ENSEMBL = "ensembl"
    ENTREZ = "entrez"
    HGNC = "hgnc"
    REFSEQ = "refseq"
    ILLUMINA_PROBE = "illumina_probe"
    VCF = "vcf"
    CUSTOM = "custom"


class MatrixScale(StrEnum):
    """Declared numerical scale for a genomic feature matrix."""

    RAW_COUNTS = "raw_counts"
    CPM = "cpm"
    LOG_CPM = "log_cpm"
    TPM = "tpm"
    NORMALIZED_EXPRESSION = "normalized_expression"
    METHYLATION_BETA = "methylation_beta"
    METHYLATION_M = "methylation_m"
    VARIANT_DOSAGE = "variant_dosage"
    EMBEDDING = "embedding"


class GenomicSample(BaseModel):
    """One assay sample aligned to a matrix row."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    timestamp: datetime | None = None
    tissue: str = Field(min_length=1)
    species_taxon_id: int = Field(gt=0)
    cohort: str | None = None
    assay_id: str | None = None
    batch_id: str | None = None
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        """Require timezone information whenever a sample timestamp is known."""
        if self.timestamp is not None and (
            self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None
        ):
            raise ValueError("sample timestamp must be timezone-aware")
        return self


class GenomicFeature(BaseModel):
    """One genomic feature aligned to a matrix column."""

    model_config = ConfigDict(frozen=True)

    feature_id: str = Field(min_length=1)
    feature_type: GenomicFeatureType
    namespace: FeatureNamespace
    symbol: str | None = None
    genome_assembly: str | None = None
    chromosome: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        """Reject partial or inverted zero-based half-open coordinates."""
        coordinate_values = (self.chromosome, self.start, self.end)
        supplied = sum(value is not None for value in coordinate_values)
        if supplied not in (0, 3):
            raise ValueError("chromosome, start, and end must be supplied together")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("feature end must be greater than start")
        if supplied == 3 and self.genome_assembly is None:
            raise ValueError("genome_assembly is required with genomic coordinates")
        return self


class GenomicMatrixProvenance(BaseModel):
    """Reproducibility metadata for a genomic matrix artifact."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    source_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preprocessing: tuple[str, ...] = ()
    software_versions: dict[str, str] = Field(default_factory=dict)
    reference_resource_ids: tuple[str, ...] = ()


MatrixValues: TypeAlias = npt.NDArray[np.float64] | sparse.csr_matrix


@dataclass(frozen=True, slots=True)
class GenomicMatrix:
    """Validated samples-by-features matrix without scalar-row expansion.

    Dense matrices may use ``NaN`` for missing values. Sparse matrices treat
    absent entries as measured zeros and therefore cannot encode missingness
    implicitly.
    """

    values: MatrixValues
    samples: tuple[GenomicSample, ...]
    features: tuple[GenomicFeature, ...]
    scale: MatrixScale
    provenance: GenomicMatrixProvenance

    def __post_init__(self) -> None:
        """Validate shape, identifiers, domain metadata, and scale constraints."""
        matrix = self.values
        if sparse.issparse(matrix):
            normalized = sparse.csr_matrix(cast(Any, matrix), dtype=np.float64, copy=True)
            if not np.isfinite(normalized.data).all():
                raise ValueError("sparse genomic matrix values must be finite")
            normalized.sort_indices()
            normalized.data.flags.writeable = False
            normalized.indices.flags.writeable = False
            normalized.indptr.flags.writeable = False
            object.__setattr__(self, "values", normalized)
        else:
            normalized_dense = np.asarray(matrix, dtype=float).copy()
            if normalized_dense.ndim != 2:
                raise ValueError("genomic matrix values must be two-dimensional")
            if np.isinf(normalized_dense).any():
                raise ValueError("genomic matrix values cannot contain infinity")
            normalized_dense.flags.writeable = False
            object.__setattr__(self, "values", normalized_dense)
        if self.shape != (len(self.samples), len(self.features)):
            raise ValueError("matrix shape must match sample and feature metadata")
        sample_ids = self.sample_ids
        feature_ids = self.feature_ids
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample identifiers must be unique")
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("feature identifiers must be unique")
        if not self.samples or not self.features:
            raise ValueError("genomic matrix must contain samples and features")
        self._validate_scale()

    @property
    def shape(self) -> tuple[int, int]:
        """Return samples-by-features dimensions."""
        return int(self.values.shape[0]), int(self.values.shape[1])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return row identifiers in matrix order."""
        return tuple(item.sample_id for item in self.samples)

    @property
    def feature_ids(self) -> tuple[str, ...]:
        """Return column identifiers in matrix order."""
        return tuple(item.feature_id for item in self.features)

    @property
    def is_sparse(self) -> bool:
        """Return whether values use compressed sparse storage."""
        return isinstance(self.values, sparse.csr_matrix)

    @property
    def missing_fraction(self) -> float:
        """Return explicit dense missingness; sparse absence represents zero."""
        if isinstance(self.values, sparse.csr_matrix):
            return 0.0
        return float(np.isnan(self.values).mean())

    @property
    def content_hash(self) -> str:
        """Hash matrix values and aligned identifiers without source file access."""
        digest = sha256()
        digest.update("\0".join(self.sample_ids).encode())
        digest.update("\0".join(self.feature_ids).encode())
        if isinstance(self.values, sparse.csr_matrix):
            digest.update(self.values.data.tobytes())
            digest.update(self.values.indices.tobytes())
            digest.update(self.values.indptr.tobytes())
        else:
            digest.update(self.values.tobytes())
        return digest.hexdigest()

    def feature_indices(self, feature_ids: tuple[str, ...]) -> tuple[int, ...]:
        """Resolve requested features in caller order and reject absent IDs."""
        lookup = {identifier: index for index, identifier in enumerate(self.feature_ids)}
        missing = sorted(set(feature_ids).difference(lookup))
        if missing:
            raise ValueError(f"features are absent from matrix: {missing}")
        return tuple(lookup[identifier] for identifier in feature_ids)

    def sample_indices(self, sample_ids: tuple[str, ...]) -> tuple[int, ...]:
        """Resolve requested samples in caller order and reject absent IDs."""
        lookup = {identifier: index for index, identifier in enumerate(self.sample_ids)}
        missing = sorted(set(sample_ids).difference(lookup))
        if missing:
            raise ValueError(f"samples are absent from matrix: {missing}")
        return tuple(lookup[identifier] for identifier in sample_ids)

    def subset_samples(self, sample_ids: tuple[str, ...]) -> GenomicMatrix:
        """Return a sample subset without densifying sparse values."""
        indices = self.sample_indices(sample_ids)
        if isinstance(self.values, sparse.csr_matrix):
            values: MatrixValues = sparse.csr_matrix(cast(Any, self.values)[list(indices), :])
        else:
            values = self.values[list(indices), :]
        samples = tuple(self.samples[index] for index in indices)
        return GenomicMatrix(
            values=values,
            samples=samples,
            features=self.features,
            scale=self.scale,
            provenance=self.provenance,
        )

    def subset_features(self, feature_ids: tuple[str, ...]) -> GenomicMatrix:
        """Return a feature subset without densifying sparse values."""
        indices = self.feature_indices(feature_ids)
        if isinstance(self.values, sparse.csr_matrix):
            values: MatrixValues = sparse.csr_matrix(cast(Any, self.values)[:, list(indices)])
        else:
            values = self.values[:, list(indices)]
        features = tuple(self.features[index] for index in indices)
        return GenomicMatrix(
            values=(sparse.csr_matrix(values) if isinstance(values, sparse.csr_matrix) else values),
            samples=self.samples,
            features=features,
            scale=self.scale,
            provenance=self.provenance,
        )

    def dense_values(self, *, maximum_cells: int = 10_000_000) -> npt.NDArray[np.float64]:
        """Materialize values only beneath an explicit memory guard."""
        cells = self.shape[0] * self.shape[1]
        if cells > maximum_cells:
            raise ValueError(
                f"dense materialization would create {cells:,} cells; "
                f"maximum_cells={maximum_cells:,}"
            )
        if isinstance(self.values, sparse.csr_matrix):
            return np.asarray(self.values.toarray(), dtype=np.float64)
        return np.asarray(self.values, dtype=np.float64).copy()

    def _finite_values(self) -> npt.NDArray[np.float64]:
        if isinstance(self.values, sparse.csr_matrix):
            return np.asarray(self.values.data, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        return cast(npt.NDArray[np.float64], values[np.isfinite(values)])

    def _validate_scale(self) -> None:
        finite = self._finite_values()
        if self.scale is MatrixScale.RAW_COUNTS:
            if np.any(finite < 0) or not np.allclose(finite, np.round(finite)):
                raise ValueError("raw counts must be nonnegative integers")
        elif self.scale in (MatrixScale.CPM, MatrixScale.TPM):
            if np.any(finite < 0):
                raise ValueError(f"{self.scale.value} values must be nonnegative")
        elif self.scale is MatrixScale.METHYLATION_BETA:
            if np.any((finite < 0) | (finite > 1)):
                raise ValueError("methylation beta values must lie in [0, 1]")
        elif self.scale is MatrixScale.VARIANT_DOSAGE:
            if np.any((finite < 0) | (finite > 2)):
                raise ValueError("diploid variant dosage values must lie in [0, 2]")
