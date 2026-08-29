"""Matrix-oriented contracts for genome-scale measurements."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Any, Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
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
    PROTEIN = "protein"
    EMBEDDING_DIMENSION = "embedding_dimension"


class FeatureNamespace(StrEnum):
    """Identifier namespace for genomic features.

    ``HGNC`` and ``STRING`` are retained only for compatibility with early
    RejuvenationKit artifacts.  They do not distinguish HGNC IDs from symbols,
    or STRING protein IDs from provider-preferred symbols.  New artifacts must
    choose one of the explicit namespace members instead.
    """

    ENSEMBL = "ensembl"
    ENTREZ = "entrez"
    HGNC_ID = "hgnc_id"
    HGNC_SYMBOL = "hgnc_symbol"
    STRING_PROTEIN = "string_protein"
    STRING_PREFERRED_SYMBOL = "string_preferred_symbol"
    # Deprecated ambiguous namespaces retained for artifact deserialization.
    HGNC = "hgnc"
    STRING = "string"
    REFSEQ = "refseq"
    ILLUMINA_PROBE = "illumina_probe"
    UNIPROT = "uniprot"
    RSID = "rsid"
    HGVS = "hgvs"
    VCF = "vcf"
    CUSTOM = "custom"

    @property
    def is_legacy_ambiguous(self) -> bool:
        """Return whether this member lacks one exact identifier interpretation."""
        return self in {FeatureNamespace.HGNC, FeatureNamespace.STRING}


_ALLOWED_NAMESPACES_BY_FEATURE_TYPE: dict[
    GenomicFeatureType,
    frozenset[FeatureNamespace],
] = {
    GenomicFeatureType.GENE: frozenset(
        {
            FeatureNamespace.ENSEMBL,
            FeatureNamespace.ENTREZ,
            FeatureNamespace.HGNC_ID,
            FeatureNamespace.HGNC_SYMBOL,
            FeatureNamespace.STRING_PREFERRED_SYMBOL,
            FeatureNamespace.REFSEQ,
            # Compatibility-only ambiguous values.
            FeatureNamespace.HGNC,
            FeatureNamespace.STRING,
        }
    ),
    GenomicFeatureType.TRANSCRIPT: frozenset({FeatureNamespace.ENSEMBL, FeatureNamespace.REFSEQ}),
    GenomicFeatureType.CPG: frozenset({FeatureNamespace.ILLUMINA_PROBE}),
    GenomicFeatureType.REGION: frozenset({FeatureNamespace.ENSEMBL, FeatureNamespace.REFSEQ}),
    GenomicFeatureType.VARIANT: frozenset(
        {
            FeatureNamespace.ENSEMBL,
            FeatureNamespace.RSID,
            FeatureNamespace.HGVS,
            FeatureNamespace.VCF,
        }
    ),
    GenomicFeatureType.PROMOTER: frozenset({FeatureNamespace.ENSEMBL, FeatureNamespace.REFSEQ}),
    GenomicFeatureType.ENHANCER: frozenset({FeatureNamespace.ENSEMBL, FeatureNamespace.REFSEQ}),
    GenomicFeatureType.PROTEIN: frozenset(
        {
            FeatureNamespace.ENSEMBL,
            FeatureNamespace.REFSEQ,
            FeatureNamespace.UNIPROT,
            FeatureNamespace.STRING_PROTEIN,
            # Compatibility-only ambiguous value.
            FeatureNamespace.STRING,
        }
    ),
    GenomicFeatureType.EMBEDDING_DIMENSION: frozenset(),
}

_ASSEMBLY_REQUIRED_NAMESPACES = frozenset({FeatureNamespace.VCF})


def validate_feature_domain_compatibility(
    *,
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
    genome_assembly: str | None,
) -> None:
    """Validate one entity/identifier/assembly combination.

    ``CUSTOM`` is an explicit namespace escape for identifiers whose semantics
    are defined outside the built-in registry. Assembly is required here only
    when the identifier namespace itself is coordinate-keyed. ``GenomicFeature``
    separately requires an assembly whenever explicit coordinates are supplied.
    """
    if genome_assembly is not None and (
        not genome_assembly or genome_assembly != genome_assembly.strip()
    ):
        raise ValueError("genome_assembly must be nonempty without surrounding whitespace")
    allowed = _ALLOWED_NAMESPACES_BY_FEATURE_TYPE[feature_type]
    if namespace is not FeatureNamespace.CUSTOM and namespace not in allowed:
        raise ValueError(
            f"namespace {namespace.value!r} is incompatible with "
            f"feature_type {feature_type.value!r}; use a compatible explicit namespace "
            "or CUSTOM with externally documented semantics"
        )
    if namespace in _ASSEMBLY_REQUIRED_NAMESPACES and genome_assembly is None:
        raise ValueError(
            f"genome_assembly is required for feature_type {feature_type.value!r} "
            f"with namespace {namespace.value!r}"
        )


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
    PROTEIN_ABUNDANCE = "protein_abundance"
    LOG_PROTEIN_ABUNDANCE = "log_protein_abundance"
    EMBEDDING = "embedding"


_ALLOWED_FEATURE_TYPES_BY_SCALE: dict[MatrixScale, frozenset[GenomicFeatureType]] = {
    MatrixScale.RAW_COUNTS: frozenset({GenomicFeatureType.GENE, GenomicFeatureType.TRANSCRIPT}),
    MatrixScale.CPM: frozenset({GenomicFeatureType.GENE, GenomicFeatureType.TRANSCRIPT}),
    MatrixScale.LOG_CPM: frozenset({GenomicFeatureType.GENE, GenomicFeatureType.TRANSCRIPT}),
    MatrixScale.TPM: frozenset({GenomicFeatureType.GENE, GenomicFeatureType.TRANSCRIPT}),
    MatrixScale.NORMALIZED_EXPRESSION: frozenset(
        {GenomicFeatureType.GENE, GenomicFeatureType.TRANSCRIPT}
    ),
    MatrixScale.METHYLATION_BETA: frozenset(
        {
            GenomicFeatureType.CPG,
            GenomicFeatureType.REGION,
            GenomicFeatureType.PROMOTER,
            GenomicFeatureType.ENHANCER,
        }
    ),
    MatrixScale.METHYLATION_M: frozenset(
        {
            GenomicFeatureType.CPG,
            GenomicFeatureType.REGION,
            GenomicFeatureType.PROMOTER,
            GenomicFeatureType.ENHANCER,
        }
    ),
    MatrixScale.VARIANT_DOSAGE: frozenset({GenomicFeatureType.VARIANT}),
    MatrixScale.PROTEIN_ABUNDANCE: frozenset({GenomicFeatureType.PROTEIN}),
    MatrixScale.LOG_PROTEIN_ABUNDANCE: frozenset({GenomicFeatureType.PROTEIN}),
    MatrixScale.EMBEDDING: frozenset({GenomicFeatureType.EMBEDDING_DIMENSION}),
}


MetadataValue: TypeAlias = str | int | float | bool


def _freeze_metadata_mapping(
    values: Mapping[str, MetadataValue],
    *,
    field_name: str,
) -> Mapping[str, MetadataValue]:
    """Copy scalar metadata into an immutable, validated mapping."""
    copied: dict[str, MetadataValue] = {}
    for key, value in values.items():
        if not key or key != key.strip():
            raise ValueError(f"{field_name} keys must be nonempty without surrounding whitespace")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{field_name} float values must be finite")
        copied[key] = value
    return MappingProxyType(copied)


def _freeze_string_mapping(
    values: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    """Copy version-like string metadata into an immutable mapping."""
    copied: dict[str, str] = {}
    for key, value in values.items():
        if not key or key != key.strip():
            raise ValueError(f"{field_name} keys must be nonempty without surrounding whitespace")
        if not value or value != value.strip():
            raise ValueError(f"{field_name} values must be nonempty without surrounding whitespace")
        copied[key] = value
    return MappingProxyType(copied)


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
    attributes: Mapping[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamp_and_attributes(self) -> Self:
        """Require an aware timestamp and install immutable scalar metadata."""
        if self.timestamp is not None and (
            self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None
        ):
            raise ValueError("sample timestamp must be timezone-aware")
        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata_mapping(self.attributes, field_name="sample attributes"),
        )
        return self

    @field_serializer("attributes")
    def serialize_attributes(
        self,
        value: Mapping[str, MetadataValue],
    ) -> dict[str, MetadataValue]:
        """Serialize immutable attributes as an ordinary JSON mapping."""
        return dict(value)


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
    attributes: Mapping[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        """Reject invalid coordinates, domains, and mutable metadata."""
        coordinate_values = (self.chromosome, self.start, self.end)
        supplied = sum(value is not None for value in coordinate_values)
        if supplied not in (0, 3):
            raise ValueError("chromosome, start, and end must be supplied together")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("feature end must be greater than start")
        if supplied == 3 and self.genome_assembly is None:
            raise ValueError("genome_assembly is required with genomic coordinates")
        validate_feature_domain_compatibility(
            feature_type=self.feature_type,
            namespace=self.namespace,
            genome_assembly=self.genome_assembly,
        )
        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata_mapping(self.attributes, field_name="feature attributes"),
        )
        return self

    @field_serializer("attributes")
    def serialize_attributes(
        self,
        value: Mapping[str, MetadataValue],
    ) -> dict[str, MetadataValue]:
        """Serialize immutable attributes as an ordinary JSON mapping."""
        return dict(value)


class GenomicMatrixProvenance(BaseModel):
    """Reproducibility metadata for a genomic matrix artifact."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    source_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preprocessing: tuple[str, ...] = ()
    software_versions: Mapping[str, str] = Field(default_factory=dict)
    reference_resource_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def freeze_software_versions(self) -> Self:
        """Install an immutable copy of hash-relevant software versions."""
        object.__setattr__(
            self,
            "software_versions",
            _freeze_string_mapping(
                self.software_versions,
                field_name="software_versions",
            ),
        )
        return self

    @field_serializer("software_versions")
    def serialize_software_versions(
        self,
        value: Mapping[str, str],
    ) -> dict[str, str]:
        """Serialize immutable software versions as a JSON mapping."""
        return dict(value)


MatrixValues: TypeAlias = npt.NDArray[np.float64] | sparse.csr_matrix

_MATRIX_HASH_SCHEMA = b"rejuvenationkit.genomic-matrix-hash/v2"
_LOGICAL_MATRIX_ENCODING = b"coordinate-float64-le/v1"


def _encode_uint64(value: int) -> bytes:
    """Encode a nonnegative integer using a platform-independent width."""
    if value < 0:
        raise ValueError("canonical hash integers must be nonnegative")
    return value.to_bytes(8, byteorder="big", signed=False)


def _encode_text_sequence(values: tuple[str, ...]) -> bytes:
    """Encode text without separator or concatenation ambiguity."""
    payload = bytearray(_encode_uint64(len(values)))
    for value in values:
        encoded = value.encode("utf-8")
        payload.extend(_encode_uint64(len(encoded)))
        payload.extend(encoded)
    return bytes(payload)


def _hash_tagged_fields(fields: tuple[tuple[str, bytes], ...]) -> str:
    """Hash schema-tagged fields with explicit tag and payload boundaries."""
    digest = sha256()
    for tag, payload in fields:
        encoded_tag = tag.encode("utf-8")
        digest.update(len(encoded_tag).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded_tag)
        digest.update(_encode_uint64(len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _logical_matrix_fields(
    values: MatrixValues,
    shape: tuple[int, int],
) -> tuple[tuple[str, bytes], ...]:
    """Encode logical nonzero and missing cells independent of storage format."""
    if isinstance(values, sparse.csr_matrix):
        coordinate = values.tocoo(copy=False)
        rows = np.asarray(coordinate.row, dtype=np.uint64)
        columns = np.asarray(coordinate.col, dtype=np.uint64)
        data = np.asarray(coordinate.data, dtype=np.float64)
        if data.size:
            order = np.lexsort((columns, rows))
            rows = rows[order]
            columns = columns[order]
            data = data[order]
    else:
        dense = np.asarray(values, dtype=np.float64)
        included = (dense != 0.0) | np.isnan(dense)
        row_indices, column_indices = np.nonzero(included)
        rows = np.asarray(row_indices, dtype=np.uint64)
        columns = np.asarray(column_indices, dtype=np.uint64)
        data = np.asarray(dense[row_indices, column_indices], dtype=np.float64)

    canonical_data = np.asarray(data, dtype="<f8").copy()
    if np.isnan(canonical_data).any():
        # Collapse platform- or source-specific NaN payload bits to one missing value.
        canonical_data[np.isnan(canonical_data)] = np.nan
    return (
        ("matrix-encoding", _LOGICAL_MATRIX_ENCODING),
        ("shape", np.asarray(shape, dtype="<u8").tobytes(order="C")),
        ("entry-count", _encode_uint64(int(canonical_data.size))),
        ("row-indices", np.asarray(rows, dtype="<u8").tobytes(order="C")),
        ("column-indices", np.asarray(columns, dtype="<u8").tobytes(order="C")),
        ("values", canonical_data.tobytes(order="C")),
    )


@dataclass(frozen=True, slots=True)
class GenomicMatrix:
    """Validated samples-by-features matrix without scalar-row expansion.

    Dense matrices may use ``NaN`` for missing values. Sparse matrices treat
    absent entries as measured zeros and therefore cannot encode missingness
    implicitly. Dense versus CSR storage is deliberately not part of artifact
    identity: logically equal matrices hash identically. CSR inputs are therefore
    normalized by combining duplicates, removing explicit zeros, and sorting
    column indices before they are stored.
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
            normalized.sum_duplicates()
            normalized.eliminate_zeros()
            normalized.sort_indices()
            if not np.isfinite(normalized.data).all():
                raise ValueError("sparse genomic matrix values must be finite")
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
        """Hash logical values and aligned IDs using the versioned binary schema.

        Storage representation, scale, sample annotations, feature annotations,
        and provenance are excluded. Every included field is tagged and
        length-delimited, and dense and sparse representations of the same logical
        matrix have the same content identity.
        """
        fields = (
            ("hash-schema", _MATRIX_HASH_SCHEMA),
            ("identity-kind", b"content"),
            ("sample-ids", _encode_text_sequence(self.sample_ids)),
            ("feature-ids", _encode_text_sequence(self.feature_ids)),
            *_logical_matrix_fields(self.values, self.shape),
        )
        return _hash_tagged_fields(fields)

    @property
    def artifact_hash(self) -> str:
        """Hash values plus the complete scientific domain and provenance metadata.

        This semantic artifact identity changes when sample assignments, feature
        metadata, scale, assembly, provenance, or logical values change. Dense versus
        sparse storage is excluded from identity. Metadata and the canonical numerical
        payload are schema-tagged and length-delimited before hashing.
        """
        metadata = {
            "features": [item.model_dump(mode="json") for item in self.features],
            "provenance": self.provenance.model_dump(mode="json"),
            "samples": [item.model_dump(mode="json") for item in self.samples],
            "scale": self.scale.value,
            "shape": self.shape,
        }
        metadata_payload = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        fields = (
            ("hash-schema", _MATRIX_HASH_SCHEMA),
            ("identity-kind", b"artifact"),
            ("metadata-json", metadata_payload),
            *_logical_matrix_fields(self.values, self.shape),
        )
        return _hash_tagged_fields(fields)

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
        elif self.scale in (
            MatrixScale.CPM,
            MatrixScale.TPM,
            MatrixScale.PROTEIN_ABUNDANCE,
        ):
            if np.any(finite < 0):
                raise ValueError(f"{self.scale.value} values must be nonnegative")
        elif self.scale is MatrixScale.METHYLATION_BETA:
            if np.any((finite < 0) | (finite > 1)):
                raise ValueError("methylation beta values must lie in [0, 1]")
        elif self.scale is MatrixScale.VARIANT_DOSAGE:
            if np.any((finite < 0) | (finite > 2)):
                raise ValueError("diploid variant dosage values must lie in [0, 2]")
        feature_types = {item.feature_type for item in self.features}
        incompatible = feature_types.difference(_ALLOWED_FEATURE_TYPES_BY_SCALE[self.scale])
        if incompatible:
            names = sorted(item.value for item in incompatible)
            raise ValueError(
                f"matrix scale {self.scale.value!r} is incompatible with feature types {names}"
            )
