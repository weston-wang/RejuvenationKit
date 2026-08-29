"""Bounded-memory, resumable stores for archived biological tables.

The writer consumes caller-supplied DataFrame chunks, never concatenates the
catalog, and uses a disk-backed SQLite key index for cross-partition duplicate
checks.  It intentionally stops at an integrity-bound table store: callers can
later stream selected partitions into the typed annotation, network, or variant
importers without materializing a complete provider catalog.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rejuvenationkit.genomics.resources import (
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)

_STORE_FORMAT: Literal["rejuvenationkit-chunked-table/v1"] = "rejuvenationkit-chunked-table/v1"
_MANIFEST_NAME = "manifest.json"
_CHECKPOINT_NAME = ".checkpoint.json"
_KEY_DATABASE_NAME = ".duplicate-keys.sqlite3"
_PARTITION_DIRECTORY = "partitions"
_PARTITION_RE = re.compile(r"^part-(\d{6})\.(ndjson|parquet)$")


class ChunkedTableKind(StrEnum):
    """Supported external catalog families."""

    FUNCTIONAL_ANNOTATION = "functional_annotation"
    INTERACTION_NETWORK = "interaction_network"
    VARIANT_ANNOTATION = "variant_annotation"


class ChunkedStorageFormat(StrEnum):
    """On-disk partition encoding."""

    NDJSON = "ndjson"
    PARQUET = "parquet"


class DuplicatePolicy(StrEnum):
    """Bounded-memory handling of keys repeated within or between chunks."""

    ERROR = "error"
    DROP_LATER = "drop_later"
    ALLOW = "allow"


class TableColumn(BaseModel):
    """One ordered provider-table column and its exact pandas dtype string."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    pandas_dtype: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_column(self) -> Self:
        """Reject ambiguous whitespace in names and dtype declarations."""
        _require_clean_text(self.name, "column name")
        _require_clean_text(self.pandas_dtype, "pandas_dtype")
        return self


class ChunkedImportSpec(BaseModel):
    """Immutable configuration and scientific binding for a chunked table store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    kind: ChunkedTableKind
    domain: FeatureDomain
    resource: ResourceSnapshot
    provenance: QueryProvenance
    key_columns: tuple[str, ...] = ()
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.ERROR
    column_roles: Mapping[str, str] = Field(default_factory=dict)
    expected_schema: tuple[TableColumn, ...] | None = None
    rows_per_partition: int = Field(default=100_000, ge=1)
    storage_format: ChunkedStorageFormat = ChunkedStorageFormat.NDJSON

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        """Require exact resource/query/domain identity and an explicit key policy."""
        _require_clean_text(self.dataset_id, "dataset_id")
        if self.resource.response_sha256 is None:
            raise ValueError("chunked imports require a raw-byte resource SHA-256")
        if self.provenance.domain != self.domain:
            raise ValueError("chunked import domain must match query provenance")
        if self.resource not in self.provenance.resources:
            raise ValueError("chunked import resource must be captured in query provenance")
        _require_unique_clean_strings(self.key_columns, "key_columns")
        if self.duplicate_policy is not DuplicatePolicy.ALLOW and not self.key_columns:
            raise ValueError("error/drop_later duplicate policies require key_columns")
        roles = dict(self.column_roles)
        for role, column in roles.items():
            _require_clean_text(role, "column role")
            _require_clean_text(column, f"column for role {role}")
        if len(set(roles.values())) != len(roles):
            raise ValueError("column roles must map to unique table columns")
        object.__setattr__(self, "column_roles", MappingProxyType(dict(sorted(roles.items()))))
        if self.expected_schema is not None:
            names = tuple(column.name for column in self.expected_schema)
            _require_unique_clean_strings(names, "expected schema columns")
            _require_columns_in_schema(self.key_columns, names, "key columns")
            _require_columns_in_schema(tuple(roles.values()), names, "column-role mappings")
        return self

    @field_serializer("column_roles")
    def serialize_roles(self, value: Mapping[str, str]) -> dict[str, str]:
        """Serialize the immutable role map as ordinary JSON data."""
        return dict(value)

    @property
    def spec_hash(self) -> str:
        """Return a deterministic identity for resume-safety checks."""
        return canonical_sha256(self.model_dump(mode="python"))


class ChunkedPartition(BaseModel):
    """Integrity receipt for one bounded table partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    file_name: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    byte_count: int = Field(ge=0)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    first_key_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    last_key_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_file_name(self) -> Self:
        """Keep partition references local and aligned with their index."""
        match = _PARTITION_RE.fullmatch(self.file_name)
        if match is None or int(match.group(1)) != self.index:
            raise ValueError("partition file_name must match its zero-based partition index")
        if Path(self.file_name).name != self.file_name:
            raise ValueError("partition file_name must not contain a path")
        if (self.first_key_sha256 is None) != (self.last_key_sha256 is None):
            raise ValueError("partition key bounds must be both present or both absent")
        return self


class ChunkedImportManifest(BaseModel):
    """Final immutable manifest for a complete streamed table import."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store_format: Literal["rejuvenationkit-chunked-table/v1"] = _STORE_FORMAT
    spec: ChunkedImportSpec
    schema_definition: tuple[TableColumn, ...]
    partitions: tuple[ChunkedPartition, ...]
    source_row_count: int = Field(ge=0)
    retained_row_count: int = Field(ge=0)
    duplicate_row_count: int = Field(ge=0)
    logical_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    writer_versions: Mapping[str, str]
    finalized: Literal[True] = True
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Reconstruct row accounting, partition order, and logical identity."""
        _validate_schema_for_spec(self.schema_definition, self.spec)
        indices = tuple(partition.index for partition in self.partitions)
        if indices != tuple(range(len(self.partitions))):
            raise ValueError("chunked partitions must be contiguous and canonically ordered")
        expected_extension = self.spec.storage_format.value
        if any(not item.file_name.endswith(f".{expected_extension}") for item in self.partitions):
            raise ValueError("partition extension does not match the declared storage format")
        retained = sum(partition.row_count for partition in self.partitions)
        if retained != self.retained_row_count:
            raise ValueError("partition row counts do not match retained_row_count")
        if self.source_row_count != self.retained_row_count + self.duplicate_row_count:
            raise ValueError("source rows must partition into retained and duplicate rows")
        if self.spec.duplicate_policy is not DuplicatePolicy.DROP_LATER:
            if self.duplicate_row_count != 0:
                raise ValueError("only drop_later imports can report discarded duplicates")
        for partition in self.partitions[:-1]:
            if partition.row_count != self.spec.rows_per_partition:
                raise ValueError("all non-final partitions must use rows_per_partition")
        if self.partitions and self.partitions[-1].row_count > self.spec.rows_per_partition:
            raise ValueError("final partition exceeds rows_per_partition")
        expected_content_hash = _logical_store_hash(self.partitions)
        if self.logical_content_sha256 != expected_content_hash:
            raise ValueError("logical_content_sha256 does not match partition receipts")
        versions = dict(self.writer_versions)
        for name, version in versions.items():
            _require_clean_text(name, "writer software name")
            _require_clean_text(version, f"writer version for {name}")
        if self.spec.storage_format is ChunkedStorageFormat.PARQUET and "pyarrow" not in versions:
            raise ValueError("Parquet manifests must record the exact pyarrow version")
        object.__setattr__(
            self,
            "writer_versions",
            MappingProxyType(dict(sorted(versions.items()))),
        )
        return self

    @field_serializer("writer_versions")
    def serialize_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        """Serialize immutable writer versions as ordinary JSON data."""
        return dict(value)

    @property
    def manifest_hash(self) -> str:
        """Return a deterministic identity for the finalized store."""
        return canonical_sha256(self.model_dump(mode="python"))


class ChunkedImportCheckpoint(BaseModel):
    """Resume position containing only fully committed partitions and source rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store_format: Literal["rejuvenationkit-chunked-table/v1"] = _STORE_FORMAT
    spec: ChunkedImportSpec
    schema_definition: tuple[TableColumn, ...] | None = None
    partitions: tuple[ChunkedPartition, ...] = ()
    source_row_count: int = Field(default=0, ge=0)
    retained_row_count: int = Field(default=0, ge=0)
    duplicate_row_count: int = Field(default=0, ge=0)
    finalized: bool = False

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        """Validate the same accounting guarantees used by final manifests."""
        if self.schema_definition is not None:
            _validate_schema_for_spec(self.schema_definition, self.spec)
        elif self.partitions or self.source_row_count:
            raise ValueError("a nonempty checkpoint requires a schema definition")
        indices = tuple(item.index for item in self.partitions)
        if indices != tuple(range(len(self.partitions))):
            raise ValueError("checkpoint partitions must be contiguous")
        if sum(item.row_count for item in self.partitions) != self.retained_row_count:
            raise ValueError("checkpoint partition counts do not match retained rows")
        if self.source_row_count != self.retained_row_count + self.duplicate_row_count:
            raise ValueError("checkpoint source row accounting is inconsistent")
        return self


def write_chunked_table(
    chunks: Iterable[pd.DataFrame],
    destination: str | os.PathLike[str],
    *,
    spec: ChunkedImportSpec,
    resume: bool = False,
) -> ChunkedImportManifest:
    """Stream table chunks into an atomic, resumable, integrity-checked directory.

    On failure, only fully committed partitions count toward the checkpoint.
    When resuming, ``chunks`` must begin at ``checkpoint.source_row_count`` in
    the original source; earlier rows must not be replayed.
    """
    target = Path(destination)
    stage = _staging_path(target)
    _reject_symlink(target, "chunked table destination")
    _reject_symlink(stage, "chunked table staging directory")
    if target.exists():
        raise FileExistsError(f"chunked table destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        _require_real_directory(stage, "chunked table staging directory")
        checkpoint = _load_checkpoint(stage)
        if checkpoint.spec.spec_hash != spec.spec_hash:
            raise ValueError("resume specification does not match the staged import")
        if checkpoint.finalized:
            manifest = _read_manifest_file(stage / _MANIFEST_NAME)
            stage.rename(target)
            return manifest
        _reconcile_stage(stage, checkpoint)
    else:
        if stage.exists():
            raise FileExistsError(
                f"an unfinished import already exists at {stage}; use resume=True"
            )
        stage.mkdir(mode=0o700)
        (stage / _PARTITION_DIRECTORY).mkdir(mode=0o700)
        checkpoint = ChunkedImportCheckpoint(spec=spec)
        _write_checkpoint(stage, checkpoint)

    database = _open_rebuilt_key_database(stage, checkpoint)
    schema = checkpoint.schema_definition
    partitions = list(checkpoint.partitions)
    source_rows = checkpoint.source_row_count
    retained_rows = checkpoint.retained_row_count
    duplicate_rows = checkpoint.duplicate_row_count
    pending_rows: list[dict[str, object]] = []
    pending_key_hashes: list[str] = []
    transaction_open = False
    try:
        if database is not None:
            database.execute("BEGIN")
            transaction_open = True
        for frame in chunks:
            frame_schema = _schema_from_frame(frame)
            if schema is None:
                schema = frame_schema
                _validate_schema_for_spec(schema, spec)
            elif frame_schema != schema:
                raise ValueError("DataFrame chunk schema differs from the established schema")
            columns = tuple(column.name for column in schema)
            for values in frame.itertuples(index=False, name=None):
                row = {
                    column: _normalize_cell(value, column)
                    for column, value in zip(columns, values, strict=True)
                }
                source_rows += 1
                key_hash: str | None = None
                if spec.key_columns:
                    key_json = _key_json(row, spec.key_columns)
                    key_hash = sha256(key_json.encode("utf-8")).hexdigest()
                retained = _register_key(
                    database,
                    key_hash=key_hash,
                    key_json=key_json if spec.key_columns else None,
                    policy=spec.duplicate_policy,
                )
                if not retained:
                    duplicate_rows += 1
                    continue
                pending_rows.append(row)
                if key_hash is not None:
                    pending_key_hashes.append(key_hash)
                if len(pending_rows) == spec.rows_per_partition:
                    partition = _write_partition(
                        stage,
                        index=len(partitions),
                        rows=pending_rows,
                        key_hashes=pending_key_hashes,
                        storage_format=spec.storage_format,
                    )
                    if database is not None:
                        database.commit()
                        transaction_open = False
                    partitions.append(partition)
                    retained_rows += len(pending_rows)
                    pending_rows = []
                    pending_key_hashes = []
                    checkpoint = _checkpoint(
                        spec,
                        schema,
                        partitions,
                        source_rows,
                        retained_rows,
                        duplicate_rows,
                    )
                    _write_checkpoint(stage, checkpoint)
                    if database is not None:
                        database.execute("BEGIN")
                        transaction_open = True
        if schema is None:
            if spec.expected_schema is None:
                raise ValueError("cannot infer a schema from an empty chunk stream")
            schema = spec.expected_schema
        if pending_rows:
            partition = _write_partition(
                stage,
                index=len(partitions),
                rows=pending_rows,
                key_hashes=pending_key_hashes,
                storage_format=spec.storage_format,
            )
            partitions.append(partition)
            retained_rows += len(pending_rows)
        if database is not None:
            database.commit()
            transaction_open = False
        checkpoint = _checkpoint(
            spec,
            schema,
            partitions,
            source_rows,
            retained_rows,
            duplicate_rows,
        )
        _write_checkpoint(stage, checkpoint)
        writer_versions = {"rejuvenationkit-chunked-writer": "1"}
        if spec.storage_format is ChunkedStorageFormat.PARQUET:
            pyarrow = _import_pyarrow()
            writer_versions["pyarrow"] = str(pyarrow.__version__)
        manifest = ChunkedImportManifest(
            spec=spec,
            schema_definition=schema,
            partitions=tuple(partitions),
            source_row_count=source_rows,
            retained_row_count=retained_rows,
            duplicate_row_count=duplicate_rows,
            logical_content_sha256=_logical_store_hash(partitions),
            writer_versions=writer_versions,
        )
        _atomic_json_write(
            stage / _MANIFEST_NAME,
            {
                "manifest": manifest.model_dump(mode="json"),
                "manifest_hash": manifest.manifest_hash,
            },
        )
        finalized_checkpoint = checkpoint.model_copy(update={"finalized": True})
        _write_checkpoint(stage, finalized_checkpoint)
        _reject_symlink(target, "chunked table destination")
        if target.exists():
            raise FileExistsError(f"chunked table destination appeared during import: {target}")
        stage.rename(target)
        return manifest
    except Exception:
        if database is not None and transaction_open:
            database.rollback()
        raise
    finally:
        if database is not None:
            database.close()


def write_chunked_annotations(
    chunks: Iterable[pd.DataFrame],
    destination: str | os.PathLike[str],
    *,
    spec: ChunkedImportSpec,
    resume: bool = False,
) -> ChunkedImportManifest:
    """Write a bounded functional-annotation table after checking its declared kind."""
    _require_kind(spec, ChunkedTableKind.FUNCTIONAL_ANNOTATION)
    return write_chunked_table(chunks, destination, spec=spec, resume=resume)


def write_chunked_network(
    chunks: Iterable[pd.DataFrame],
    destination: str | os.PathLike[str],
    *,
    spec: ChunkedImportSpec,
    resume: bool = False,
) -> ChunkedImportManifest:
    """Write a bounded interaction-network table after checking its declared kind."""
    _require_kind(spec, ChunkedTableKind.INTERACTION_NETWORK)
    return write_chunked_table(chunks, destination, spec=spec, resume=resume)


def write_chunked_variants(
    chunks: Iterable[pd.DataFrame],
    destination: str | os.PathLike[str],
    *,
    spec: ChunkedImportSpec,
    resume: bool = False,
) -> ChunkedImportManifest:
    """Write a bounded variant-annotation table after checking its declared kind."""
    _require_kind(spec, ChunkedTableKind.VARIANT_ANNOTATION)
    return write_chunked_table(chunks, destination, spec=spec, resume=resume)


def read_chunked_manifest(
    source: str | os.PathLike[str],
    *,
    verify_partitions: bool = True,
) -> ChunkedImportManifest:
    """Read a finalized manifest and optionally stream-verify every partition."""
    root = Path(source)
    _require_real_directory(root, "chunked table root")
    manifest = _read_manifest_file(root / _MANIFEST_NAME)
    if verify_partitions:
        for partition in manifest.partitions:
            _read_partition(root, manifest, partition, verify=True)
    return manifest


def read_chunked_checkpoint(destination: str | os.PathLike[str]) -> ChunkedImportCheckpoint:
    """Read the durable resume position for a final or unfinished destination."""
    target = Path(destination)
    _reject_symlink(target, "chunked table destination")
    root = target if target.exists() else _staging_path(target)
    _require_real_directory(root, "chunked checkpoint root")
    return _load_checkpoint(root)


def iter_chunked_partitions(
    source: str | os.PathLike[str],
    *,
    verify: bool = True,
) -> Iterator[pd.DataFrame]:
    """Yield one verified bounded partition at a time without catalog materialization."""
    root = Path(source)
    _require_real_directory(root, "chunked table root")
    manifest = _read_manifest_file(root / _MANIFEST_NAME)
    for partition in manifest.partitions:
        yield _read_partition(root, manifest, partition, verify=verify)


def iter_csv_chunks(
    source: str | os.PathLike[str],
    *,
    chunk_rows: int = 100_000,
    read_csv_options: Mapping[str, object] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield bounded pandas chunks from one local delimited archived table."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    options = dict(read_csv_options or {})
    forbidden = sorted({"chunksize", "iterator"}.intersection(options))
    if forbidden:
        raise ValueError(f"read_csv_options cannot override streaming controls: {forbidden}")
    reader = cast(Any, pd.read_csv)(Path(source), chunksize=chunk_rows, **options)
    for frame in reader:
        yield cast(pd.DataFrame, frame)


def iter_parquet_chunks(
    source: str | os.PathLike[str],
    *,
    batch_rows: int = 100_000,
    columns: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield bounded record batches from local Parquet via lazily loaded PyArrow.

    PyArrow is optional.  Install ``rejuvenationkit[arrow]`` before using this
    path; importing :mod:`rejuvenationkit.genomics.chunked` does not import it.
    Object-backed text columns written by pandas are restored from the file's
    pandas schema metadata rather than depending on pandas' evolving default
    string-inference policy.
    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    _import_pyarrow()
    parquet = _import_pyarrow_parquet()
    parquet_file = parquet.ParquetFile(Path(source))
    object_text_columns = _pandas_object_text_columns(parquet_file.schema_arrow)
    for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=columns):
        frame = cast(pd.DataFrame, batch.to_pandas())
        for column in object_text_columns.intersection(frame.columns):
            frame[column] = frame[column].astype(object)
        yield frame


def _pandas_object_text_columns(schema: Any) -> frozenset[str]:
    """Read pandas' explicit object-string declarations from an Arrow schema."""
    metadata = getattr(schema, "metadata", None)
    if not isinstance(metadata, Mapping) or b"pandas" not in metadata:
        return frozenset()
    raw_metadata = metadata[b"pandas"]
    if not isinstance(raw_metadata, bytes):
        raise ValueError("Parquet pandas schema metadata must be UTF-8 JSON bytes")
    try:
        pandas_metadata = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Parquet pandas schema metadata is not valid UTF-8 JSON") from exc
    if not isinstance(pandas_metadata, dict) or not isinstance(
        pandas_metadata.get("columns"), list
    ):
        raise ValueError("Parquet pandas schema metadata has no valid columns declaration")
    object_text_columns: set[str] = set()
    for declaration in pandas_metadata["columns"]:
        if not isinstance(declaration, dict):
            raise ValueError("Parquet pandas column metadata must contain objects")
        field_name = declaration.get("field_name")
        if not isinstance(field_name, str):
            raise ValueError("Parquet pandas column metadata requires string field names")
        if (
            declaration.get("pandas_type") == "unicode"
            and declaration.get("numpy_type") == "object"
        ):
            object_text_columns.add(field_name)
    return frozenset(object_text_columns)


def _checkpoint(
    spec: ChunkedImportSpec,
    schema: tuple[TableColumn, ...],
    partitions: Sequence[ChunkedPartition],
    source_rows: int,
    retained_rows: int,
    duplicate_rows: int,
) -> ChunkedImportCheckpoint:
    return ChunkedImportCheckpoint(
        spec=spec,
        schema_definition=schema,
        partitions=tuple(partitions),
        source_row_count=source_rows,
        retained_row_count=retained_rows,
        duplicate_row_count=duplicate_rows,
    )


def _staging_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.rejuvenationkit-staging")


def _schema_from_frame(frame: pd.DataFrame) -> tuple[TableColumn, ...]:
    if frame.columns.duplicated().any():
        duplicates = sorted(set(str(item) for item in frame.columns[frame.columns.duplicated()]))
        raise ValueError(f"DataFrame chunk contains duplicate columns: {duplicates}")
    if any(not isinstance(column, str) for column in frame.columns):
        raise TypeError("DataFrame chunk column labels must be strings")
    return tuple(
        TableColumn(name=column, pandas_dtype=str(frame.dtypes.iloc[index]))
        for index, column in enumerate(cast(Sequence[str], frame.columns))
    )


def _validate_schema_for_spec(
    schema: tuple[TableColumn, ...],
    spec: ChunkedImportSpec,
) -> None:
    names = tuple(column.name for column in schema)
    _require_unique_clean_strings(names, "schema columns")
    _require_columns_in_schema(spec.key_columns, names, "key columns")
    _require_columns_in_schema(tuple(spec.column_roles.values()), names, "column-role mappings")
    if spec.expected_schema is not None and schema != spec.expected_schema:
        raise ValueError("table schema does not match expected_schema")


def _require_columns_in_schema(
    requested: Sequence[str],
    available: Sequence[str],
    description: str,
) -> None:
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(f"{description} are absent from the table schema: {missing}")


def _write_partition(
    stage: Path,
    *,
    index: int,
    rows: Sequence[Mapping[str, object]],
    key_hashes: Sequence[str],
    storage_format: ChunkedStorageFormat,
) -> ChunkedPartition:
    suffix = storage_format.value
    file_name = f"part-{index:06d}.{suffix}"
    partition_directory = stage / _PARTITION_DIRECTORY
    _require_real_directory(partition_directory, "staged partition directory")
    destination = partition_directory / file_name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    logical_hasher = sha256()
    try:
        if storage_format is ChunkedStorageFormat.NDJSON:
            with temporary.open("wb") as handle:
                for row in rows:
                    encoded = _row_bytes(row)
                    _update_length_prefixed(logical_hasher, encoded)
                    handle.write(encoded)
                    handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        else:
            for row in rows:
                _update_length_prefixed(logical_hasher, _row_bytes(row))
            pyarrow = _import_pyarrow()
            parquet = _import_pyarrow_parquet()
            table = pyarrow.Table.from_pylist([dict(row) for row in rows])
            parquet.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        payload_hash, byte_count = _hash_file(temporary)
        os.replace(temporary, destination)
        return ChunkedPartition(
            index=index,
            file_name=file_name,
            row_count=len(rows),
            byte_count=byte_count,
            file_sha256=payload_hash,
            logical_sha256=logical_hasher.hexdigest(),
            first_key_sha256=key_hashes[0] if key_hashes else None,
            last_key_sha256=key_hashes[-1] if key_hashes else None,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _read_partition(
    root: Path,
    manifest: ChunkedImportManifest,
    partition: ChunkedPartition,
    *,
    verify: bool,
) -> pd.DataFrame:
    partition_directory = root / _PARTITION_DIRECTORY
    _require_real_directory(partition_directory, "chunked partition directory")
    path = partition_directory / partition.file_name
    _require_regular_file(path, f"chunked partition {partition.file_name}")
    if verify:
        payload_hash, byte_count = _hash_file(path)
        if payload_hash != partition.file_sha256 or byte_count != partition.byte_count:
            raise ValueError(f"partition bytes do not match manifest: {partition.file_name}")
    rows = list(_iter_partition_rows(path, manifest.spec.storage_format))
    if len(rows) != partition.row_count:
        raise ValueError(f"partition row count does not match manifest: {partition.file_name}")
    if verify:
        logical_hasher = sha256()
        for row in rows:
            _update_length_prefixed(logical_hasher, _row_bytes(row))
        if logical_hasher.hexdigest() != partition.logical_sha256:
            raise ValueError(f"partition logical checksum does not match: {partition.file_name}")
    frame = pd.DataFrame.from_records(
        rows,
        columns=[column.name for column in manifest.schema_definition],
    )
    return _restore_schema(frame, manifest.schema_definition)


def _iter_partition_rows(
    path: Path,
    storage_format: ChunkedStorageFormat,
) -> Iterator[dict[str, object]]:
    if storage_format is ChunkedStorageFormat.NDJSON:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid NDJSON in {path.name} at line {line_number}"
                    ) from exc
                if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
                    raise ValueError(f"partition row is not a string-keyed object: {path.name}")
                yield cast(dict[str, object], row)
        return
    _import_pyarrow()
    parquet = _import_pyarrow_parquet()
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            yield cast(dict[str, object], row)


def _restore_schema(
    frame: pd.DataFrame,
    schema: Sequence[TableColumn],
) -> pd.DataFrame:
    restored = frame.copy()
    for column in schema:
        if column.pandas_dtype.startswith("datetime64"):
            restored[column.name] = pd.to_datetime(restored[column.name])
        else:
            try:
                restored[column.name] = restored[column.name].astype(cast(Any, column.pandas_dtype))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot restore column {column.name!r} as {column.pandas_dtype!r}"
                ) from exc
    return restored


def _normalize_cell(value: object, column: str) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError(f"column {column!r} contains a non-finite value")
        return value
    missing = pd.isna(cast(Any, value))
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    raise TypeError(
        f"column {column!r} contains unsupported non-scalar type {type(value).__name__}"
    )


def _row_bytes(row: Mapping[str, object]) -> bytes:
    return json.dumps(
        row,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _key_json(row: Mapping[str, object], columns: Sequence[str]) -> str:
    values: list[dict[str, object]] = []
    for column in columns:
        value = row[column]
        if value is None:
            raise ValueError(f"duplicate key column {column!r} cannot be missing")
        values.append({"type": type(value).__name__, "value": value})
    return json.dumps(values, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def _register_key(
    database: sqlite3.Connection | None,
    *,
    key_hash: str | None,
    key_json: str | None,
    policy: DuplicatePolicy,
) -> bool:
    if policy is DuplicatePolicy.ALLOW:
        return True
    if database is None or key_hash is None or key_json is None:
        raise RuntimeError("disk-backed duplicate index is unavailable")
    existing = database.execute(
        "SELECT key_json FROM seen_keys WHERE key_hash = ?", (key_hash,)
    ).fetchone()
    if existing is None:
        database.execute(
            "INSERT INTO seen_keys (key_hash, key_json) VALUES (?, ?)",
            (key_hash, key_json),
        )
        return True
    if existing[0] != key_json:
        raise RuntimeError("SHA-256 collision detected in duplicate-key index")
    if policy is DuplicatePolicy.ERROR:
        raise ValueError(f"duplicate archived-table key encountered: {key_json}")
    return False


def _open_rebuilt_key_database(
    stage: Path,
    checkpoint: ChunkedImportCheckpoint,
) -> sqlite3.Connection | None:
    if checkpoint.spec.duplicate_policy is DuplicatePolicy.ALLOW:
        return None
    database_path = stage / _KEY_DATABASE_NAME
    database_path.unlink(missing_ok=True)
    database = sqlite3.connect(database_path)
    database.execute("CREATE TABLE seen_keys (key_hash TEXT PRIMARY KEY, key_json TEXT NOT NULL)")
    if checkpoint.schema_definition is None:
        return database
    partition_directory = stage / _PARTITION_DIRECTORY
    _require_real_directory(partition_directory, "staged partition directory")
    for partition in checkpoint.partitions:
        path = partition_directory / partition.file_name
        _require_regular_file(path, f"staged partition {partition.file_name}")
        for row in _iter_partition_rows(path, checkpoint.spec.storage_format):
            key_json = _key_json(row, checkpoint.spec.key_columns)
            key_hash = sha256(key_json.encode("utf-8")).hexdigest()
            try:
                database.execute(
                    "INSERT INTO seen_keys (key_hash, key_json) VALUES (?, ?)",
                    (key_hash, key_json),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("committed partitions contain duplicate keys") from exc
    database.commit()
    return database


def _reconcile_stage(stage: Path, checkpoint: ChunkedImportCheckpoint) -> None:
    _require_real_directory(stage, "chunked table staging directory")
    partition_dir = stage / _PARTITION_DIRECTORY
    _require_real_directory(partition_dir, "staged partition directory")
    expected = {item.file_name for item in checkpoint.partitions}
    for path in partition_dir.iterdir():
        if path.name not in expected:
            path.unlink()
    for partition in checkpoint.partitions:
        path = partition_dir / partition.file_name
        _require_regular_file(path, f"staged partition {path.name}")
        payload_hash, byte_count = _hash_file(path)
        if payload_hash != partition.file_sha256 or byte_count != partition.byte_count:
            raise ValueError(f"staged partition failed checksum verification: {path.name}")


def _write_checkpoint(stage: Path, checkpoint: ChunkedImportCheckpoint) -> None:
    _atomic_json_write(
        stage / _CHECKPOINT_NAME,
        {
            "checkpoint": checkpoint.model_dump(mode="json"),
            "spec_hash": checkpoint.spec.spec_hash,
        },
    )


def _load_checkpoint(stage: Path) -> ChunkedImportCheckpoint:
    path = stage / _CHECKPOINT_NAME
    _require_regular_file(path, "chunked import checkpoint")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"no resumable chunked import exists at {stage}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("chunked import checkpoint is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("chunked import checkpoint must be a JSON object")
    checkpoint = ChunkedImportCheckpoint.model_validate(payload.get("checkpoint"))
    if payload.get("spec_hash") != checkpoint.spec.spec_hash:
        raise ValueError("chunked import checkpoint specification hash does not match")
    return checkpoint


def _read_manifest_file(path: Path) -> ChunkedImportManifest:
    _require_regular_file(path, "chunked table manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"chunked table manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("chunked table manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("chunked table manifest must be a JSON object")
    manifest = ChunkedImportManifest.model_validate(payload.get("manifest"))
    if payload.get("manifest_hash") != manifest.manifest_hash:
        raise ValueError("chunked table manifest hash does not match")
    return manifest


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_file(path: Path) -> tuple[str, int]:
    hasher = sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
            byte_count += len(block)
    return hasher.hexdigest(), byte_count


def _reject_symlink(path: Path, description: str) -> None:
    """Reject a final path component that could redirect store I/O."""
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symbolic link: {path}")


def _require_real_directory(path: Path, description: str) -> None:
    """Require a present directory whose final component is not a symlink."""
    _reject_symlink(path, description)
    if not path.is_dir():
        raise ValueError(f"{description} is missing or is not a directory: {path}")


def _require_regular_file(path: Path, description: str) -> None:
    """Require a present regular file whose final component is not a symlink."""
    _reject_symlink(path, description)
    if not path.is_file():
        raise ValueError(f"{description} is missing or is not a regular file: {path}")


def _update_length_prefixed(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _logical_store_hash(partitions: Sequence[ChunkedPartition]) -> str:
    return canonical_sha256(
        [
            {
                "index": item.index,
                "row_count": item.row_count,
                "logical_sha256": item.logical_sha256,
            }
            for item in partitions
        ]
    )


def _require_kind(spec: ChunkedImportSpec, expected: ChunkedTableKind) -> None:
    if spec.kind is not expected:
        raise ValueError(f"chunked helper requires kind={expected.value!r}")


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_unique_clean_strings(values: Sequence[str], field_name: str) -> None:
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")


def _import_pyarrow() -> Any:
    try:
        return importlib.import_module("pyarrow")
    except ImportError as exc:
        raise ImportError(
            "PyArrow support requires the optional 'arrow' extra; install rejuvenationkit[arrow]"
        ) from exc


def _import_pyarrow_parquet() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires the optional 'arrow' extra; install rejuvenationkit[arrow]"
        ) from exc
