from __future__ import annotations

import importlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

import rejuvenationkit.genomics.chunked as chunked_module
from rejuvenationkit.genomics.chunked import (
    ChunkedImportSpec,
    ChunkedStorageFormat,
    ChunkedTableKind,
    DuplicatePolicy,
    TableColumn,
    iter_chunked_partitions,
    iter_csv_chunks,
    iter_parquet_chunks,
    read_chunked_checkpoint,
    read_chunked_manifest,
    write_chunked_annotations,
    write_chunked_network,
    write_chunked_table,
    write_chunked_variants,
)
from rejuvenationkit.genomics.resources import FeatureDomain, QueryProvenance, ResourceSnapshot
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType


def domain() -> FeatureDomain:
    return FeatureDomain(
        species_taxon_id=9615,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        genome_assembly="CanFam4",
    )


def resource() -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id="gene-ontology",
        resource_id="goa-dog",
        resource_release="2026-07-01",
        retrieved_at=datetime(2026, 8, 20, 12, tzinfo=UTC),
        response_sha256="a" * 64,
        source_uri="https://example.org/goa.tsv",
        license_id="CC-BY-4.0",
    )


def provenance(
    *,
    selected_domain: FeatureDomain | None = None,
    selected_resource: ResourceSnapshot | None = None,
) -> QueryProvenance:
    selected_resource = selected_resource or resource()
    return QueryProvenance(
        provider_id="go-toolkit",
        provider_version="1.0.0",
        resources=(selected_resource,),
        domain=selected_domain or domain(),
        retrieved_at=datetime(2026, 8, 20, 12, tzinfo=UTC),
        input_hash="b" * 64,
        query_parameters={"taxon_id": 9615},
        software_versions={"go-toolkit": "1.0.0"},
    )


def spec(**updates: object) -> ChunkedImportSpec:
    selected_resource = resource()
    values: dict[str, object] = {
        "dataset_id": "canine-go-annotations",
        "kind": ChunkedTableKind.FUNCTIONAL_ANNOTATION,
        "domain": domain(),
        "resource": selected_resource,
        "provenance": provenance(selected_resource=selected_resource),
        "key_columns": ("feature_id", "term_id"),
        "duplicate_policy": DuplicatePolicy.ERROR,
        "column_roles": {"feature": "feature_id", "term": "term_id"},
        "rows_per_partition": 2,
    }
    values.update(updates)
    return ChunkedImportSpec(**values)


def frame(ids: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": pd.Series([f"ENSCAFG{value}" for value in ids], dtype="object"),
            "term_id": pd.Series([f"GO:{value:07d}" for value in ids], dtype="object"),
            "score": pd.Series([float(value) / 10 for value in ids], dtype="float64"),
        }
    )


def test_chunked_import_is_partition_deterministic_across_input_chunk_boundaries(
    tmp_path: Path,
) -> None:
    source = frame()
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"

    first = write_chunked_annotations(
        (source.iloc[:1], source.iloc[1:4], source.iloc[4:]), first_path, spec=spec()
    )
    second = write_chunked_annotations((source,), second_path, spec=spec())

    assert [item.row_count for item in first.partitions] == [2, 2, 1]
    assert first.logical_content_sha256 == second.logical_content_sha256
    assert [item.logical_sha256 for item in first.partitions] == [
        item.logical_sha256 for item in second.partitions
    ]
    assert [item.file_sha256 for item in first.partitions] == [
        item.file_sha256 for item in second.partitions
    ]
    assert first.manifest_hash == second.manifest_hash
    assert first.spec.domain == domain()
    assert first.spec.resource.snapshot_id == resource().snapshot_id
    assert first.spec.provenance.query_hash == provenance().query_hash
    assert first.fusion_eligibility == "not_fusible"


def test_partition_iterator_reconstructs_rows_one_bounded_partition_at_a_time(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "annotations"
    write_chunked_table((frame(),), destination, spec=spec())

    partitions = list(iter_chunked_partitions(destination))
    assert [len(item) for item in partitions] == [2, 2, 1]
    reconstructed = pd.concat(partitions, ignore_index=True)
    pd.testing.assert_frame_equal(reconstructed, frame())
    verified = read_chunked_manifest(destination)
    assert verified.retained_row_count == 5


def test_drop_later_uses_cross_partition_duplicate_policy_and_audit(tmp_path: Path) -> None:
    repeated = pd.concat([frame((1, 2)), frame((2, 3)), frame((1, 4))], ignore_index=True)
    destination = tmp_path / "drop-later"
    manifest = write_chunked_table(
        (repeated.iloc[:2], repeated.iloc[2:4], repeated.iloc[4:]),
        destination,
        spec=spec(duplicate_policy=DuplicatePolicy.DROP_LATER),
    )

    assert manifest.source_row_count == 6
    assert manifest.retained_row_count == 4
    assert manifest.duplicate_row_count == 2
    reconstructed = pd.concat(iter_chunked_partitions(destination), ignore_index=True)
    assert reconstructed["feature_id"].tolist() == [
        "ENSCAFG1",
        "ENSCAFG2",
        "ENSCAFG3",
        "ENSCAFG4",
    ]


def test_error_duplicate_policy_rejects_repeat_without_finalizing(tmp_path: Path) -> None:
    destination = tmp_path / "duplicates"
    with pytest.raises(ValueError, match="duplicate archived-table key"):
        write_chunked_table((frame((1, 2, 2)),), destination, spec=spec())

    assert not destination.exists()
    checkpoint = read_chunked_checkpoint(destination)
    assert checkpoint.source_row_count == 2
    assert checkpoint.retained_row_count == 2


def test_allow_duplicate_policy_retains_all_rows_without_key_columns(tmp_path: Path) -> None:
    destination = tmp_path / "allow"
    allow = spec(key_columns=(), duplicate_policy=DuplicatePolicy.ALLOW, column_roles={})
    manifest = write_chunked_table((frame((1, 1, 2)),), destination, spec=allow)

    assert manifest.source_row_count == 3
    assert manifest.retained_row_count == 3
    assert manifest.duplicate_row_count == 0
    assert all(item.first_key_sha256 is None for item in manifest.partitions)


def test_resume_starts_from_committed_source_count_and_rebuilds_disk_key_index(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "resumable"

    def interrupted() -> Iterator[pd.DataFrame]:
        yield frame((1, 2))
        raise RuntimeError("simulated source interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        write_chunked_table(interrupted(), destination, spec=spec())
    checkpoint = read_chunked_checkpoint(destination)
    assert checkpoint.source_row_count == 2
    assert checkpoint.retained_row_count == 2
    assert not destination.exists()

    manifest = write_chunked_table((frame((3, 4, 5)),), destination, spec=spec(), resume=True)
    assert manifest.source_row_count == 5
    reconstructed = pd.concat(iter_chunked_partitions(destination), ignore_index=True)
    pd.testing.assert_frame_equal(reconstructed, frame())


def test_resume_rejects_a_different_scientific_or_storage_spec(tmp_path: Path) -> None:
    destination = tmp_path / "mismatch"

    def interrupted() -> Iterator[pd.DataFrame]:
        yield frame((1, 2))
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        write_chunked_table(interrupted(), destination, spec=spec())
    with pytest.raises(ValueError, match="does not match"):
        write_chunked_table(
            (),
            destination,
            spec=spec(rows_per_partition=3),
            resume=True,
        )


def test_schema_consistency_and_exact_expected_schema_are_enforced(tmp_path: Path) -> None:
    destination = tmp_path / "schema-change"
    changed = frame((3, 4)).assign(score=lambda value: value["score"].astype("float32"))
    with pytest.raises(ValueError, match="schema differs"):
        write_chunked_table((frame((1, 2)), changed), destination, spec=spec())

    expected = (
        TableColumn(name="feature_id", pandas_dtype="object"),
        TableColumn(name="term_id", pandas_dtype="object"),
        TableColumn(name="score", pandas_dtype="float64"),
    )
    empty_path = tmp_path / "empty"
    manifest = write_chunked_table((), empty_path, spec=spec(expected_schema=expected))
    assert manifest.partitions == ()
    assert manifest.schema_definition == expected


def test_spec_rejects_unbound_resource_domain_and_missing_keys() -> None:
    with pytest.raises(ValidationError, match="raw-byte resource SHA-256"):
        unhashable = resource().model_copy(update={"response_sha256": None})
        spec(resource=unhashable, provenance=provenance(selected_resource=unhashable))
    mouse = domain().model_copy(update={"species_taxon_id": 10090})
    with pytest.raises(ValidationError, match="domain must match"):
        spec(domain=mouse)
    with pytest.raises(ValidationError, match="require key_columns"):
        spec(key_columns=())
    with pytest.raises(ValidationError, match="absent"):
        spec(
            key_columns=("missing",),
            expected_schema=(TableColumn(name="x", pandas_dtype="object"),),
        )


def test_kind_specific_wrappers_fail_closed(tmp_path: Path) -> None:
    network_spec = spec(kind=ChunkedTableKind.INTERACTION_NETWORK)
    variant_spec = spec(kind=ChunkedTableKind.VARIANT_ANNOTATION)
    assert (
        write_chunked_network((frame(),), tmp_path / "network", spec=network_spec).spec.kind
        is ChunkedTableKind.INTERACTION_NETWORK
    )
    assert (
        write_chunked_variants((frame(),), tmp_path / "variants", spec=variant_spec).spec.kind
        is ChunkedTableKind.VARIANT_ANNOTATION
    )
    with pytest.raises(ValueError, match="requires kind"):
        write_chunked_annotations((frame(),), tmp_path / "wrong", spec=network_spec)


def test_partition_tampering_is_detected_before_use(tmp_path: Path) -> None:
    destination = tmp_path / "tampered"
    manifest = write_chunked_table((frame(),), destination, spec=spec())
    partition_path = destination / "partitions" / manifest.partitions[0].file_name
    partition_path.write_bytes(partition_path.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="partition bytes"):
        read_chunked_manifest(destination)


def test_chunked_store_readers_reject_symlinked_integrity_paths(tmp_path: Path) -> None:
    real_store = tmp_path / "real-store"
    manifest = write_chunked_table((frame(),), real_store, spec=spec())

    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_store, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        read_chunked_manifest(root_link)

    manifest_path = real_store / "manifest.json"
    real_manifest = real_store / "manifest.real.json"
    manifest_path.rename(real_manifest)
    manifest_path.symlink_to(real_manifest.name)
    with pytest.raises(ValueError, match="symbolic link"):
        read_chunked_manifest(real_store)
    manifest_path.unlink()
    real_manifest.rename(manifest_path)

    partition_path = real_store / "partitions" / manifest.partitions[0].file_name
    real_partition = partition_path.with_suffix(f"{partition_path.suffix}.real")
    partition_path.rename(real_partition)
    partition_path.symlink_to(real_partition.name)
    with pytest.raises(ValueError, match="symbolic link"):
        list(iter_chunked_partitions(real_store, verify=False))


def test_chunked_resume_rejects_predictable_staging_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "redirected"
    external = tmp_path / "attacker-controlled"
    external.mkdir()
    stage = tmp_path / ".redirected.rejuvenationkit-staging"
    stage.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="staging directory must not be a symbolic link"):
        write_chunked_table((), destination, spec=spec(), resume=True)
    assert not destination.exists()


def test_local_csv_reader_is_chunked_and_disallows_stream_override(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    frame().to_csv(source, sep="\t", index=False)

    chunks = list(iter_csv_chunks(source, chunk_rows=2, read_csv_options={"sep": "\t"}))
    assert [len(item) for item in chunks] == [2, 2, 1]
    with pytest.raises(ValueError, match="streaming controls"):
        list(iter_csv_chunks(source, read_csv_options={"chunksize": 1}))


def test_pyarrow_is_lazy_and_reports_the_required_optional_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = importlib.import_module

    def fail_pyarrow(name: str, package: str | None = None) -> object:
        if name.startswith("pyarrow"):
            raise ImportError("not installed")
        return original_import(name, package)

    monkeypatch.setattr(chunked_module.importlib, "import_module", fail_pyarrow)
    iterator = iter_parquet_chunks(tmp_path / "missing.parquet")
    with pytest.raises(ImportError, match=r"rejuvenationkit\[arrow\]"):
        next(iterator)
    parquet_spec = spec(storage_format=ChunkedStorageFormat.PARQUET)
    with pytest.raises(ImportError, match=r"rejuvenationkit\[arrow\]"):
        write_chunked_table((frame(),), tmp_path / "parquet", spec=parquet_spec)


def test_pyarrow_parquet_write_read_and_bounded_source_round_trip(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    destination = tmp_path / "parquet-store"
    parquet_spec = spec(storage_format=ChunkedStorageFormat.PARQUET)

    manifest = write_chunked_table((frame(),), destination, spec=parquet_spec)
    verified = read_chunked_manifest(destination)
    reconstructed = pd.concat(iter_chunked_partitions(destination), ignore_index=True)

    assert verified == manifest
    assert manifest.writer_versions["pyarrow"] == pyarrow.__version__
    assert [item.file_name for item in manifest.partitions] == [
        "part-000000.parquet",
        "part-000001.parquet",
        "part-000002.parquet",
    ]
    pd.testing.assert_frame_equal(reconstructed, frame())

    source = tmp_path / "source.parquet"
    source_frame = frame()
    source_frame["term_id"] = source_frame["term_id"].astype("string")
    source_frame.to_parquet(source, engine="pyarrow", index=False)
    source_chunks = list(iter_parquet_chunks(source, batch_rows=2))
    assert [len(item) for item in source_chunks] == [2, 2, 1]
    pd.testing.assert_frame_equal(pd.concat(source_chunks, ignore_index=True), source_frame)


def test_pyarrow_parquet_resume_and_tamper_detection(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    destination = tmp_path / "parquet-resume"
    parquet_spec = spec(storage_format=ChunkedStorageFormat.PARQUET)

    def interrupted() -> Iterator[pd.DataFrame]:
        yield frame((1, 2))
        raise RuntimeError("simulated parquet interruption")

    with pytest.raises(RuntimeError, match="simulated parquet interruption"):
        write_chunked_table(interrupted(), destination, spec=parquet_spec)
    checkpoint = read_chunked_checkpoint(destination)
    assert checkpoint.source_row_count == 2
    assert checkpoint.retained_row_count == 2
    assert checkpoint.partitions[0].file_name.endswith(".parquet")

    manifest = write_chunked_table(
        (frame((3, 4, 5)),),
        destination,
        spec=parquet_spec,
        resume=True,
    )
    reconstructed = pd.concat(iter_chunked_partitions(destination), ignore_index=True)
    pd.testing.assert_frame_equal(reconstructed, frame())

    partition_path = destination / "partitions" / manifest.partitions[0].file_name
    partition_path.write_bytes(partition_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="partition bytes"):
        read_chunked_manifest(destination)
