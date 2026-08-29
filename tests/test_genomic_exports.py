from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.exports import (
    ExportLicense,
    ExportLicensePolicy,
    ExportPageReceipt,
    ExternalExportProvider,
    PaginationAudit,
    PaginationMode,
    ProviderExportEnvelope,
    build_ensembl_export,
    build_go_export,
    build_sra_export,
    build_string_export,
    load_export_archive,
    save_export_archive,
    single_response_pagination,
)
from rejuvenationkit.genomics.resources import FeatureDomain, canonical_sha256
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType

RAW = b"feature\tterm\nENSCAFG1\tGO:0006914\n"


def domain() -> FeatureDomain:
    return FeatureDomain(
        species_taxon_id=9615,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        genome_assembly="CanFam4",
    )


def metadata(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "resource_id": "go-annotations",
        "resource_release": "2026-07-01",
        "source_uri": "https://example.org/goa.tsv",
        "retrieved_at": datetime(2026, 8, 20, 12, tzinfo=UTC),
        "license": ExportLicense.spdx("CC-BY-4.0"),
        "connector_id": "go-toolkit",
        "connector_version": "1.0.0",
        "software_versions": {"archive-client": "2.3.1"},
        "input_hash": "a" * 64,
        "request_parameters": {"taxon_id": 9615, "aspects": ["biological_process"]},
        "raw_media_type": "text/tab-separated-values",
        "pagination": single_response_pagination(RAW, item_count=1),
    }
    values.update(updates)
    return values


@pytest.mark.parametrize(
    ("builder", "provider"),
    [
        (build_go_export, ExternalExportProvider.GENE_ONTOLOGY),
        (build_string_export, ExternalExportProvider.STRING),
        (build_ensembl_export, ExternalExportProvider.ENSEMBL),
        (build_sra_export, ExternalExportProvider.NCBI_SRA),
    ],
)
def test_provider_helpers_bind_raw_bytes_and_core_provenance(
    builder: object, provider: object
) -> None:
    envelope = builder(RAW, **metadata())

    assert envelope.provider is provider
    assert envelope.raw_byte_count == len(RAW)
    assert envelope.raw_bytes_sha256 == sha256(RAW).hexdigest()
    assert len(envelope.request_hash) == 64
    assert len(envelope.envelope_hash) == 64
    snapshot = envelope.to_resource_snapshot()
    assert snapshot.response_sha256 == sha256(RAW).hexdigest()
    assert snapshot.resource_release == "2026-07-01"
    assert snapshot.license_id == "CC-BY-4.0"
    provenance = envelope.to_query_provenance(domain())
    assert provenance.input_hash == "a" * 64
    assert provenance.complete
    assert provenance.response_checksum is None
    assert provenance.software_versions["go-toolkit"] == "1.0.0"


def test_request_hash_is_independent_of_mapping_order_and_sensitive_to_request() -> None:
    first = build_go_export(RAW, **metadata())
    reordered = build_go_export(
        RAW,
        **metadata(request_parameters={"aspects": ["biological_process"], "taxon_id": 9615}),
    )
    changed = build_go_export(RAW, **metadata(request_parameters={"taxon_id": 9606}))

    assert first.request_hash == reordered.request_hash
    assert first.request_hash != changed.request_hash
    with pytest.raises(ValidationError, match="does not match"):
        ProviderExportEnvelope.model_validate(
            {**first.model_dump(mode="python"), "request_hash": "f" * 64}
        )


def test_license_must_be_spdx_or_explicitly_unknown_with_warning() -> None:
    assert ExportLicense.spdx("Apache-2.0").policy is ExportLicensePolicy.SPDX
    with pytest.raises(ValidationError, match="requires an explicit warning"):
        ExportLicense(policy=ExportLicensePolicy.UNKNOWN_WITH_WARNING)
    with pytest.raises(ValidationError, match="unknown licenses require"):
        ExportLicense.spdx("NOASSERTION")

    unknown = build_ensembl_export(
        RAW,
        **metadata(
            license=ExportLicense.unknown(
                "No authoritative license statement was retained with this export."
            )
        ),
    )
    assert unknown.to_resource_snapshot().license_id is None
    assert "upstream_license_unknown" in unknown.warnings


def test_incomplete_paginated_export_is_fail_visible() -> None:
    page = ExportPageReceipt(
        sequence=1,
        request_marker_sha256=canonical_sha256({"cursor": "redacted-by-hash"}),
        response_sha256=sha256(RAW).hexdigest(),
        byte_count=len(RAW),
        item_count=25,
        provider_reported_more=True,
    )
    pagination = PaginationAudit(
        mode=PaginationMode.CURSOR,
        pages=(page,),
        complete=False,
        truncated=True,
        provider_reported_total_items=100,
    )
    envelope = build_string_export(RAW, **metadata(pagination=pagination))

    assert not envelope.pagination.complete
    assert "provider_export_incomplete" in envelope.warnings
    assert "provider_export_truncated" in envelope.warnings
    assert not envelope.to_query_provenance(domain()).complete


def test_paginated_export_binds_every_receipt_to_ordered_archived_bytes() -> None:
    pages = (b"first-page\n", b"second-page\n")
    pagination = PaginationAudit(
        mode=PaginationMode.CURSOR,
        pages=tuple(
            ExportPageReceipt(
                sequence=index,
                request_marker_sha256=canonical_sha256({"page": index}),
                response_sha256=sha256(payload).hexdigest(),
                byte_count=len(payload),
                item_count=1,
                provider_reported_more=index < len(pages),
            )
            for index, payload in enumerate(pages, start=1)
        ),
        complete=True,
        provider_reported_total_items=2,
    )
    raw = b"".join(pages)
    envelope = build_string_export(raw, **metadata(pagination=pagination))
    assert envelope.raw_byte_count == sum(page.byte_count for page in pagination.pages)

    wrong_hash = pagination.model_copy(
        update={
            "pages": (
                pagination.pages[0].model_copy(update={"response_sha256": "f" * 64}),
                pagination.pages[1],
            )
        }
    )
    with pytest.raises(ValueError, match="pagination receipt for page 1"):
        build_string_export(raw, **metadata(pagination=wrong_hash))

    wrong_boundary = pagination.model_copy(
        update={
            "pages": (
                pagination.pages[0].model_copy(
                    update={"byte_count": pagination.pages[0].byte_count + 1}
                ),
                pagination.pages[1].model_copy(
                    update={"byte_count": pagination.pages[1].byte_count - 1}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="pagination receipt for page 1"):
        build_string_export(raw, **metadata(pagination=wrong_boundary))


def test_pagination_rejects_gaps_and_false_completion() -> None:
    receipt = ExportPageReceipt(
        sequence=2,
        request_marker_sha256="a" * 64,
        response_sha256="b" * 64,
        byte_count=5,
        item_count=2,
        provider_reported_more=False,
    )
    with pytest.raises(ValidationError, match="contiguous"):
        PaginationAudit(mode=PaginationMode.CURSOR, pages=(receipt,), complete=True)
    with pytest.raises(ValidationError, match="exact archived raw bytes"):
        build_go_export(
            RAW,
            **metadata(pagination=single_response_pagination(b"different", item_count=1)),
        )

    first_stops_early = receipt.model_copy(
        update={"sequence": 1, "request_marker_sha256": "c" * 64}
    )
    second = receipt.model_copy(update={"request_marker_sha256": "d" * 64})
    with pytest.raises(ValidationError, match="non-final page"):
        PaginationAudit(
            mode=PaginationMode.CURSOR,
            pages=(first_stops_early, second),
            complete=True,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"request_parameters": {"filters": {"api_token": "secret"}}},
        {"source_uri": "https://user:password@example.org/export"},
        {"source_uri": "https://example.org/export?access_token=secret"},
        {"software_versions": {"client_secret": "1.0.0"}},
    ],
)
def test_export_rejects_secret_like_metadata(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=r"secret|credentials"):
        build_go_export(RAW, **metadata(**updates))


@pytest.mark.parametrize("field", ["resource_release", "connector_version"])
def test_export_requires_resolved_upstream_and_connector_versions(field: str) -> None:
    with pytest.raises(ValidationError, match="exact resolved"):
        build_go_export(RAW, **metadata(**{field: "latest"}))


def test_export_archive_round_trip_is_offline_and_no_clobber(tmp_path: Path) -> None:
    envelope = build_go_export(RAW, **metadata())
    archive_path = tmp_path / "go-export.rkexport"

    assert save_export_archive(archive_path, envelope, RAW) == archive_path
    loaded, loaded_raw = load_export_archive(archive_path)
    assert loaded == envelope
    assert loaded_raw == RAW
    with pytest.raises(FileExistsError, match="already exists"):
        save_export_archive(archive_path, envelope, RAW)
    with pytest.raises(ValueError, match="maximum_raw_bytes"):
        load_export_archive(archive_path, maximum_raw_bytes=len(RAW) - 1)


def test_v1_archive_loads_only_through_the_current_byte_receipt_contract(tmp_path: Path) -> None:
    envelope = build_go_export(RAW, **metadata())
    archive_path = tmp_path / "legacy-v1.rkexport"
    payload = {
        "archive_format": "rejuvenationkit-provider-export/v1",
        "envelope": envelope.model_dump(mode="json"),
        "envelope_hash": envelope.envelope_hash,
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(payload))
        archive.writestr("raw-response.bin", RAW)

    loaded, loaded_raw = load_export_archive(archive_path)

    assert loaded == envelope
    assert loaded_raw == RAW


def test_export_archive_rejects_mismatched_or_tampered_bytes(tmp_path: Path) -> None:
    envelope = build_go_export(RAW, **metadata())
    with pytest.raises(ValueError, match=r"byte count|SHA-256"):
        save_export_archive(tmp_path / "bad.rkexport", envelope, b"wrong")
    assert not (tmp_path / "bad.rkexport").exists()

    valid = tmp_path / "valid.rkexport"
    save_export_archive(valid, envelope, RAW)
    with zipfile.ZipFile(valid, "r") as archive:
        manifest = archive.read("manifest.json")
    tampered = tmp_path / "tampered.rkexport"
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("raw-response.bin", b"x" * len(RAW))
    with pytest.raises(ValueError, match="SHA-256"):
        load_export_archive(tampered)


def test_export_archive_rejects_manifest_hash_forgery(tmp_path: Path) -> None:
    envelope = build_go_export(RAW, **metadata())
    archive_path = tmp_path / "forged.rkexport"
    payload = {
        "archive_format": "rejuvenationkit-provider-export/v1",
        "envelope": envelope.model_dump(mode="json"),
        "envelope_hash": "0" * 64,
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(payload))
        archive.writestr("raw-response.bin", RAW)

    with pytest.raises(ValueError, match="envelope hash"):
        load_export_archive(archive_path)
