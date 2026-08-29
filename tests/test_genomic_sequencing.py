from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.resources import FeatureDomain, QueryProvenance, ResourceSnapshot
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType
from rejuvenationkit.genomics.sequencing import (
    ExternalSequencingFile,
    ExternalSequencingRun,
    ExternalSequencingSample,
    ExternalSequencingStudyManifest,
    SequencingChecksumAlgorithm,
    SequencingFileColumns,
    SequencingLibraryLayout,
    SequencingManifestRequest,
    SequencingSelectionParameter,
    SubjectMappingBasis,
    SubjectMappingDeclaration,
    SubjectMappingStatus,
    read_external_sequencing_manifest,
)


def domain(*, species_taxon_id: int = 9615) -> FeatureDomain:
    return FeatureDomain(
        species_taxon_id=species_taxon_id,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        genome_assembly="CanFam4" if species_taxon_id == 9615 else "GRCm39",
    )


def resource(
    *,
    release: str = "metadata-2026-08-01",
    raw_checksum: str = "c" * 64,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id="ncbi",
        resource_id="sra",
        resource_release=release,
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        response_sha256=raw_checksum,
        source_uri="https://www.ncbi.nlm.nih.gov/sra",
    )


def request(**updates: object) -> SequencingManifestRequest:
    values: dict[str, object] = {
        "study_accession": "PRJNA-DOG-AGING",
        "domain": domain(),
        "resource": resource(),
        "selection_parameters": (
            SequencingSelectionParameter(name="library_strategy", values=("RNA-Seq",)),
            SequencingSelectionParameter(name="tissue", values=("liver",)),
        ),
    }
    values.update(updates)
    return SequencingManifestRequest(**values)


def provenance(
    manifest_request: SequencingManifestRequest,
    *,
    input_hash: str | None = None,
    response_checksum: str | None = None,
    resources: tuple[ResourceSnapshot, ...] | None = None,
) -> QueryProvenance:
    return QueryProvenance(
        provider_id="smarts.bio/ncbi-sra",
        provider_version="1.0.0",
        resources=resources or (manifest_request.resource,),
        domain=manifest_request.domain,
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        input_hash=input_hash or manifest_request.input_hash,
        response_checksum=response_checksum,
        query_parameters={"operation": "study_manifest"},
    )


def verified_mapping() -> SubjectMappingDeclaration:
    return SubjectMappingDeclaration(
        status=SubjectMappingStatus.VERIFIED,
        basis=SubjectMappingBasis.STUDY_METADATA,
        source_id="PRJNA-DOG-AGING/subject-crosswalk.tsv",
        source_checksum="a" * 64,
        verification_method="two-person review against study metadata",
    )


def unverified_mapping() -> SubjectMappingDeclaration:
    return SubjectMappingDeclaration(
        status=SubjectMappingStatus.COMPLETE_UNVERIFIED,
        basis=SubjectMappingBasis.PROVIDER_METADATA,
        source_id="NCBI BioSample attributes",
    )


def incomplete_mapping() -> SubjectMappingDeclaration:
    return SubjectMappingDeclaration(
        status=SubjectMappingStatus.INCOMPLETE,
        basis=SubjectMappingBasis.STUDY_METADATA,
        source_id="PRJNA-DOG-AGING/partial-subject-map.tsv",
    )


def missing_mapping() -> SubjectMappingDeclaration:
    return SubjectMappingDeclaration(
        status=SubjectMappingStatus.NOT_PROVIDED,
        basis=SubjectMappingBasis.NONE,
    )


def frame(*, include_subjects: bool = True) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "study": ["PRJNA-DOG-AGING"] * 3,
        "sample": ["SAMN1", "SAMN1", "SAMN2"],
        "experiment": ["SRX1", "SRX1", "SRX2"],
        "run": ["SRR1", "SRR2", "SRR3"],
        "taxon": [9615, 9615, 9615],
        "strategy": ["RNA-Seq", "RNA-Seq", "RNA-Seq"],
        "layout": ["paired-ended", "paired-ended", "single-ended"],
        "tissue": ["liver", "liver", "liver"],
        "platform": ["ILLUMINA", "ILLUMINA", "ILLUMINA"],
        "breed": ["mixed", "mixed", "beagle"],
    }
    if include_subjects:
        values["subject"] = ["dog-1", "dog-1", "dog-2"]
    return pd.DataFrame(values)


def read(
    frame_value: pd.DataFrame,
    *,
    subject_column: str | None,
    declaration: SubjectMappingDeclaration,
    manifest_request: SequencingManifestRequest | None = None,
    query_provenance: QueryProvenance | None = None,
) -> ExternalSequencingStudyManifest:
    selected_request = manifest_request or request()
    selected_provenance = query_provenance or provenance(selected_request)
    return read_external_sequencing_manifest(
        frame_value,
        request=selected_request,
        provenance=selected_provenance,
        subject_mapping=declaration,
        study_accession_column="study",
        sample_accession_column="sample",
        experiment_accession_column="experiment",
        run_accession_column="run",
        species_taxon_id_column="taxon",
        library_strategy_column="strategy",
        library_layout_column="layout",
        subject_id_column=subject_column,
        tissue_column="tissue",
        platform_column="platform",
        sample_attribute_columns={"breed": "breed"},
    )


def test_manifest_preserves_hierarchy_and_binds_both_checksum_layers() -> None:
    manifest = read(frame(), subject_column="subject", declaration=verified_mapping())

    assert manifest.study_accession == "PRJNA-DOG-AGING"
    assert len(manifest.samples) == 2
    assert len(manifest.experiments) == 2
    assert len(manifest.runs) == 3
    assert manifest.subject_mapping_complete
    assert manifest.subject_mapping_verified
    assert manifest.independent_subject_count == 2
    assert manifest.samples[0].attributes == {"breed": "mixed"}
    assert [item.library_layout for item in manifest.experiments] == [
        SequencingLibraryLayout.PAIRED,
        SequencingLibraryLayout.SINGLE,
    ]
    assert manifest.resource.response_sha256 == "c" * 64
    assert manifest.provenance.response_checksum == manifest.normalized_manifest_checksum
    assert manifest.normalized_manifest_checksum != manifest.resource.response_sha256
    assert manifest.fusion_eligibility == "not_fusible"
    assert not hasattr(manifest, "to_evidence")


def test_sample_attributes_are_copied_immutable_and_round_trip_serializable() -> None:
    supplied = {"breed": "mixed"}
    sample = ExternalSequencingSample(
        sample_accession="SAMN1",
        subject_id="dog-1",
        species_taxon_id=9615,
        attributes=supplied,
    )
    supplied["breed"] = "beagle"

    assert sample.attributes == {"breed": "mixed"}
    with pytest.raises(TypeError):
        sample.attributes["breed"] = "beagle"  # type: ignore[index]
    assert ExternalSequencingSample.model_validate(sample.model_dump()) == sample


def test_manifest_without_subject_ids_never_assumes_independent_subjects() -> None:
    manifest = read(
        frame(include_subjects=False),
        subject_column=None,
        declaration=missing_mapping(),
    )

    assert not manifest.subject_mapping_complete
    assert not manifest.subject_mapping_verified
    assert manifest.independent_subject_count is None
    assert manifest.warnings == ("independent_subject_mapping_incomplete",)
    with pytest.raises(ValueError, match="cannot be assumed"):
        manifest.require_subject_mapping()


def test_complete_but_unverified_mapping_does_not_report_subject_count() -> None:
    manifest = read(
        frame(),
        subject_column="subject",
        declaration=unverified_mapping(),
    )

    assert manifest.subject_mapping_complete
    assert not manifest.subject_mapping_verified
    assert manifest.independent_subject_count is None
    assert manifest.warnings == ("independent_subject_mapping_unverified",)
    with pytest.raises(ValueError, match="complete but unverified"):
        manifest.require_subject_mapping()


def test_partial_mapping_requires_an_explicit_incomplete_declaration() -> None:
    partial = frame()
    partial.loc[2, "subject"] = pd.NA

    manifest = read(
        partial,
        subject_column="subject",
        declaration=incomplete_mapping(),
    )
    assert manifest.independent_subject_count is None
    assert manifest.warnings == ("independent_subject_mapping_incomplete",)

    with pytest.raises(ValueError, match="status must be 'incomplete'"):
        read(partial, subject_column="subject", declaration=unverified_mapping())


def test_subject_mapping_declaration_requires_source_and_verification_evidence() -> None:
    with pytest.raises(ValidationError, match="basis and source_id"):
        SubjectMappingDeclaration(
            status=SubjectMappingStatus.COMPLETE_UNVERIFIED,
            basis=SubjectMappingBasis.NONE,
        )
    with pytest.raises(ValidationError, match="requires verification_method"):
        SubjectMappingDeclaration(
            status=SubjectMappingStatus.VERIFIED,
            basis=SubjectMappingBasis.PUBLICATION,
            source_id="PMID:123",
        )
    with pytest.raises(ValidationError, match="only a verified"):
        SubjectMappingDeclaration(
            status=SubjectMappingStatus.INCOMPLETE,
            basis=SubjectMappingBasis.STUDY_METADATA,
            source_id="partial-map.tsv",
            verification_method="not actually complete",
        )


def test_request_hash_binds_study_domain_resource_and_selection_parameters() -> None:
    baseline = request()
    reordered = request(selection_parameters=tuple(reversed(baseline.selection_parameters)))

    assert baseline.input_hash == reordered.input_hash
    assert baseline.input_hash != request(study_accession="PRJNA-OTHER").input_hash
    assert baseline.input_hash != request(domain=domain(species_taxon_id=10090)).input_hash
    assert (
        baseline.input_hash != request(resource=resource(release="metadata-2026-08-02")).input_hash
    )
    assert (
        baseline.input_hash
        != request(
            selection_parameters=(
                SequencingSelectionParameter(name="library_strategy", values=("WGS",)),
            )
        ).input_hash
    )


def test_selection_parameters_are_immutable_canonical_and_secret_free() -> None:
    parameter = SequencingSelectionParameter(name="tissue", values=("muscle", "liver"))
    assert parameter.values == ("liver", "muscle")
    with pytest.raises(ValidationError, match="frozen"):
        parameter.name = "organ"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="secret-like"):
        SequencingSelectionParameter(name="api_token", values=("do-not-store",))
    with pytest.raises(ValidationError, match="unique"):
        SequencingSelectionParameter(name="tissue", values=("liver", "liver"))


def test_loader_requires_provenance_input_hash_and_resource_to_match_request() -> None:
    manifest_request = request()
    wrong_hash = provenance(manifest_request, input_hash="d" * 64)
    with pytest.raises(ValueError, match="request hash must match"):
        read(
            frame(),
            subject_column="subject",
            declaration=verified_mapping(),
            manifest_request=manifest_request,
            query_provenance=wrong_hash,
        )

    other_resource = resource(release="metadata-2026-08-02")
    missing_resource = provenance(manifest_request, resources=(other_resource,))
    with pytest.raises(ValueError, match="captured in query provenance"):
        read(
            frame(),
            subject_column="subject",
            declaration=verified_mapping(),
            manifest_request=manifest_request,
            query_provenance=missing_resource,
        )


def test_normalized_manifest_round_trip_rejects_content_or_checksum_forgery() -> None:
    manifest = read(frame(), subject_column="subject", declaration=verified_mapping())
    payload = manifest.model_dump(mode="python")
    reproduced = ExternalSequencingStudyManifest.model_validate(payload)
    assert reproduced.normalized_manifest_checksum == manifest.normalized_manifest_checksum

    altered_content = manifest.model_dump(mode="python")
    altered_content["samples"][0]["tissue"] = "blood"
    with pytest.raises(ValidationError, match="normalized manifest"):
        ExternalSequencingStudyManifest.model_validate(altered_content)

    altered_checksum = manifest.model_dump(mode="python")
    altered_checksum["provenance"]["response_checksum"] = "f" * 64
    with pytest.raises(ValidationError, match="normalized manifest"):
        ExternalSequencingStudyManifest.model_validate(altered_checksum)


def test_loader_rejects_predeclared_response_checksum_that_does_not_match() -> None:
    manifest_request = request()
    forged = provenance(manifest_request, response_checksum="f" * 64)
    with pytest.raises(ValueError, match="normalized manifest"):
        read(
            frame(),
            subject_column="subject",
            declaration=verified_mapping(),
            manifest_request=manifest_request,
            query_provenance=forged,
        )


def test_public_manifest_construction_rejects_orphan_samples_and_experiments() -> None:
    manifest = read(frame(), subject_column="subject", declaration=verified_mapping())
    orphan_sample = manifest.model_dump(mode="python")
    orphan_sample["samples"] = (
        *orphan_sample["samples"],
        {
            "sample_accession": "SAMN9",
            "subject_id": "dog-9",
            "species_taxon_id": 9615,
            "tissue": "liver",
            "attributes": {},
        },
    )
    with pytest.raises(ValidationError, match="orphan samples"):
        ExternalSequencingStudyManifest.model_validate(orphan_sample)

    orphan_experiment = manifest.model_dump(mode="python")
    orphan_experiment["experiments"] = (
        *orphan_experiment["experiments"],
        {
            "experiment_accession": "SRX9",
            "sample_accession": "SAMN2",
            "library_strategy": "RNA-Seq",
            "library_source": None,
            "library_selection": None,
            "library_layout": "paired",
            "platform": "ILLUMINA",
        },
    )
    with pytest.raises(ValidationError, match="orphan experiments"):
        ExternalSequencingStudyManifest.model_validate(orphan_experiment)


def test_manifest_rejects_conflicting_metadata_for_one_biosample() -> None:
    conflicting = frame()
    conflicting.loc[1, "tissue"] = "blood"

    with pytest.raises(ValueError, match="conflicting sample metadata"):
        read(conflicting, subject_column="subject", declaration=verified_mapping())


def test_manifest_rejects_every_duplicate_run_accession() -> None:
    duplicate = frame()
    duplicate.loc[2, "run"] = "SRR1"
    with pytest.raises(ValueError, match="duplicate sequencing run accession"):
        read(duplicate, subject_column="subject", declaration=verified_mapping())

    exact_duplicate = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sequencing run accession"):
        read(exact_duplicate, subject_column="subject", declaration=verified_mapping())


def test_manifest_rejects_rows_from_a_different_study() -> None:
    conflicting = frame()
    conflicting.loc[2, "study"] = "PRJNA-OTHER"

    with pytest.raises(ValueError, match="does not match requested study"):
        read(conflicting, subject_column="subject", declaration=verified_mapping())


@pytest.mark.parametrize("bad_taxon", [9615.5, True, False, float("nan")])
def test_loader_rejects_fractional_boolean_or_nonfinite_taxon_ids(bad_taxon: object) -> None:
    invalid = frame()
    invalid["taxon"] = invalid["taxon"].astype(object)
    invalid.loc[0, "taxon"] = bad_taxon

    with pytest.raises(ValueError, match="must be a positive integer"):
        read(invalid, subject_column="subject", declaration=verified_mapping())


def test_sample_model_itself_uses_strict_positive_taxon_ids() -> None:
    with pytest.raises(ValidationError, match="valid integer"):
        ExternalSequencingSample(
            sample_accession="SAMN1",
            subject_id="dog-1",
            species_taxon_id=True,
        )


@pytest.mark.parametrize(
    "column",
    ["study", "sample", "experiment", "run", "taxon", "strategy", "layout"],
)
def test_loader_rejects_missing_required_cells(column: str) -> None:
    invalid = frame()
    invalid.loc[0, column] = pd.NA

    with pytest.raises(ValueError, match=r"cannot be missing|must be a positive integer"):
        read(invalid, subject_column="subject", declaration=verified_mapping())


def test_loader_rejects_duplicate_dataframe_columns_and_column_mappings() -> None:
    duplicated_columns = pd.concat([frame(), frame()[["run"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns"):
        read(duplicated_columns, subject_column="subject", declaration=verified_mapping())

    manifest_request = request()
    with pytest.raises(ValueError, match="column mappings must be unique"):
        read_external_sequencing_manifest(
            frame(),
            request=manifest_request,
            provenance=provenance(manifest_request),
            subject_mapping=verified_mapping(),
            study_accession_column="study",
            sample_accession_column="sample",
            experiment_accession_column="experiment",
            run_accession_column="run",
            species_taxon_id_column="taxon",
            library_strategy_column="strategy",
            library_layout_column="layout",
            subject_id_column="subject",
            tissue_column="strategy",
        )


def test_normalized_manifest_is_deterministic_across_input_row_order() -> None:
    first = read(frame(), subject_column="subject", declaration=verified_mapping())
    shuffled = frame().iloc[[2, 0, 1]].reset_index(drop=True)
    second = read(shuffled, subject_column="subject", declaration=verified_mapping())

    assert first.samples == second.samples
    assert first.experiments == second.experiments
    assert first.runs == second.runs
    assert first.normalized_manifest_checksum == second.normalized_manifest_checksum


def test_manifest_supports_multiple_integrity_bound_files_per_run() -> None:
    """Paired FASTQs remain separate files with explicit digest algorithms."""
    data = frame()
    data["read_1"] = [f"s3://archive/{run}_R1.fastq.gz" for run in data["run"]]
    data["read_2"] = [
        "s3://archive/SRR1_R2.fastq.gz",
        "s3://archive/SRR2_R2.fastq.gz",
        pd.NA,
    ]
    data["read_1_md5"] = ["a" * 32, "b" * 32, "c" * 32]
    data["read_2_md5"] = ["d" * 32, "e" * 32, pd.NA]
    manifest_request = request()
    manifest = read_external_sequencing_manifest(
        data,
        request=manifest_request,
        provenance=provenance(manifest_request),
        subject_mapping=verified_mapping(),
        study_accession_column="study",
        sample_accession_column="sample",
        experiment_accession_column="experiment",
        run_accession_column="run",
        species_taxon_id_column="taxon",
        library_strategy_column="strategy",
        library_layout_column="layout",
        subject_id_column="subject",
        tissue_column="tissue",
        platform_column="platform",
        sample_attribute_columns={"breed": "breed"},
        file_columns=(
            SequencingFileColumns(
                uri="read_1",
                checksum="read_1_md5",
                checksum_algorithm=SequencingChecksumAlgorithm.MD5,
                role="read_1",
            ),
            SequencingFileColumns(
                uri="read_2",
                checksum="read_2_md5",
                checksum_algorithm=SequencingChecksumAlgorithm.MD5,
                role="read_2",
            ),
        ),
    )

    assert len(manifest.runs[0].files) == 2
    assert manifest.runs[0].files[0].role == "read_1"
    assert manifest.runs[0].files[1].role == "read_2"
    assert manifest.runs[2].files[0].uri.endswith("SRR3_R1.fastq.gz")
    assert manifest.runs[2].files[0].checksum_algorithm is SequencingChecksumAlgorithm.MD5
    assert (
        ExternalSequencingStudyManifest.model_validate(manifest.model_dump(mode="python"))
        == manifest
    )


def test_legacy_one_file_fields_are_accepted_only_when_unambiguous() -> None:
    """The old fields normalize to the same explicit one-file contract."""
    run = ExternalSequencingRun(
        run_accession="SRR1",
        experiment_accession="SRX1",
        file_uri="https://archive.invalid/SRR1.fastq.gz",
        file_checksum="a" * 64,
        file_checksum_algorithm=SequencingChecksumAlgorithm.SHA256,
    )
    assert run.files == (
        ExternalSequencingFile(
            uri="https://archive.invalid/SRR1.fastq.gz",
            checksum="a" * 64,
            checksum_algorithm=SequencingChecksumAlgorithm.SHA256,
        ),
    )
    assert ExternalSequencingRun.model_validate(run.model_dump(mode="python")) == run

    with pytest.raises(ValidationError, match="supplied together"):
        ExternalSequencingRun(
            run_accession="SRR1",
            experiment_accession="SRX1",
            file_uri="https://archive.invalid/SRR1.fastq.gz",
            file_checksum="a" * 64,
        )

    read_1 = ExternalSequencingFile(uri="https://archive.invalid/R1.fastq.gz", role="read_1")
    read_2 = ExternalSequencingFile(uri="https://archive.invalid/R2.fastq.gz", role="read_2")
    forward = ExternalSequencingRun(
        run_accession="SRR2",
        experiment_accession="SRX1",
        files=(read_1, read_2),
    )
    reversed_input = ExternalSequencingRun(
        run_accession="SRR2",
        experiment_accession="SRX1",
        files=(read_2, read_1),
    )
    assert forward == reversed_input


def test_sequencing_file_checksums_are_algorithm_specific_and_anti_forgery_bound() -> None:
    """A mislabeled digest or post-import file mutation cannot validate."""
    with pytest.raises(ValidationError, match="exactly 64"):
        ExternalSequencingFile(
            uri="https://archive.invalid/read.fastq.gz",
            checksum="a" * 32,
            checksum_algorithm=SequencingChecksumAlgorithm.SHA256,
        )

    data = frame()
    data["file"] = [f"https://archive.invalid/{run}.fastq.gz" for run in data["run"]]
    data["sha256"] = ["a" * 64, "b" * 64, "c" * 64]
    manifest_request = request()
    manifest = read_external_sequencing_manifest(
        data,
        request=manifest_request,
        provenance=provenance(manifest_request),
        subject_mapping=verified_mapping(),
        study_accession_column="study",
        sample_accession_column="sample",
        experiment_accession_column="experiment",
        run_accession_column="run",
        species_taxon_id_column="taxon",
        library_strategy_column="strategy",
        library_layout_column="layout",
        subject_id_column="subject",
        tissue_column="tissue",
        platform_column="platform",
        sample_attribute_columns={"breed": "breed"},
        file_columns=(
            SequencingFileColumns(
                uri="file",
                checksum="sha256",
                checksum_algorithm=SequencingChecksumAlgorithm.SHA256,
            ),
        ),
    )
    forged = manifest.model_dump(mode="python")
    forged["runs"][0]["files"][0]["uri"] = "https://attacker.invalid/read.fastq.gz"
    with pytest.raises(ValidationError, match="normalized manifest"):
        ExternalSequencingStudyManifest.model_validate(forged)


def test_file_column_contract_rejects_undeclared_or_orphaned_checksums() -> None:
    """Every imported digest has an algorithm and a corresponding file URI."""
    with pytest.raises(ValidationError, match="must be supplied together"):
        SequencingFileColumns(uri="file", checksum="digest")

    data = frame()
    data["file"] = [pd.NA, "https://archive.invalid/SRR2.fastq.gz", pd.NA]
    data["digest"] = ["a" * 32, "b" * 32, pd.NA]
    manifest_request = request()
    with pytest.raises(ValueError, match="no corresponding URI"):
        read_external_sequencing_manifest(
            data,
            request=manifest_request,
            provenance=provenance(manifest_request),
            subject_mapping=verified_mapping(),
            study_accession_column="study",
            sample_accession_column="sample",
            experiment_accession_column="experiment",
            run_accession_column="run",
            species_taxon_id_column="taxon",
            library_strategy_column="strategy",
            library_layout_column="layout",
            subject_id_column="subject",
            file_columns=(
                SequencingFileColumns(
                    uri="file",
                    checksum="digest",
                    checksum_algorithm=SequencingChecksumAlgorithm.MD5,
                ),
            ),
        )
