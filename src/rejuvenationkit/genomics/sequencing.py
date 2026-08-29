"""Typed, integrity-bound manifests for archived public sequencing studies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Self, TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rejuvenationkit.genomics.resources import (
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)
from rejuvenationkit.genomics.schemas import GenomicFeatureType

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_SUPPORTED_FEATURE_TYPES = frozenset(
    {
        GenomicFeatureType.GENE,
        GenomicFeatureType.TRANSCRIPT,
        GenomicFeatureType.VARIANT,
        GenomicFeatureType.REGION,
        GenomicFeatureType.PROTEIN,
    }
)
_SECRET_KEY_COMPACT_FORMS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "bearer",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "token",
    }
)


class SequencingLibraryLayout(StrEnum):
    """Read layout declared by a sequencing experiment."""

    SINGLE = "single"
    PAIRED = "paired"
    OTHER = "other"


class SequencingChecksumAlgorithm(StrEnum):
    """Supported checksum algorithms for one archived sequencing file."""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


class SubjectMappingStatus(StrEnum):
    """Completeness and verification state of sample-to-subject mappings."""

    NOT_PROVIDED = "not_provided"
    INCOMPLETE = "incomplete"
    COMPLETE_UNVERIFIED = "complete_unverified"
    VERIFIED = "verified"


class SubjectMappingBasis(StrEnum):
    """Evidence source used to map archived samples to independent subjects."""

    NONE = "none"
    PROVIDER_METADATA = "provider_metadata"
    STUDY_METADATA = "study_metadata"
    PUBLICATION = "publication"
    CURATED_CROSSWALK = "curated_crosswalk"


class SequencingSelectionParameter(BaseModel):
    """One secret-free, immutable selection constraint in a manifest request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_parameter(self) -> Self:
        """Require a clean public key and deterministic set-like values."""
        _require_clean_text(self.name, "selection parameter name")
        _reject_secret_like_name(self.name)
        if not self.values:
            raise ValueError("selection parameter values must be nonempty")
        _require_unique_clean_strings(self.values, "selection parameter values")
        object.__setattr__(self, "values", tuple(sorted(self.values)))
        return self


class SequencingManifestRequest(BaseModel):
    """Immutable scientific identity for one archived study-manifest request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_accession: str = Field(min_length=1)
    domain: FeatureDomain
    resource: ResourceSnapshot
    selection_parameters: tuple[SequencingSelectionParameter, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Validate the study boundary and canonicalize selection parameters."""
        _require_clean_text(self.study_accession, "study_accession")
        if self.domain.feature_type not in _SUPPORTED_FEATURE_TYPES:
            raise ValueError("sequencing manifest request has an incompatible feature type")
        parameters = tuple(
            SequencingSelectionParameter.model_validate(item.model_dump(mode="python"))
            for item in self.selection_parameters
        )
        names = tuple(item.name for item in parameters)
        if len(set(names)) != len(names):
            raise ValueError("selection parameter names must be unique")
        object.__setattr__(
            self,
            "selection_parameters",
            tuple(sorted(parameters, key=lambda item: item.name)),
        )
        return self

    @property
    def input_hash(self) -> str:
        """Hash study, scientific domain, raw resource, and selection constraints."""
        return canonical_sha256(
            {
                "study_accession": self.study_accession,
                "domain_hash": self.domain.domain_hash,
                "resource_snapshot_id": self.resource.snapshot_id,
                "selection_parameters": [
                    item.model_dump(mode="python") for item in self.selection_parameters
                ],
            }
        )


class SubjectMappingDeclaration(BaseModel):
    """Auditable claim about how samples were mapped to biological subjects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SubjectMappingStatus
    basis: SubjectMappingBasis
    source_id: str | None = None
    source_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verification_method: str | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        """Require source and verification evidence appropriate to the status."""
        for field_name, value in (
            ("source_id", self.source_id),
            ("verification_method", self.verification_method),
        ):
            if value is not None:
                _require_clean_text(value, field_name)
        if self.status is SubjectMappingStatus.NOT_PROVIDED:
            if self.basis is not SubjectMappingBasis.NONE:
                raise ValueError("an absent subject mapping must use basis='none'")
            if any(
                value is not None
                for value in (self.source_id, self.source_checksum, self.verification_method)
            ):
                raise ValueError("an absent subject mapping cannot claim a source or verification")
            return self
        if self.basis is SubjectMappingBasis.NONE or self.source_id is None:
            raise ValueError("a supplied subject mapping requires an explicit basis and source_id")
        if self.status is SubjectMappingStatus.VERIFIED:
            if self.verification_method is None:
                raise ValueError("a verified subject mapping requires verification_method")
        elif self.verification_method is not None:
            raise ValueError("only a verified subject mapping can claim verification_method")
        return self


class ExternalSequencingSample(BaseModel):
    """One biological sample, kept separate from libraries and runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_accession: str = Field(min_length=1)
    subject_id: str | None = None
    species_taxon_id: int = Field(strict=True, gt=0)
    tissue: str | None = None
    attributes: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sample(self) -> Self:
        """Reject ambiguous identifiers and metadata."""
        _require_clean_text(self.sample_accession, "sample_accession")
        if self.subject_id is not None:
            _require_clean_text(self.subject_id, "subject_id")
        if self.tissue is not None:
            _require_clean_text(self.tissue, "tissue")
        for key, value in self.attributes.items():
            _require_clean_text(key, "sample attribute name")
            _require_clean_text(value, f"sample attribute {key}")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        return self

    @field_serializer("attributes")
    def serialize_attributes(self, value: Mapping[str, str]) -> dict[str, str]:
        """Serialize the immutable metadata mapping as ordinary JSON data."""
        return dict(value)


class ExternalSequencingExperiment(BaseModel):
    """One library/experiment associated with a biological sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_accession: str = Field(min_length=1)
    sample_accession: str = Field(min_length=1)
    library_strategy: str = Field(min_length=1)
    library_source: str | None = None
    library_selection: str | None = None
    library_layout: SequencingLibraryLayout
    platform: str | None = None

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        """Reject ambiguous experiment identifiers and metadata."""
        for field_name, value in (
            ("experiment_accession", self.experiment_accession),
            ("sample_accession", self.sample_accession),
            ("library_strategy", self.library_strategy),
        ):
            _require_clean_text(value, field_name)
        for field_name, optional_value in (
            ("library_source", self.library_source),
            ("library_selection", self.library_selection),
            ("platform", self.platform),
        ):
            if optional_value is not None:
                _require_clean_text(optional_value, field_name)
        return self


class SequencingFileColumns(BaseModel):
    """Explicit columns and fixed semantics for one file in each run row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(min_length=1)
    checksum: str | None = Field(default=None, min_length=1)
    checksum_algorithm: SequencingChecksumAlgorithm | None = None
    role: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        """Require an algorithm whenever a checksum column is mapped."""
        _require_clean_text(self.uri, "sequencing file URI column")
        if self.checksum is not None:
            _require_clean_text(self.checksum, "sequencing file checksum column")
        if (self.checksum is None) != (self.checksum_algorithm is None):
            raise ValueError(
                "sequencing file checksum column and checksum_algorithm must be supplied together"
            )
        if self.role is not None:
            _require_clean_text(self.role, "sequencing file role")
        return self

    @property
    def referenced_columns(self) -> tuple[str, ...]:
        """Return the DataFrame columns used by this file declaration."""
        if self.checksum is None:
            return (self.uri,)
        return self.uri, self.checksum


class ExternalSequencingFile(BaseModel):
    """One immutable file belonging to a technical sequencing run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]+$")
    checksum_algorithm: SequencingChecksumAlgorithm | None = None
    role: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_file(self) -> Self:
        """Bind every checksum to an explicit algorithm and exact digest length."""
        _require_clean_text(self.uri, "sequencing file URI")
        if self.role is not None:
            _require_clean_text(self.role, "sequencing file role")
        if (self.checksum is None) != (self.checksum_algorithm is None):
            raise ValueError("sequencing file checksum and algorithm must be supplied together")
        if self.checksum is not None and self.checksum_algorithm is not None:
            expected_length = {
                SequencingChecksumAlgorithm.MD5: 32,
                SequencingChecksumAlgorithm.SHA1: 40,
                SequencingChecksumAlgorithm.SHA256: 64,
                SequencingChecksumAlgorithm.SHA512: 128,
            }[self.checksum_algorithm]
            if len(self.checksum) != expected_length:
                raise ValueError(
                    f"{self.checksum_algorithm.value} sequencing checksum must contain "
                    f"exactly {expected_length} hexadecimal characters"
                )
        return self


class ExternalSequencingRun(BaseModel):
    """One technical sequencing run associated with an experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_accession: str = Field(min_length=1)
    experiment_accession: str = Field(min_length=1)
    files: tuple[ExternalSequencingFile, ...] = ()
    # Deprecated one-file compatibility fields. They are accepted only when
    # they form an unambiguous ExternalSequencingFile declaration.
    file_uri: str | None = None
    file_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]+$")
    file_checksum_algorithm: SequencingChecksumAlgorithm | None = None

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Reject ambiguous run identifiers and metadata."""
        _require_clean_text(self.run_accession, "run_accession")
        _require_clean_text(self.experiment_accession, "experiment_accession")
        normalized_files = tuple(
            ExternalSequencingFile.model_validate(item.model_dump(mode="python"))
            for item in self.files
        )
        if self.file_uri is not None:
            _require_clean_text(self.file_uri, "file_uri")
            legacy_file = ExternalSequencingFile(
                uri=self.file_uri,
                checksum=self.file_checksum,
                checksum_algorithm=self.file_checksum_algorithm,
            )
            if normalized_files and normalized_files != (legacy_file,):
                raise ValueError(
                    "legacy one-file fields must exactly match the explicit files declaration"
                )
            normalized_files = (legacy_file,)
        elif self.file_checksum is not None or self.file_checksum_algorithm is not None:
            raise ValueError("legacy sequencing checksum fields require file_uri")
        file_keys = tuple(_sequencing_file_sort_key(item) for item in normalized_files)
        if len({item.uri for item in normalized_files}) != len(normalized_files):
            raise ValueError("sequencing run file URIs must be unique")
        if file_keys != tuple(sorted(file_keys)):
            normalized_files = tuple(sorted(normalized_files, key=_sequencing_file_sort_key))
        object.__setattr__(self, "files", normalized_files)
        return self


class ExternalSequencingStudyManifest(BaseModel):
    """Integrity-bound study/sample/experiment/run hierarchy from an archive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: SequencingManifestRequest
    samples: tuple[ExternalSequencingSample, ...]
    experiments: tuple[ExternalSequencingExperiment, ...]
    runs: tuple[ExternalSequencingRun, ...]
    subject_mapping: SubjectMappingDeclaration
    provenance: QueryProvenance
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_hierarchy(self) -> Self:
        """Reconstruct hierarchy, query identity, and normalized response integrity."""
        request = SequencingManifestRequest.model_validate(self.request.model_dump(mode="python"))
        provenance = QueryProvenance.model_validate(self.provenance.model_dump(mode="python"))
        mapping = SubjectMappingDeclaration.model_validate(
            self.subject_mapping.model_dump(mode="python")
        )
        samples = tuple(
            ExternalSequencingSample.model_validate(item.model_dump(mode="python"))
            for item in self.samples
        )
        experiments = tuple(
            ExternalSequencingExperiment.model_validate(item.model_dump(mode="python"))
            for item in self.experiments
        )
        runs = tuple(
            ExternalSequencingRun.model_validate(item.model_dump(mode="python"))
            for item in self.runs
        )
        if not samples or not experiments or not runs:
            raise ValueError("sequencing manifest must contain samples, experiments, and runs")
        sample_ids = tuple(item.sample_accession for item in samples)
        experiment_ids = tuple(item.experiment_accession for item in experiments)
        run_ids = tuple(item.run_accession for item in runs)
        _require_unique_clean_strings(sample_ids, "sequencing sample accessions")
        _require_unique_clean_strings(experiment_ids, "sequencing experiment accessions")
        _require_unique_clean_strings(run_ids, "sequencing run accessions")
        if sample_ids != tuple(sorted(sample_ids)):
            raise ValueError("sequencing samples must use deterministic accession order")
        if experiment_ids != tuple(sorted(experiment_ids)):
            raise ValueError("sequencing experiments must use deterministic accession order")
        if run_ids != tuple(sorted(run_ids)):
            raise ValueError("sequencing runs must use deterministic accession order")
        experiment_sample_ids = {item.sample_accession for item in experiments}
        if not experiment_sample_ids.issubset(sample_ids):
            raise ValueError("every sequencing experiment must reference a manifest sample")
        orphan_samples = set(sample_ids).difference(experiment_sample_ids)
        if orphan_samples:
            raise ValueError(
                f"sequencing manifest contains orphan samples: {sorted(orphan_samples)}"
            )
        run_experiment_ids = {item.experiment_accession for item in runs}
        if not run_experiment_ids.issubset(experiment_ids):
            raise ValueError("every sequencing run must reference a manifest experiment")
        orphan_experiments = set(experiment_ids).difference(run_experiment_ids)
        if orphan_experiments:
            raise ValueError(
                f"sequencing manifest contains orphan experiments: {sorted(orphan_experiments)}"
            )
        species = {item.species_taxon_id for item in samples}
        if species != {request.domain.species_taxon_id}:
            raise ValueError("sequencing samples and request must share one species")
        if provenance.domain != request.domain:
            raise ValueError("sequencing request and provenance domains must match")
        if provenance.input_hash != request.input_hash:
            raise ValueError("sequencing request hash must match provenance input_hash")
        if request.resource not in provenance.resources:
            raise ValueError("sequencing resource must be captured in query provenance")
        if not provenance.complete:
            raise ValueError("sequencing manifest provenance must be complete")
        _require_unique_clean_strings(self.warnings, "warnings")
        expected_warnings = _validate_subject_mapping(samples, mapping)
        if self.warnings != expected_warnings:
            raise ValueError("sequencing manifest warnings do not match subject mapping status")
        expected_checksum = _normalized_manifest_checksum(
            request=request,
            samples=samples,
            experiments=experiments,
            runs=runs,
            subject_mapping=mapping,
            warnings=self.warnings,
        )
        if provenance.response_checksum != expected_checksum:
            raise ValueError(
                "sequencing provenance response checksum does not match the normalized manifest"
            )
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "subject_mapping", mapping)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "runs", runs)
        return self

    @property
    def study_accession(self) -> str:
        """Return the requested study accession for read-only convenience."""
        return self.request.study_accession

    @property
    def resource(self) -> ResourceSnapshot:
        """Return the exact raw archive snapshot for read-only convenience."""
        return self.request.resource

    @property
    def normalized_manifest_checksum(self) -> str:
        """Return the canonical checksum bound to provenance.response_checksum."""
        return _normalized_manifest_checksum(
            request=self.request,
            samples=self.samples,
            experiments=self.experiments,
            runs=self.runs,
            subject_mapping=self.subject_mapping,
            warnings=self.warnings,
        )

    @property
    def subject_mapping_complete(self) -> bool:
        """Return whether every archived sample has a nonempty subject identifier."""
        return all(item.subject_id is not None for item in self.samples)

    @property
    def subject_mapping_verified(self) -> bool:
        """Return whether the complete mapping has explicit verification evidence."""
        return (
            self.subject_mapping_complete
            and self.subject_mapping.status is SubjectMappingStatus.VERIFIED
        )

    @property
    def independent_subject_count(self) -> int | None:
        """Count independent subjects only when the mapping is explicitly verified."""
        if not self.subject_mapping_verified:
            return None
        return len({item.subject_id for item in self.samples})

    def require_subject_mapping(self) -> None:
        """Reject independent-subject analyses without a verified complete mapping."""
        if not self.subject_mapping_complete:
            raise ValueError(
                "sequencing subject mapping is incomplete; runs and samples cannot be "
                "assumed to be independent subjects"
            )
        if not self.subject_mapping_verified:
            raise ValueError(
                "sequencing subject mapping is complete but unverified; independent-subject "
                "analyses require explicit verification"
            )


def read_external_sequencing_manifest(
    frame: pd.DataFrame,
    *,
    request: SequencingManifestRequest,
    provenance: QueryProvenance,
    subject_mapping: SubjectMappingDeclaration,
    study_accession_column: str,
    sample_accession_column: str,
    experiment_accession_column: str,
    run_accession_column: str,
    species_taxon_id_column: str,
    library_strategy_column: str,
    library_layout_column: str,
    subject_id_column: str | None = None,
    tissue_column: str | None = None,
    library_source_column: str | None = None,
    library_selection_column: str | None = None,
    platform_column: str | None = None,
    file_uri_column: str | None = None,
    file_checksum_column: str | None = None,
    file_checksum_algorithm: SequencingChecksumAlgorithm | None = None,
    file_columns: tuple[SequencingFileColumns, ...] = (),
    sample_attribute_columns: Mapping[str, str] | None = None,
) -> ExternalSequencingStudyManifest:
    """Read an archived SRA-like export and bind its normalized typed content."""
    request = SequencingManifestRequest.model_validate(request.model_dump(mode="python"))
    provenance = QueryProvenance.model_validate(provenance.model_dump(mode="python"))
    subject_mapping = SubjectMappingDeclaration.model_validate(
        subject_mapping.model_dump(mode="python")
    )
    if request.domain != provenance.domain:
        raise ValueError("sequencing request and provenance domains must match")
    if request.input_hash != provenance.input_hash:
        raise ValueError("sequencing request hash must match provenance input_hash")
    if request.resource not in provenance.resources:
        raise ValueError("sequencing resource must be captured in query provenance")
    if not provenance.complete:
        raise ValueError("sequencing manifest provenance must be complete")
    if file_columns and any(
        item is not None
        for item in (file_uri_column, file_checksum_column, file_checksum_algorithm)
    ):
        raise ValueError(
            "explicit file_columns cannot be combined with legacy one-file column arguments"
        )
    if (file_checksum_column is None) != (file_checksum_algorithm is None):
        raise ValueError(
            "legacy file_checksum_column and file_checksum_algorithm must be supplied together"
        )
    if file_uri_column is None and file_checksum_column is not None:
        raise ValueError("legacy file_checksum_column requires file_uri_column")
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
        raise ValueError(f"sequencing manifest contains duplicate columns: {duplicates}")
    required_columns = (
        study_accession_column,
        sample_accession_column,
        experiment_accession_column,
        run_accession_column,
        species_taxon_id_column,
        library_strategy_column,
        library_layout_column,
    )
    optional_columns = tuple(
        value
        for value in (
            subject_id_column,
            tissue_column,
            library_source_column,
            library_selection_column,
            platform_column,
            file_uri_column,
            file_checksum_column,
        )
        if value is not None
    )
    declared_file_columns = tuple(
        column
        for file_declaration in file_columns
        for column in file_declaration.referenced_columns
    )
    attributes = dict(sample_attribute_columns or {})
    for name, column in attributes.items():
        _require_clean_text(name, "sample attribute name")
        _require_clean_text(column, f"column for sample attribute {name}")
    mapped_columns = (
        *required_columns,
        *optional_columns,
        *declared_file_columns,
        *attributes.values(),
    )
    if len(set(mapped_columns)) != len(mapped_columns):
        raise ValueError("sequencing manifest column mappings must be unique")
    missing = sorted(set(mapped_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"sequencing manifest columns are absent: {missing}")
    if frame.empty:
        raise ValueError("sequencing manifest table is empty")
    if (
        subject_id_column is None
        and subject_mapping.status is not SubjectMappingStatus.NOT_PROVIDED
    ):
        raise ValueError("a declared subject mapping requires subject_id_column")

    samples: dict[str, ExternalSequencingSample] = {}
    experiments: dict[str, ExternalSequencingExperiment] = {}
    runs: dict[str, ExternalSequencingRun] = {}
    for row_index in range(len(frame)):
        row = frame.iloc[row_index]
        row_study = _required_string(row, study_accession_column, row_index)
        if row_study != request.study_accession:
            raise ValueError(
                f"sequencing row study {row_study!r} does not match requested "
                f"study {request.study_accession!r}"
            )
        sample = ExternalSequencingSample(
            sample_accession=_required_string(row, sample_accession_column, row_index),
            subject_id=_optional_string(row, subject_id_column, row_index),
            species_taxon_id=_strict_species_taxon_id(
                row[species_taxon_id_column],
                column=species_taxon_id_column,
                row_index=row_index,
            ),
            tissue=_optional_string(row, tissue_column, row_index),
            attributes={
                name: value
                for name, column in attributes.items()
                if (value := _optional_string(row, column, row_index)) is not None
            },
        )
        if sample.species_taxon_id != request.domain.species_taxon_id:
            raise ValueError(
                f"sequencing row {row_index} taxon {sample.species_taxon_id} does not match "
                f"requested taxon {request.domain.species_taxon_id}"
            )
        _insert_consistent(samples, sample.sample_accession, sample, "sample")
        experiment = ExternalSequencingExperiment(
            experiment_accession=_required_string(row, experiment_accession_column, row_index),
            sample_accession=sample.sample_accession,
            library_strategy=_required_string(row, library_strategy_column, row_index),
            library_source=_optional_string(row, library_source_column, row_index),
            library_selection=_optional_string(row, library_selection_column, row_index),
            library_layout=_parse_layout(
                row[library_layout_column],
                column=library_layout_column,
                row_index=row_index,
            ),
            platform=_optional_string(row, platform_column, row_index),
        )
        _insert_consistent(
            experiments,
            experiment.experiment_accession,
            experiment,
            "experiment",
        )
        explicit_files = tuple(
            file
            for declaration in file_columns
            if (
                file := _sequencing_file_from_row(
                    row,
                    declaration=declaration,
                    row_index=row_index,
                )
            )
            is not None
        )
        legacy_file_uri = _optional_string(row, file_uri_column, row_index)
        legacy_file_checksum = _optional_string(row, file_checksum_column, row_index)
        run = ExternalSequencingRun(
            run_accession=_required_string(row, run_accession_column, row_index),
            experiment_accession=experiment.experiment_accession,
            files=explicit_files,
            file_uri=legacy_file_uri,
            file_checksum=legacy_file_checksum,
            file_checksum_algorithm=(
                file_checksum_algorithm if legacy_file_checksum is not None else None
            ),
        )
        if run.run_accession in runs:
            raise ValueError(f"duplicate sequencing run accession {run.run_accession!r}")
        runs[run.run_accession] = run

    normalized_samples = tuple(samples[key] for key in sorted(samples))
    normalized_experiments = tuple(experiments[key] for key in sorted(experiments))
    normalized_runs = tuple(runs[key] for key in sorted(runs))
    manifest_warnings = _validate_subject_mapping(normalized_samples, subject_mapping)
    normalized_checksum = _normalized_manifest_checksum(
        request=request,
        samples=normalized_samples,
        experiments=normalized_experiments,
        runs=normalized_runs,
        subject_mapping=subject_mapping,
        warnings=manifest_warnings,
    )
    if (
        provenance.response_checksum is not None
        and provenance.response_checksum != normalized_checksum
    ):
        raise ValueError(
            "sequencing provenance response checksum does not match the normalized manifest"
        )
    provenance_payload = provenance.model_dump(mode="python")
    provenance_payload["response_checksum"] = normalized_checksum
    provenance_payload["warnings"] = manifest_warnings
    normalized_provenance = QueryProvenance.model_validate(provenance_payload)
    return ExternalSequencingStudyManifest(
        request=request,
        samples=normalized_samples,
        experiments=normalized_experiments,
        runs=normalized_runs,
        subject_mapping=subject_mapping,
        provenance=normalized_provenance,
        warnings=manifest_warnings,
    )


def _insert_consistent(
    collection: dict[str, _ModelT],
    identifier: str,
    value: _ModelT,
    entity_name: str,
) -> None:
    previous = collection.get(identifier)
    if previous is not None and previous != value:
        raise ValueError(f"conflicting {entity_name} metadata for {identifier!r}")
    collection[identifier] = value


def _validate_subject_mapping(
    samples: Sequence[ExternalSequencingSample],
    declaration: SubjectMappingDeclaration,
) -> tuple[str, ...]:
    present = sum(item.subject_id is not None for item in samples)
    if present == 0:
        if declaration.status is not SubjectMappingStatus.NOT_PROVIDED:
            raise ValueError("subject mapping status must be 'not_provided' when no IDs exist")
        return ("independent_subject_mapping_incomplete",)
    if present < len(samples):
        if declaration.status is not SubjectMappingStatus.INCOMPLETE:
            raise ValueError("subject mapping status must be 'incomplete' when some IDs are absent")
        return ("independent_subject_mapping_incomplete",)
    if declaration.status not in {
        SubjectMappingStatus.COMPLETE_UNVERIFIED,
        SubjectMappingStatus.VERIFIED,
    }:
        raise ValueError(
            "a complete sample-to-subject mapping must be declared complete_unverified or verified"
        )
    if declaration.status is SubjectMappingStatus.COMPLETE_UNVERIFIED:
        return ("independent_subject_mapping_unverified",)
    return ()


def _normalized_manifest_checksum(
    *,
    request: SequencingManifestRequest,
    samples: Sequence[ExternalSequencingSample],
    experiments: Sequence[ExternalSequencingExperiment],
    runs: Sequence[ExternalSequencingRun],
    subject_mapping: SubjectMappingDeclaration,
    warnings: tuple[str, ...],
) -> str:
    return canonical_sha256(
        {
            "schema": "rejuvenationkit.external-sequencing-manifest/v3",
            "request": request.model_dump(mode="python"),
            "request_input_hash": request.input_hash,
            "samples": [item.model_dump(mode="python") for item in samples],
            "experiments": [item.model_dump(mode="python") for item in experiments],
            "runs": [_normalized_run_payload(item) for item in runs],
            "subject_mapping": subject_mapping.model_dump(mode="python"),
            "warnings": warnings,
            "fusion_eligibility": "not_fusible",
        }
    )


def _parse_layout(
    value: object,
    *,
    column: str,
    row_index: int,
) -> SequencingLibraryLayout:
    if _is_missing_scalar(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"sequencing manifest value in {column!r} at row {row_index} cannot be missing"
        )
    normalized = re.sub(r"[\s_-]+", "_", str(value).strip().casefold())
    if normalized in {"single", "single_end", "single_ended"}:
        return SequencingLibraryLayout.SINGLE
    if normalized in {"paired", "paired_end", "paired_ended"}:
        return SequencingLibraryLayout.PAIRED
    if normalized:
        return SequencingLibraryLayout.OTHER
    raise ValueError(
        f"sequencing manifest value in {column!r} at row {row_index} cannot be missing"
    )


def _sequencing_file_from_row(
    row: pd.Series,
    *,
    declaration: SequencingFileColumns,
    row_index: int,
) -> ExternalSequencingFile | None:
    """Read one optional file slot while rejecting orphaned checksums."""
    uri = _optional_string(row, declaration.uri, row_index)
    checksum = _optional_string(row, declaration.checksum, row_index)
    if uri is None:
        if checksum is not None:
            raise ValueError(
                f"sequencing file checksum at row {row_index} has no corresponding URI"
            )
        return None
    return ExternalSequencingFile(
        uri=uri,
        checksum=checksum,
        checksum_algorithm=declaration.checksum_algorithm if checksum is not None else None,
        role=declaration.role,
    )


def _sequencing_file_sort_key(file: ExternalSequencingFile) -> tuple[str, str, str, str]:
    """Return a deterministic order for files within one run."""
    return (
        file.role or "",
        file.uri,
        file.checksum_algorithm.value if file.checksum_algorithm is not None else "",
        file.checksum or "",
    )


def _normalized_run_payload(run: ExternalSequencingRun) -> dict[str, object]:
    """Exclude deprecated aliases from the canonical scientific representation."""
    return {
        "run_accession": run.run_accession,
        "experiment_accession": run.experiment_accession,
        "files": [file.model_dump(mode="python") for file in run.files],
    }


def _strict_species_taxon_id(value: object, *, column: str, row_index: int) -> int:
    message = (
        f"sequencing species taxon ID in {column!r} at row {row_index} must be a positive integer"
    )
    if _is_missing_scalar(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    parsed: int
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not isfinite(numeric) or not numeric.is_integer():
            raise ValueError(message)
        parsed = int(numeric)
    elif isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(message)
    if parsed <= 0:
        raise ValueError(message)
    return parsed


def _required_string(row: pd.Series, column: str, row_index: int) -> str:
    value = _optional_string(row, column, row_index)
    if value is None:
        raise ValueError(
            f"sequencing manifest value in {column!r} at row {row_index} cannot be missing"
        )
    return value


def _optional_string(row: pd.Series, column: str | None, row_index: int) -> str | None:
    if column is None:
        return None
    value = row[column]
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"sequencing manifest value in {column!r} at row {row_index} must be text")
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip()
    return normalized or None


def _is_missing_scalar(value: object) -> bool:
    missing = pd.isna(cast(Any, value))
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    raise ValueError("sequencing manifest cells must contain scalar values")


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_unique_clean_strings(values: Sequence[str], field_name: str) -> None:
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")


def _reject_secret_like_name(name: str) -> None:
    lowered = name.casefold()
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    segments = frozenset(filter(None, re.split(r"[^a-z0-9]+", lowered)))
    secret_compounds = (
        "accesskey",
        "accesstoken",
        "apikey",
        "apitoken",
        "authtoken",
        "clientsecret",
        "privatekey",
    )
    secret_segments = {
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
    }
    if (
        compact in _SECRET_KEY_COMPACT_FORMS
        or any(marker in compact for marker in secret_compounds)
        or bool(segments.intersection(secret_segments))
    ):
        raise ValueError(f"selection parameter contains a secret-like name: {name}")
