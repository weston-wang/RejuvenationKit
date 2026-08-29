"""Provider-neutral functional annotations bound to exact query provenance."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, Self, TypeAlias, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    canonical_sha256,
)

AssociationKey: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    str | None,
    tuple[str, ...],
]


class FunctionalAssociation(BaseModel):
    """One provider assertion connecting a query feature to a functional term."""

    model_config = ConfigDict(frozen=True)

    feature_id: str = Field(min_length=1)
    term_id: str = Field(min_length=1)
    term_name: str = Field(min_length=1)
    term_namespace: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    evidence_code: str | None = None
    qualifiers: tuple[str, ...] = ()
    source_record_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_association(self) -> Self:
        """Reject ambiguous identifiers and canonicalize set-like metadata."""
        for field_name, value in (
            ("feature_id", self.feature_id),
            ("term_id", self.term_id),
            ("term_name", self.term_name),
            ("term_namespace", self.term_namespace),
            ("relation", self.relation),
        ):
            _require_clean_text(value, field_name)
        if self.evidence_code is not None:
            _require_clean_text(self.evidence_code, "evidence_code")
        _require_unique_clean_strings(self.qualifiers, "qualifiers")
        _require_unique_clean_strings(self.source_record_ids, "source_record_ids")
        object.__setattr__(self, "qualifiers", tuple(sorted(self.qualifiers)))
        object.__setattr__(self, "source_record_ids", tuple(sorted(self.source_record_ids)))
        return self

    @property
    def association_hash(self) -> str:
        """Return a deterministic hash of the complete provider assertion."""
        return canonical_sha256(self.model_dump(mode="python"))


class FunctionalAnnotationResult(BaseModel):
    """Functional associations and explicit query coverage from one provider result."""

    model_config = ConfigDict(frozen=True)

    query_collection_id: str = Field(min_length=1)
    query_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_source_snapshot_id: str = Field(min_length=1)
    domain: FeatureDomain
    query_feature_ids: tuple[str, ...]
    associations: tuple[FunctionalAssociation, ...]
    matched_feature_ids: tuple[str, ...]
    unmatched_feature_ids: tuple[str, ...]
    provenance: QueryProvenance
    complete: bool
    truncated: bool = False
    warnings: tuple[str, ...] = ()
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate exact query coverage, provenance binding, and completeness."""
        _require_clean_text(self.query_collection_id, "query_collection_id")
        _require_clean_text(self.query_source_snapshot_id, "query_source_snapshot_id")
        if not self.query_feature_ids:
            raise ValueError("query_feature_ids must be nonempty")
        _require_unique_clean_strings(self.query_feature_ids, "query_feature_ids")
        _require_unique_clean_strings(self.matched_feature_ids, "matched_feature_ids")
        _require_unique_clean_strings(self.unmatched_feature_ids, "unmatched_feature_ids")
        _require_unique_clean_strings(self.warnings, "warnings")
        if self.provenance.domain != self.domain:
            raise ValueError("functional annotation domain must match query provenance")
        if self.provenance.input_hash != self.query_content_hash:
            raise ValueError("functional annotation input hash must match query provenance")
        if not self.provenance.resources:
            raise ValueError("functional annotation provenance requires a resource snapshot")
        if self.complete and not self.provenance.complete:
            raise ValueError("a result cannot be complete when query provenance is incomplete")
        if self.complete and self.truncated:
            raise ValueError("a truncated functional annotation result cannot be complete")
        query_set = set(self.query_feature_ids)
        matched_set = set(self.matched_feature_ids)
        unmatched_set = set(self.unmatched_feature_ids)
        if matched_set.intersection(unmatched_set):
            raise ValueError("matched and unmatched query features must be disjoint")
        if matched_set.union(unmatched_set) != query_set:
            raise ValueError("matched and unmatched features must partition the exact query")
        associated_features = {item.feature_id for item in self.associations}
        if associated_features != matched_set:
            raise ValueError("matched features must exactly equal features with associations")
        if self.associations != tuple(sorted(self.associations, key=_association_sort_key)):
            raise ValueError("functional associations must use deterministic canonical order")
        if len({item.association_hash for item in self.associations}) != len(self.associations):
            raise ValueError("functional associations must be unique")
        if self.truncated and "functional_annotation_result_truncated" not in self.warnings:
            raise ValueError("truncated results must include a truncation warning")
        if not self.complete and "functional_annotation_result_incomplete" not in self.warnings:
            raise ValueError("incomplete results must include an incompleteness warning")
        normalized_hash = _annotation_response_hash(
            query_content_hash=self.query_content_hash,
            associations=self.associations,
            matched_feature_ids=self.matched_feature_ids,
            unmatched_feature_ids=self.unmatched_feature_ids,
            complete=self.complete,
            truncated=self.truncated,
            warnings=self.warnings,
        )
        if self.provenance.response_checksum != normalized_hash:
            raise ValueError(
                "functional annotation response checksum does not match normalized imported data"
            )
        return self

    @property
    def result_hash(self) -> str:
        """Hash result content, query identity, and provider response provenance."""
        return canonical_sha256(
            {
                "query_collection_id": self.query_collection_id,
                "query_content_hash": self.query_content_hash,
                "query_source_snapshot_id": self.query_source_snapshot_id,
                "domain": self.domain.model_dump(mode="python"),
                "query_feature_ids": sorted(self.query_feature_ids),
                "associations": [item.model_dump(mode="python") for item in self.associations],
                "matched_feature_ids": sorted(self.matched_feature_ids),
                "unmatched_feature_ids": sorted(self.unmatched_feature_ids),
                "query_hash": self.provenance.query_hash,
                "response_checksum": self.provenance.response_checksum,
                "complete": self.complete,
                "truncated": self.truncated,
                "warnings": sorted(self.warnings),
                "fusion_eligibility": self.fusion_eligibility,
            }
        )


def read_functional_annotations(
    frame: pd.DataFrame,
    *,
    query: FeatureCollection,
    provenance: QueryProvenance,
    feature_id_column: str,
    term_id_column: str,
    term_name_column: str,
    term_namespace_column: str,
    relation_column: str,
    evidence_code_column: str | None = None,
    qualifiers_column: str | None = None,
    source_record_ids_column: str | None = None,
    multi_value_separator: str = "|",
    complete: bool | None = None,
    truncated: bool = False,
    warnings: tuple[str, ...] = (),
) -> FunctionalAnnotationResult:
    """Import an archived provider table through explicit, auditable mappings.

    Exact duplicate semantic associations are collapsed and their source record
    identifiers are merged. Distinct evidence codes or qualifiers remain
    distinct assertions. Conflicting names for one namespaced term are rejected.
    """
    if query.domain != provenance.domain:
        raise ValueError("query feature domain must exactly match query provenance")
    if query.content_hash != provenance.input_hash:
        raise ValueError("query feature collection hash must match provenance input_hash")
    if not provenance.resources:
        raise ValueError("functional annotation provenance requires a resource snapshot")
    if not multi_value_separator:
        raise ValueError("multi_value_separator must be nonempty")
    _require_unique_clean_strings(warnings, "warnings")
    required_columns = {
        feature_id_column,
        term_id_column,
        term_name_column,
        term_namespace_column,
        relation_column,
    }
    optional_columns = {
        value
        for value in (evidence_code_column, qualifiers_column, source_record_ids_column)
        if value is not None
    }
    missing_columns = sorted((required_columns | optional_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"functional annotation columns are absent: {missing_columns}")

    query_ids = set(query.feature_ids)
    term_names: dict[tuple[str, str], str] = {}
    associations_by_key: dict[AssociationKey, FunctionalAssociation] = {}
    source_ids_by_key: defaultdict[AssociationKey, set[str]] = defaultdict(set)
    source_id_owner: dict[str, AssociationKey] = {}
    unexpected_features: set[str] = set()

    for row_index in range(len(frame)):
        row = frame.iloc[row_index]
        feature_id = _required_cell_text(row, feature_id_column, row_index)
        term_id = _required_cell_text(row, term_id_column, row_index)
        term_name = _required_cell_text(row, term_name_column, row_index)
        term_namespace = _required_cell_text(row, term_namespace_column, row_index)
        relation = _required_cell_text(row, relation_column, row_index)
        evidence_code = _optional_cell_text(row, evidence_code_column, row_index)
        qualifiers = _multi_value_cell(
            row,
            qualifiers_column,
            row_index=row_index,
            separator=multi_value_separator,
        )
        source_record_ids = _multi_value_cell(
            row,
            source_record_ids_column,
            row_index=row_index,
            separator=multi_value_separator,
        )
        if feature_id not in query_ids:
            unexpected_features.add(feature_id)
            continue
        term_key = (term_namespace, term_id)
        previous_name = term_names.setdefault(term_key, term_name)
        if previous_name != term_name:
            raise ValueError(
                f"conflicting term names for {term_namespace}:{term_id}: "
                f"{previous_name!r} versus {term_name!r}"
            )
        semantic_key = (
            feature_id,
            term_id,
            term_name,
            term_namespace,
            relation,
            evidence_code,
            qualifiers,
        )
        association = FunctionalAssociation(
            feature_id=feature_id,
            term_id=term_id,
            term_name=term_name,
            term_namespace=term_namespace,
            relation=relation,
            evidence_code=evidence_code,
            qualifiers=qualifiers,
            source_record_ids=(),
        )
        associations_by_key.setdefault(semantic_key, association)
        for source_record_id in source_record_ids:
            owner = source_id_owner.setdefault(source_record_id, semantic_key)
            if owner != semantic_key:
                raise ValueError(
                    f"source record {source_record_id!r} maps to conflicting associations"
                )
            source_ids_by_key[semantic_key].add(source_record_id)

    if unexpected_features:
        raise ValueError(
            "functional annotation result contains features outside the exact query: "
            f"{sorted(unexpected_features)}"
        )
    associations = tuple(
        sorted(
            (
                association.model_copy(
                    update={"source_record_ids": tuple(sorted(source_ids_by_key[semantic_key]))}
                )
                for semantic_key, association in associations_by_key.items()
            ),
            key=_association_sort_key,
        )
    )
    associated_features = {item.feature_id for item in associations}
    matched = tuple(
        feature_id for feature_id in query.feature_ids if feature_id in associated_features
    )
    unmatched = tuple(
        feature_id for feature_id in query.feature_ids if feature_id not in associated_features
    )

    result_complete = provenance.complete and not truncated if complete is None else complete
    if result_complete and not provenance.complete:
        raise ValueError("result cannot be complete when query provenance is incomplete")
    if result_complete and truncated:
        raise ValueError("truncated functional annotation results cannot be complete")
    result_warnings = list(dict.fromkeys((*provenance.warnings, *warnings)))
    if truncated:
        result_warnings.append("functional_annotation_result_truncated")
    if not result_complete:
        result_warnings.append("functional_annotation_result_incomplete")
    result_warnings = list(dict.fromkeys(result_warnings))

    normalized_response_hash = _annotation_response_hash(
        query_content_hash=query.content_hash,
        associations=associations,
        matched_feature_ids=matched,
        unmatched_feature_ids=unmatched,
        complete=result_complete,
        truncated=truncated,
        warnings=tuple(result_warnings),
    )
    if (
        provenance.response_checksum is not None
        and provenance.response_checksum != normalized_response_hash
    ):
        raise ValueError(
            "supplied response checksum does not match normalized functional annotations"
        )
    normalized_provenance = provenance.model_copy(
        update={
            "response_checksum": normalized_response_hash,
            "complete": result_complete,
            "warnings": tuple(result_warnings),
        }
    )

    return FunctionalAnnotationResult(
        query_collection_id=query.collection_id,
        query_content_hash=query.content_hash,
        query_source_snapshot_id=query.source_snapshot_id,
        domain=query.domain,
        query_feature_ids=query.feature_ids,
        associations=associations,
        matched_feature_ids=matched,
        unmatched_feature_ids=unmatched,
        provenance=normalized_provenance,
        complete=result_complete,
        truncated=truncated,
        warnings=tuple(result_warnings),
    )


def _annotation_response_hash(
    *,
    query_content_hash: str,
    associations: tuple[FunctionalAssociation, ...],
    matched_feature_ids: tuple[str, ...],
    unmatched_feature_ids: tuple[str, ...],
    complete: bool,
    truncated: bool,
    warnings: tuple[str, ...],
) -> str:
    """Hash the canonical imported response separately from raw provider bytes."""
    return canonical_sha256(
        {
            "query_content_hash": query_content_hash,
            "associations": [item.model_dump(mode="python") for item in associations],
            "matched_feature_ids": sorted(matched_feature_ids),
            "unmatched_feature_ids": sorted(unmatched_feature_ids),
            "complete": complete,
            "truncated": truncated,
            "warnings": sorted(warnings),
        }
    )


def _association_sort_key(
    association: FunctionalAssociation,
) -> tuple[str, str, str, str, str, str, tuple[str, ...], tuple[str, ...]]:
    return (
        association.feature_id,
        association.term_namespace,
        association.term_id,
        association.term_name,
        association.relation,
        association.evidence_code or "",
        association.qualifiers,
        association.source_record_ids,
    )


def _required_cell_text(row: pd.Series, column: str, row_index: int) -> str:
    value = row[column]
    if _is_missing(value):
        raise ValueError(
            f"row {row_index} required functional annotation field {column!r} is missing"
        )
    text = str(value)
    _require_clean_text(text, f"row {row_index} field {column}")
    return text


def _optional_cell_text(row: pd.Series, column: str | None, row_index: int) -> str | None:
    if column is None:
        return None
    value = row[column]
    if _is_missing(value):
        return None
    text = str(value)
    _require_clean_text(text, f"row {row_index} field {column}")
    return text


def _multi_value_cell(
    row: pd.Series,
    column: str | None,
    *,
    row_index: int,
    separator: str,
) -> tuple[str, ...]:
    if column is None:
        return ()
    value = row[column]
    if _is_missing(value):
        return ()
    if isinstance(value, str):
        items = tuple(value.split(separator))
    elif isinstance(value, (tuple, list, set, frozenset)):
        items = tuple(str(item) for item in value)
    else:
        items = (str(value),)
    for item in items:
        _require_clean_text(item, f"row {row_index} field {column}")
    if len(set(items)) != len(items):
        raise ValueError(f"row {row_index} field {column} contains duplicate values")
    return tuple(sorted(items))


def _is_missing(value: object) -> bool:
    missing = pd.isna(cast(Any, value))
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _require_clean_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")


def _require_unique_clean_strings(values: tuple[str, ...], field_name: str) -> None:
    for value in values:
        _require_clean_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
