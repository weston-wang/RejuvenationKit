"""Typed ingestion of assay-pipeline feature effects."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicFeatureType,
    validate_feature_domain_compatibility,
)
from rejuvenationkit.schemas import Modality


class FeatureEffect(BaseModel):
    """One gene-, CpG-, region-, or variant-level contrast from an upstream model."""

    model_config = ConfigDict(frozen=True)

    feature_id: str = Field(min_length=1)
    feature_type: GenomicFeatureType
    namespace: FeatureNamespace
    modality: Modality
    contrast: str = Field(min_length=1)
    effect: float
    standard_error: float = Field(gt=0)
    effect_unit: str = Field(min_length=1)
    statistic: float | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    adjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    species_taxon_id: int = Field(gt=0)
    tissue: str = Field(min_length=1)
    genome_assembly: str | None = None
    provenance_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        """Reject invalid domains or numerical outputs from upstream models."""
        for field_name, value in (
            ("feature_id", self.feature_id),
            ("contrast", self.contrast),
            ("effect_unit", self.effect_unit),
            ("tissue", self.tissue),
            ("provenance_id", self.provenance_id),
        ):
            if value != value.strip() or not value:
                raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")
        validate_feature_domain_compatibility(
            feature_type=self.feature_type,
            namespace=self.namespace,
            genome_assembly=self.genome_assembly,
        )
        values = (
            self.effect,
            self.standard_error,
            self.statistic,
            self.p_value,
            self.adjusted_p_value,
        )
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("feature-effect numerical values must be finite")
        return self


class FeatureEffectBatch(BaseModel):
    """Feature effects plus the independent-subject estimand and model context."""

    model_config = ConfigDict(frozen=True)

    effects: tuple[FeatureEffect, ...]
    contrast: str = Field(min_length=1)
    estimand_population: str = Field(min_length=1)
    time_contrast: str = Field(min_length=1)
    treated_subjects: int = Field(gt=0)
    control_subjects: int = Field(gt=0)
    independent_subject_definition: str = Field(min_length=1)
    design_formula: str = Field(min_length=1)
    covariates: tuple[str, ...] = ()
    normalization_method: str = Field(min_length=1)
    inference_method: str = Field(min_length=1)
    inference_version: str = Field(min_length=1)
    tested_feature_ids: tuple[str, ...]
    multiple_testing_method: str = Field(min_length=1)
    effect_provenance_id: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        """Require one coherent contrast and an explicit tested universe."""
        if not self.effects:
            raise ValueError("feature-effect batch cannot be empty")
        feature_ids = [item.feature_id for item in self.effects]
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("feature-effect batch identifiers must be unique")
        if not self.tested_feature_ids or len(set(self.tested_feature_ids)) != len(
            self.tested_feature_ids
        ):
            raise ValueError("tested feature identifiers must be nonempty and unique")
        if not set(feature_ids).issubset(self.tested_feature_ids):
            raise ValueError("feature effects must be a subset of the tested feature universe")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("feature-effect covariates must be unique")
        if any(item.contrast != self.contrast for item in self.effects):
            raise ValueError("every feature effect must match the batch contrast")
        if any(item.provenance_id != self.effect_provenance_id for item in self.effects):
            raise ValueError("every feature effect must match the batch effect_provenance_id")
        domains = {
            (
                item.feature_type,
                item.namespace,
                item.modality,
                item.contrast,
                item.effect_unit,
                item.species_taxon_id,
                item.tissue,
                item.genome_assembly,
            )
            for item in self.effects
        }
        if len(domains) != 1:
            raise ValueError("feature-effect batch must describe one coherent upstream contrast")
        return self

    @property
    def artifact_hash(self) -> str:
        """Hash every algorithm-relevant field with set-like inputs canonicalized."""
        payload = self.model_dump(mode="json")
        payload["effects"] = sorted(payload["effects"], key=lambda item: item["feature_id"])
        payload["tested_feature_ids"] = sorted(payload["tested_feature_ids"])
        payload["covariates"] = sorted(payload["covariates"])
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def read_feature_effects(
    frame: pd.DataFrame,
    *,
    feature_id_column: str,
    effect_column: str,
    standard_error_column: str,
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
    modality: Modality,
    contrast: str,
    effect_unit: str,
    species_taxon_id: int,
    tissue: str,
    provenance_id: str,
    statistic_column: str | None = None,
    p_value_column: str | None = None,
    adjusted_p_value_column: str | None = None,
    genome_assembly: str | None = None,
) -> tuple[FeatureEffect, ...]:
    """Read DESeq2/edgeR/limma-style tables through explicit column mappings."""
    required = {feature_id_column, effect_column, standard_error_column}
    optional = {
        value
        for value in (statistic_column, p_value_column, adjusted_p_value_column)
        if value is not None
    }
    missing = sorted((required | optional).difference(frame.columns))
    if missing:
        raise ValueError(f"feature-effect columns are absent: {missing}")
    identifiers: list[str] = []
    for value in frame[feature_id_column]:
        if pd.isna(value):
            raise ValueError("feature-effect identifiers cannot be missing")
        identifier = str(value)
        if not identifier or identifier != identifier.strip():
            raise ValueError(
                "feature-effect identifiers must be nonempty without surrounding whitespace"
            )
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("feature-effect identifiers must be unique")
    effects: list[FeatureEffect] = []
    for row_index, identifier in enumerate(identifiers):
        row = frame.iloc[row_index]
        effects.append(
            FeatureEffect(
                feature_id=identifier,
                feature_type=feature_type,
                namespace=namespace,
                modality=modality,
                contrast=contrast,
                effect=float(row[effect_column]),
                standard_error=float(row[standard_error_column]),
                effect_unit=effect_unit,
                statistic=_optional_float(row, statistic_column),
                p_value=_optional_float(row, p_value_column),
                adjusted_p_value=_optional_float(row, adjusted_p_value_column),
                species_taxon_id=species_taxon_id,
                tissue=tissue,
                genome_assembly=genome_assembly,
                provenance_id=provenance_id,
            )
        )
    if not effects:
        raise ValueError("feature-effect table is empty")
    return tuple(effects)


def _optional_float(row: pd.Series, column: str | None) -> float | None:
    if column is None or pd.isna(row[column]):
        return None
    return float(row[column])
