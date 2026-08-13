"""Typed ingestion of assay-pipeline feature effects."""

from __future__ import annotations

from math import isfinite
from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType
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
        """Reject invalid numerical outputs from upstream feature models."""
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
    identifiers = tuple(str(value) for value in frame[feature_id_column])
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
