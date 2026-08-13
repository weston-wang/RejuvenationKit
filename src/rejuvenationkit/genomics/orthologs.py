"""Versioned, ambiguity-reporting cross-species signature translation."""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from math import isfinite
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.genomics.schemas import FeatureNamespace
from rejuvenationkit.genomics.signatures import (
    GeneSignature,
    SignatureFeature,
)


class OrthologAmbiguityPolicy(StrEnum):
    """Handling of one-to-many source-to-target ortholog relationships."""

    ERROR = "error"
    DROP_AMBIGUOUS = "drop_ambiguous"
    SELECT_HIGHEST_CONFIDENCE = "select_highest_confidence"
    DISTRIBUTE_WEIGHT = "distribute_weight"


class OrthologRecord(BaseModel):
    """One versioned source-to-target ortholog assertion."""

    model_config = ConfigDict(frozen=True)

    source_feature_id: str = Field(min_length=1)
    target_feature_id: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    relationship: str = Field(default="ortholog", min_length=1)
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        """Reject non-finite confidence and duplicate evidence identifiers."""
        if not isfinite(self.confidence):
            raise ValueError("ortholog confidence must be finite")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("ortholog evidence_ids must be unique")
        return self


class OrthologMap(BaseModel):
    """Local, immutable mapping resource with explicit species and namespaces."""

    model_config = ConfigDict(frozen=True)

    source_species_taxon_id: int = Field(gt=0)
    target_species_taxon_id: int = Field(gt=0)
    source_namespace: FeatureNamespace
    target_namespace: FeatureNamespace
    resource_id: str = Field(min_length=1)
    resource_version: str = Field(min_length=1)
    records: tuple[OrthologRecord, ...]

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        """Require nonempty unique mapping assertions and distinct species."""
        if self.source_species_taxon_id == self.target_species_taxon_id:
            raise ValueError("ortholog map source and target species must differ")
        if not self.records:
            raise ValueError("ortholog map must contain records")
        pairs = [(item.source_feature_id, item.target_feature_id) for item in self.records]
        if len(set(pairs)) != len(pairs):
            raise ValueError("ortholog source-target pairs must be unique")
        return self


class OrthologTranslation(BaseModel):
    """Translated signature and complete mapping-loss/ambiguity audit."""

    model_config = ConfigDict(frozen=True)

    signature: GeneSignature
    source_signature_id: str
    mapping_resource_id: str
    mapping_resource_version: str
    policy: OrthologAmbiguityPolicy
    allowed_relationships: tuple[str, ...]
    mapped_source_feature_ids: tuple[str, ...]
    missing_source_feature_ids: tuple[str, ...]
    ambiguous_source_feature_ids: tuple[str, ...]
    dropped_source_feature_ids: tuple[str, ...]
    selected_pairs: tuple[tuple[str, str], ...]
    source_absolute_weight: float = Field(gt=0)
    retained_absolute_weight: float = Field(ge=0)
    retained_weight_fraction: float = Field(ge=0, le=1)
    post_aggregation_absolute_weight: float = Field(ge=0)
    post_aggregation_weight_fraction: float = Field(ge=0, le=1)
    warnings: tuple[str, ...] = ()


def translate_signature(
    signature: GeneSignature,
    mapping: OrthologMap,
    *,
    policy: OrthologAmbiguityPolicy = OrthologAmbiguityPolicy.ERROR,
    minimum_confidence: float = 0.0,
    minimum_retained_weight_fraction: float = 0.7,
    target_tissue: str | None = None,
    allowed_relationships: tuple[str, ...] = ("ortholog",),
) -> OrthologTranslation:
    """Translate feature weights without hiding mapping loss or one-to-many choices."""
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must lie in [0, 1]")
    if not 0 < minimum_retained_weight_fraction <= 1:
        raise ValueError("minimum_retained_weight_fraction must lie in (0, 1]")
    if not allowed_relationships or len(set(allowed_relationships)) != len(allowed_relationships):
        raise ValueError("allowed_relationships must be nonempty and unique")
    if signature.species_taxon_id != mapping.source_species_taxon_id:
        raise ValueError("signature species does not match ortholog-map source species")
    if signature.namespace is not mapping.source_namespace:
        raise ValueError("signature namespace does not match ortholog-map source namespace")

    by_source: defaultdict[str, list[OrthologRecord]] = defaultdict(list)
    relationships_excluded = False
    for record in mapping.records:
        if record.relationship not in allowed_relationships:
            relationships_excluded = True
            continue
        if record.confidence >= minimum_confidence:
            by_source[record.source_feature_id].append(record)
    translated_weights: defaultdict[str, float] = defaultdict(float)
    mapped: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    dropped: list[str] = []
    selected_pairs: list[tuple[str, str]] = []
    retained_source_weight = 0.0
    for feature in signature.features:
        candidates = sorted(
            by_source.get(feature.feature_id, ()),
            key=lambda item: (-item.confidence, item.target_feature_id),
        )
        if not candidates:
            missing.append(feature.feature_id)
            dropped.append(feature.feature_id)
            continue
        if len(candidates) == 1:
            selected = candidates
        else:
            ambiguous.append(feature.feature_id)
            if policy is OrthologAmbiguityPolicy.ERROR:
                targets = [item.target_feature_id for item in candidates]
                raise ValueError(f"ambiguous ortholog mapping for {feature.feature_id}: {targets}")
            if policy is OrthologAmbiguityPolicy.DROP_AMBIGUOUS:
                dropped.append(feature.feature_id)
                continue
            if policy is OrthologAmbiguityPolicy.SELECT_HIGHEST_CONFIDENCE:
                selected = candidates[:1]
            else:
                selected = candidates
        mapped.append(feature.feature_id)
        retained_source_weight += abs(feature.weight)
        divided_weight = feature.weight / len(selected)
        for record in selected:
            translated_weights[record.target_feature_id] += divided_weight
            selected_pairs.append((feature.feature_id, record.target_feature_id))

    source_weight = sum(abs(item.weight) for item in signature.features)
    retained_fraction = retained_source_weight / source_weight
    if retained_fraction < minimum_retained_weight_fraction:
        raise ValueError(
            f"ortholog retained-weight fraction {retained_fraction:.3f} is below minimum "
            f"{minimum_retained_weight_fraction:.3f}"
        )
    translated_features = tuple(
        SignatureFeature(feature_id=feature_id, weight=weight)
        for feature_id, weight in sorted(translated_weights.items())
        if weight != 0
    )
    if not translated_features:
        raise ValueError("ortholog translation produced no nonzero target features")
    post_aggregation_weight = sum(abs(item.weight) for item in translated_features)
    post_aggregation_fraction = min(1.0, post_aggregation_weight / source_weight)
    target_ids = [target for _, target in selected_pairs]
    target_collisions = len(set(target_ids)) < len(target_ids)
    warnings: list[str] = []
    if missing:
        warnings.append("ortholog_features_missing")
    if ambiguous:
        warnings.append("one_to_many_orthologs_present")
    if dropped:
        warnings.append("ortholog_features_dropped")
    if target_collisions:
        warnings.append("multiple_source_features_share_target_ortholog")
    if relationships_excluded:
        warnings.append("ortholog_relationships_excluded")
    translated = GeneSignature(
        signature_id=(
            f"{signature.signature_id}:{mapping.source_species_taxon_id}-to-"
            f"{mapping.target_species_taxon_id}"
        ),
        version=(f"{signature.version}+ortholog-{mapping.resource_id}-{mapping.resource_version}"),
        name=f"{signature.name} ({mapping.target_species_taxon_id} ortholog translation)",
        features=translated_features,
        namespace=mapping.target_namespace,
        species_taxon_id=mapping.target_species_taxon_id,
        tissue=target_tissue if target_tissue is not None else signature.tissue,
        target_name=signature.target_name,
        target_unit=signature.target_unit,
        direction=signature.direction,
        resource_id=(
            f"{signature.resource_id};ortholog={mapping.resource_id}:{mapping.resource_version}"
        ),
        allowed_scales=signature.allowed_scales,
    )
    return OrthologTranslation(
        signature=translated,
        source_signature_id=signature.signature_id,
        mapping_resource_id=mapping.resource_id,
        mapping_resource_version=mapping.resource_version,
        policy=policy,
        allowed_relationships=allowed_relationships,
        mapped_source_feature_ids=tuple(mapped),
        missing_source_feature_ids=tuple(missing),
        ambiguous_source_feature_ids=tuple(ambiguous),
        dropped_source_feature_ids=tuple(dropped),
        selected_pairs=tuple(selected_pairs),
        source_absolute_weight=source_weight,
        retained_absolute_weight=retained_source_weight,
        retained_weight_fraction=retained_fraction,
        post_aggregation_absolute_weight=post_aggregation_weight,
        post_aggregation_weight_fraction=post_aggregation_fraction,
        warnings=tuple(warnings),
    )
