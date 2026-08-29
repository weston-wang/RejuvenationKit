"""Provider-neutral contracts for imported interaction-network snapshots.

Network-database scores describe database records.  They are not subject-level
experimental estimates and cannot be converted into fusion evidence.
"""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Any, Literal, Self, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    canonical_sha256,
)


class InteractionDirection(StrEnum):
    """Direction semantics attached to a provider interaction record."""

    UNDIRECTED = "undirected"
    DIRECTED = "directed"
    BIDIRECTIONAL = "bidirectional"
    UNKNOWN = "unknown"


class InteractionSign(StrEnum):
    """Signed interpretation supplied by the interaction resource."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNSIGNED = "unsigned"
    UNKNOWN = "unknown"


class SelfLoopPolicy(StrEnum):
    """Explicit handling of provider records whose endpoints are identical."""

    ERROR = "error"
    DROP = "drop"
    ALLOW = "allow"


class EvidenceChannelColumn(BaseModel):
    """Explicit mapping from one DataFrame column to one evidence channel."""

    model_config = ConfigDict(frozen=True)

    channel_id: str = Field(min_length=1)
    column: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class InteractionNetworkColumns(BaseModel):
    """Explicit provider-table column mapping; no names are guessed."""

    model_config = ConfigDict(frozen=True)

    source_feature_id: str = Field(min_length=1)
    target_feature_id: str = Field(min_length=1)
    source_provider_id: str = Field(min_length=1)
    target_provider_id: str = Field(min_length=1)
    confidence: str = Field(min_length=1)
    relation_type: str | None = Field(default=None, min_length=1)
    provider_record_id: str | None = Field(default=None, min_length=1)
    direction: str | None = Field(default=None, min_length=1)
    sign: str | None = Field(default=None, min_length=1)
    evidence_channels: tuple[EvidenceChannelColumn, ...] = ()

    @model_validator(mode="after")
    def validate_evidence_channels(self) -> Self:
        """Reject ambiguous evidence-channel definitions."""
        channel_ids = tuple(item.channel_id for item in self.evidence_channels)
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("evidence-channel identifiers must be unique")
        channel_columns = tuple(item.column for item in self.evidence_channels)
        if len(set(channel_columns)) != len(channel_columns):
            raise ValueError("evidence-channel columns must be unique")
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return all referenced DataFrame columns in deterministic order."""
        values = [
            self.source_feature_id,
            self.target_feature_id,
            self.source_provider_id,
            self.target_provider_id,
            self.confidence,
        ]
        if self.direction is not None:
            values.append(self.direction)
        if self.sign is not None:
            values.append(self.sign)
        if self.relation_type is not None:
            values.append(self.relation_type)
        if self.provider_record_id is not None:
            values.append(self.provider_record_id)
        values.extend(item.column for item in self.evidence_channels)
        return tuple(dict.fromkeys(values))


class InteractionNode(BaseModel):
    """One provider-resolved node in an interaction-network snapshot."""

    model_config = ConfigDict(frozen=True)

    feature_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    domain: FeatureDomain


class InteractionEvidenceChannel(BaseModel):
    """One finite provider evidence-channel value and its meaning."""

    model_config = ConfigDict(frozen=True)

    channel_id: str = Field(min_length=1)
    value: float
    definition: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        """Reject missing or infinite channel scores without assuming a scale."""
        if not isfinite(value):
            raise ValueError("evidence-channel values must be finite")
        return value


class InteractionEdge(BaseModel):
    """One canonicalized provider interaction record."""

    model_config = ConfigDict(frozen=True)

    source_feature_id: str = Field(min_length=1)
    target_feature_id: str = Field(min_length=1)
    source_provider_id: str = Field(min_length=1)
    target_provider_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    direction: InteractionDirection
    sign: InteractionSign
    relation_type: str = Field(default="unspecified", min_length=1)
    provider_record_id: str | None = Field(default=None, min_length=1)
    evidence_channels: tuple[InteractionEvidenceChannel, ...] = ()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Require a normalized, finite database-confidence value."""
        if not isfinite(value):
            raise ValueError("interaction confidence must be finite")
        return value

    @model_validator(mode="after")
    def validate_channels_and_order(self) -> Self:
        """Require unique channels and canonical symmetric endpoints."""
        _require_clean_model_text(self.relation_type, "interaction relation_type")
        if self.provider_record_id is not None:
            _require_clean_model_text(
                self.provider_record_id,
                "interaction provider_record_id",
            )
        channel_ids = tuple(item.channel_id for item in self.evidence_channels)
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("interaction evidence-channel identifiers must be unique")
        if channel_ids != tuple(sorted(channel_ids)):
            raise ValueError("interaction evidence channels must be canonically ordered")
        if self.direction in (
            InteractionDirection.UNDIRECTED,
            InteractionDirection.BIDIRECTIONAL,
        ):
            source_key = (self.source_feature_id, self.source_provider_id)
            target_key = (self.target_feature_id, self.target_provider_id)
            if target_key < source_key:
                raise ValueError("symmetric interaction endpoints must be canonically ordered")
        return self

    @property
    def source_key(self) -> tuple[str, str]:
        """Return the source node key."""
        return self.source_feature_id, self.source_provider_id

    @property
    def target_key(self) -> tuple[str, str]:
        """Return the target node key."""
        return self.target_feature_id, self.target_provider_id


class InteractionNetwork(BaseModel):
    """Audited offline network import that is never fusion-ready evidence."""

    model_config = ConfigDict(frozen=True)

    network_id: str = Field(min_length=1)
    domain: FeatureDomain
    resource: ResourceSnapshot
    provenance: QueryProvenance
    seed_features: FeatureCollection
    matched_seed_feature_ids: tuple[str, ...]
    unmatched_seed_feature_ids: tuple[str, ...]
    nodes: tuple[InteractionNode, ...]
    edges: tuple[InteractionEdge, ...]
    confidence_definition: str = Field(min_length=1)
    minimum_confidence: float = Field(ge=0, le=1)
    self_loop_policy: SelfLoopPolicy
    input_row_count: int = Field(ge=0)
    below_threshold_edge_count: int = Field(ge=0)
    dropped_self_loop_count: int = Field(ge=0)
    complete: bool
    warnings: tuple[str, ...] = ()
    fusion_eligibility: Literal["not_fusible"] = "not_fusible"

    @field_validator("minimum_confidence")
    @classmethod
    def validate_minimum_confidence(cls, value: float) -> float:
        """Reject non-finite thresholds."""
        if not isfinite(value):
            raise ValueError("minimum_confidence must be finite")
        return value

    @model_validator(mode="after")
    def validate_network_contract(self) -> Self:
        """Validate domain, resource, audit partitions, and deterministic order."""
        nodes = tuple(
            InteractionNode.model_validate(item.model_dump(mode="python")) for item in self.nodes
        )
        edges = tuple(
            InteractionEdge.model_validate(item.model_dump(mode="python")) for item in self.edges
        )
        if self.seed_features.domain != self.domain:
            raise ValueError("seed feature collection domain must match network domain")
        if self.provenance.domain != self.domain:
            raise ValueError("query provenance domain must match network domain")
        if self.resource not in self.provenance.resources:
            raise ValueError("network resource must be present in query provenance resources")
        if self.provenance.input_hash != self.seed_features.content_hash:
            raise ValueError(
                "query provenance input_hash must match the seed feature collection content hash"
            )
        if self.complete != self.provenance.complete:
            raise ValueError("network completeness must match query provenance")
        if not self.complete and "interaction_network_response_truncated" not in self.warnings:
            raise ValueError("an incomplete network must report a truncation warning")

        node_keys = tuple((item.feature_id, item.provider_id) for item in nodes)
        if len(set(node_keys)) != len(node_keys):
            raise ValueError("interaction nodes must be unique")
        if node_keys != tuple(sorted(node_keys)):
            raise ValueError("interaction nodes must be canonically ordered")
        if any(node.domain != self.domain for node in nodes):
            raise ValueError("every interaction node must use the exact network domain")

        edge_keys = tuple(_edge_sort_key(item) for item in edges)
        if edge_keys != tuple(sorted(edge_keys)):
            raise ValueError("interaction edges must be canonically ordered")
        duplicate_keys = tuple(_duplicate_edge_key(item) for item in edges)
        if len(set(duplicate_keys)) != len(duplicate_keys):
            raise ValueError("interaction edges must not contain duplicate endpoint records")
        node_key_set = set(node_keys)
        for edge in edges:
            if edge.source_key not in node_key_set or edge.target_key not in node_key_set:
                raise ValueError("every interaction edge endpoint must reference a network node")

        requested = set(self.seed_features.feature_ids)
        matched = set(self.matched_seed_feature_ids)
        unmatched = set(self.unmatched_seed_feature_ids)
        if matched & unmatched or matched | unmatched != requested:
            raise ValueError("matched and unmatched seeds must partition the requested seeds")
        if not matched.issubset({item.feature_id for item in nodes}):
            raise ValueError("matched seeds must be present among retained network nodes")
        if self.matched_seed_feature_ids != tuple(sorted(self.matched_seed_feature_ids)):
            raise ValueError("matched seed identifiers must be canonically ordered")
        if self.unmatched_seed_feature_ids != tuple(sorted(self.unmatched_seed_feature_ids)):
            raise ValueError("unmatched seed identifiers must be canonically ordered")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("network warnings must be unique")
        if self.provenance.warnings != self.warnings:
            raise ValueError("network warnings must match query provenance")
        accounted_rows = len(edges) + self.below_threshold_edge_count + self.dropped_self_loop_count
        if accounted_rows != self.input_row_count:
            raise ValueError("network row audit must account for every input row exactly")
        normalized_hash = _network_response_hash(
            network_id=self.network_id,
            domain=self.domain,
            resource=self.resource,
            seed_features=self.seed_features,
            nodes=nodes,
            edges=edges,
            matched_seed_feature_ids=self.matched_seed_feature_ids,
            unmatched_seed_feature_ids=self.unmatched_seed_feature_ids,
            confidence_definition=self.confidence_definition,
            minimum_confidence=self.minimum_confidence,
            self_loop_policy=self.self_loop_policy,
            input_row_count=self.input_row_count,
            below_threshold_edge_count=self.below_threshold_edge_count,
            dropped_self_loop_count=self.dropped_self_loop_count,
            complete=self.complete,
            warnings=self.warnings,
        )
        if self.provenance.response_checksum != normalized_hash:
            raise ValueError(
                "network response checksum does not match the normalized imported network"
            )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        return self


def load_interaction_network_frame(
    frame: pd.DataFrame,
    *,
    network_id: str,
    columns: InteractionNetworkColumns,
    domain: FeatureDomain,
    resource: ResourceSnapshot,
    provenance: QueryProvenance,
    seed_features: FeatureCollection,
    confidence_definition: str,
    minimum_confidence: float = 0.0,
    default_direction: InteractionDirection = InteractionDirection.UNDIRECTED,
    default_sign: InteractionSign = InteractionSign.UNSIGNED,
    default_relation_type: str = "unspecified",
    self_loop_policy: SelfLoopPolicy = SelfLoopPolicy.ERROR,
    response_truncated: bool = False,
) -> InteractionNetwork:
    """Import a provider export without making provider confidence fusible.

    Reversed undirected or bidirectional edges canonicalize to the same endpoint
    key and are rejected as duplicate database records.  This intentionally
    avoids treating repeated provider rows as independent evidence.
    """
    if seed_features.domain != domain:
        raise ValueError("seed feature collection domain must match network domain")
    if provenance.domain != domain:
        raise ValueError("query provenance domain must match network domain")
    if resource not in provenance.resources:
        raise ValueError("network resource must be present in query provenance resources")
    if provenance.input_hash != seed_features.content_hash:
        raise ValueError(
            "query provenance input_hash must match the seed feature collection content hash"
        )
    if not isinstance(minimum_confidence, (int, float)) or isinstance(minimum_confidence, bool):
        raise TypeError("minimum_confidence must be a real number")
    threshold = float(minimum_confidence)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("minimum_confidence must be finite and lie in [0, 1]")
    if not confidence_definition.strip():
        raise ValueError("confidence_definition must be nonempty")
    _require_clean_model_text(default_relation_type, "default_relation_type")
    if frame.columns.duplicated().any():
        raise ValueError("interaction DataFrame column labels must be unique")

    missing_columns = sorted(set(columns.required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"interaction DataFrame columns are absent: {missing_columns}")

    nodes_by_key: dict[tuple[str, str], InteractionNode] = {}
    provider_id_by_feature: dict[str, str] = {}
    edges_by_key: dict[tuple[object, ...], InteractionEdge] = {}
    below_threshold = 0
    dropped_self_loops = 0
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        source_feature_id = _required_text(row[columns.source_feature_id], row_number, "source")
        target_feature_id = _required_text(row[columns.target_feature_id], row_number, "target")
        source_provider_id = _required_text(
            row[columns.source_provider_id], row_number, "source provider"
        )
        target_provider_id = _required_text(
            row[columns.target_provider_id], row_number, "target provider"
        )
        confidence = _normalized_confidence(row[columns.confidence], row_number)
        direction = (
            default_direction
            if columns.direction is None
            else _parse_direction(row[columns.direction], row_number)
        )
        sign = default_sign if columns.sign is None else _parse_sign(row[columns.sign], row_number)
        relation_type = (
            default_relation_type
            if columns.relation_type is None
            else _required_text(row[columns.relation_type], row_number, "relation type")
        )
        provider_record_id = (
            None
            if columns.provider_record_id is None
            else _required_text(
                row[columns.provider_record_id],
                row_number,
                "provider record",
            )
        )

        _register_provider_mapping(
            provider_id_by_feature, source_feature_id, source_provider_id, row_number
        )
        _register_provider_mapping(
            provider_id_by_feature, target_feature_id, target_provider_id, row_number
        )

        if source_feature_id == target_feature_id:
            if self_loop_policy is SelfLoopPolicy.ERROR:
                raise ValueError(
                    f"interaction row {row_number} contains a self-loop for {source_feature_id}"
                )
            if self_loop_policy is SelfLoopPolicy.DROP:
                dropped_self_loops += 1
                continue
        if confidence < threshold:
            below_threshold += 1
            continue

        source_key = (source_feature_id, source_provider_id)
        target_key = (target_feature_id, target_provider_id)
        if direction in (InteractionDirection.UNDIRECTED, InteractionDirection.BIDIRECTIONAL):
            if target_key < source_key:
                source_key, target_key = target_key, source_key
        evidence_channels = tuple(
            sorted(
                (
                    InteractionEvidenceChannel(
                        channel_id=item.channel_id,
                        value=_finite_channel_value(row[item.column], row_number, item.channel_id),
                        definition=item.definition,
                    )
                    for item in columns.evidence_channels
                    if not _is_missing(row[item.column])
                ),
                key=lambda item: item.channel_id,
            )
        )
        edge = InteractionEdge(
            source_feature_id=source_key[0],
            target_feature_id=target_key[0],
            source_provider_id=source_key[1],
            target_provider_id=target_key[1],
            confidence=confidence,
            direction=direction,
            sign=sign,
            relation_type=relation_type,
            provider_record_id=provider_record_id,
            evidence_channels=evidence_channels,
        )
        duplicate_key = _duplicate_edge_key(edge)
        if duplicate_key in edges_by_key:
            raise ValueError(
                "duplicate interaction edge after canonicalization: "
                f"{edge.source_feature_id}--{edge.target_feature_id} "
                f"({edge.relation_type}, {edge.direction.value}, {edge.sign.value})"
            )
        edges_by_key[duplicate_key] = edge
        nodes_by_key.setdefault(
            source_key,
            InteractionNode(feature_id=source_key[0], provider_id=source_key[1], domain=domain),
        )
        nodes_by_key.setdefault(
            target_key,
            InteractionNode(feature_id=target_key[0], provider_id=target_key[1], domain=domain),
        )

    nodes = tuple(nodes_by_key[key] for key in sorted(nodes_by_key))
    edges = tuple(sorted(edges_by_key.values(), key=_edge_sort_key))
    retained_feature_ids = {item.feature_id for item in nodes}
    requested_seed_ids = set(seed_features.feature_ids)
    matched = tuple(sorted(requested_seed_ids & retained_feature_ids))
    unmatched = tuple(sorted(requested_seed_ids - retained_feature_ids))

    warnings = list(provenance.warnings)
    if response_truncated or not provenance.complete:
        _append_once(warnings, "interaction_network_response_truncated")
    if unmatched:
        _append_once(warnings, f"unmatched_seed_features:{len(unmatched)}")
    if below_threshold:
        _append_once(warnings, f"interaction_edges_below_confidence_threshold:{below_threshold}")
    if dropped_self_loops:
        _append_once(warnings, f"interaction_self_loops_dropped:{dropped_self_loops}")
    complete = provenance.complete and not response_truncated
    normalized_hash = _network_response_hash(
        network_id=network_id,
        domain=domain,
        resource=resource,
        seed_features=seed_features,
        nodes=nodes,
        edges=edges,
        matched_seed_feature_ids=matched,
        unmatched_seed_feature_ids=unmatched,
        confidence_definition=confidence_definition.strip(),
        minimum_confidence=threshold,
        self_loop_policy=self_loop_policy,
        input_row_count=len(frame),
        below_threshold_edge_count=below_threshold,
        dropped_self_loop_count=dropped_self_loops,
        complete=complete,
        warnings=tuple(warnings),
    )
    if provenance.response_checksum is not None and provenance.response_checksum != normalized_hash:
        raise ValueError("supplied response checksum does not match normalized network")
    normalized_provenance = provenance.model_copy(
        update={
            "response_checksum": normalized_hash,
            "complete": complete,
            "warnings": tuple(warnings),
        }
    )

    return InteractionNetwork(
        network_id=network_id,
        domain=domain,
        resource=resource,
        provenance=normalized_provenance,
        seed_features=seed_features,
        matched_seed_feature_ids=matched,
        unmatched_seed_feature_ids=unmatched,
        nodes=nodes,
        edges=edges,
        confidence_definition=confidence_definition.strip(),
        minimum_confidence=threshold,
        self_loop_policy=self_loop_policy,
        input_row_count=len(frame),
        below_threshold_edge_count=below_threshold,
        dropped_self_loop_count=dropped_self_loops,
        complete=complete,
        warnings=tuple(warnings),
    )


def _required_text(value: object, row_number: int, label: str) -> str:
    if _is_missing(value):
        raise ValueError(f"interaction row {row_number} has a missing {label} identifier")
    text = str(value).strip()
    if not text:
        raise ValueError(f"interaction row {row_number} has an empty {label} identifier")
    return text


def _normalized_confidence(value: object, row_number: int) -> float:
    if isinstance(value, bool) or _is_missing(value):
        raise ValueError(f"interaction row {row_number} confidence must be numeric")
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"interaction row {row_number} confidence must be numeric") from error
    if not isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(
            f"interaction row {row_number} confidence must be finite and lie in [0, 1]"
        )
    return numeric


def _finite_channel_value(value: object, row_number: int, channel_id: str) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"interaction row {row_number} evidence channel {channel_id} must be numeric"
        )
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"interaction row {row_number} evidence channel {channel_id} must be numeric"
        ) from error
    if not isfinite(numeric):
        raise ValueError(
            f"interaction row {row_number} evidence channel {channel_id} must be finite"
        )
    return numeric


def _parse_direction(value: object, row_number: int) -> InteractionDirection:
    raw = _required_text(value, row_number, "direction").lower()
    if raw == "->":
        return InteractionDirection.DIRECTED
    text = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "undirected": InteractionDirection.UNDIRECTED,
        "directed": InteractionDirection.DIRECTED,
        "source_to_target": InteractionDirection.DIRECTED,
        "bidirectional": InteractionDirection.BIDIRECTIONAL,
        "both": InteractionDirection.BIDIRECTIONAL,
        "unknown": InteractionDirection.UNKNOWN,
        "unspecified": InteractionDirection.UNKNOWN,
    }
    try:
        return aliases[text]
    except KeyError as error:
        raise ValueError(
            f"interaction row {row_number} has unknown direction: {value!r}"
        ) from error


def _parse_sign(value: object, row_number: int) -> InteractionSign:
    raw = _required_text(value, row_number, "sign").lower()
    if raw == "-":
        return InteractionSign.NEGATIVE
    text = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "positive": InteractionSign.POSITIVE,
        "activation": InteractionSign.POSITIVE,
        "+": InteractionSign.POSITIVE,
        "negative": InteractionSign.NEGATIVE,
        "inhibition": InteractionSign.NEGATIVE,
        "unsigned": InteractionSign.UNSIGNED,
        "none": InteractionSign.UNSIGNED,
        "unknown": InteractionSign.UNKNOWN,
        "unspecified": InteractionSign.UNKNOWN,
    }
    try:
        return aliases[text]
    except KeyError as error:
        raise ValueError(f"interaction row {row_number} has unknown sign: {value!r}") from error


def _duplicate_edge_key(edge: InteractionEdge) -> tuple[object, ...]:
    """Return one exact provider-record identity in the typed multigraph."""
    return (
        edge.source_feature_id,
        edge.source_provider_id,
        edge.target_feature_id,
        edge.target_provider_id,
        edge.relation_type,
        edge.direction.value,
        edge.sign.value,
        edge.provider_record_id,
    )


def _edge_sort_key(edge: InteractionEdge) -> tuple[object, ...]:
    """Return a deterministic total order independent of provider row order."""
    channel_key = tuple(
        (item.channel_id, item.value, item.definition) for item in edge.evidence_channels
    )
    return (
        edge.source_feature_id,
        edge.source_provider_id,
        edge.target_feature_id,
        edge.target_provider_id,
        edge.relation_type,
        edge.direction.value,
        edge.sign.value,
        edge.provider_record_id or "",
        edge.confidence,
        channel_key,
    )


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _network_response_hash(
    *,
    network_id: str,
    domain: FeatureDomain,
    resource: ResourceSnapshot,
    seed_features: FeatureCollection,
    nodes: tuple[InteractionNode, ...],
    edges: tuple[InteractionEdge, ...],
    matched_seed_feature_ids: tuple[str, ...],
    unmatched_seed_feature_ids: tuple[str, ...],
    confidence_definition: str,
    minimum_confidence: float,
    self_loop_policy: SelfLoopPolicy,
    input_row_count: int,
    below_threshold_edge_count: int,
    dropped_self_loop_count: int,
    complete: bool,
    warnings: tuple[str, ...],
) -> str:
    """Hash the canonical typed import separately from archived provider bytes."""
    return canonical_sha256(
        {
            "network_id": network_id,
            "domain_hash": domain.domain_hash,
            "resource_snapshot_id": resource.snapshot_id,
            "seed_feature_hash": seed_features.content_hash,
            "nodes": [item.model_dump(mode="json") for item in nodes],
            "edges": [item.model_dump(mode="json") for item in edges],
            "matched_seed_feature_ids": sorted(matched_seed_feature_ids),
            "unmatched_seed_feature_ids": sorted(unmatched_seed_feature_ids),
            "confidence_definition": confidence_definition,
            "minimum_confidence": minimum_confidence,
            "self_loop_policy": self_loop_policy.value,
            "input_row_count": input_row_count,
            "below_threshold_edge_count": below_threshold_edge_count,
            "dropped_self_loop_count": dropped_self_loop_count,
            "complete": complete,
            "warnings": sorted(warnings),
        }
    )


def _register_provider_mapping(
    provider_id_by_feature: dict[str, str],
    feature_id: str,
    provider_id: str,
    row_number: int,
) -> None:
    """Reject one canonical feature resolving to several provider nodes."""
    previous = provider_id_by_feature.setdefault(feature_id, provider_id)
    if previous != provider_id:
        raise ValueError(
            f"interaction row {row_number} maps feature {feature_id} to multiple provider IDs: "
            f"{previous!r} and {provider_id!r}"
        )


def _is_missing(value: object) -> bool:
    """Apply pandas scalar missingness semantics to one DataFrame cell."""
    return bool(pd.isna(cast(Any, value)))


def _require_clean_model_text(value: str, field_name: str) -> None:
    """Reject empty or whitespace-padded text in direct model construction."""
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonempty without surrounding whitespace")
