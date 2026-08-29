"""Tests for provider-neutral offline interaction-network imports."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.networks import (
    EvidenceChannelColumn,
    InteractionDirection,
    InteractionEdge,
    InteractionNetwork,
    InteractionNetworkColumns,
    InteractionSign,
    SelfLoopPolicy,
    load_interaction_network_frame,
)
from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
)
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType

NOW = datetime(2026, 8, 15, tzinfo=UTC)


def domain(*, species_taxon_id: int = 9615) -> FeatureDomain:
    """Return one canine Ensembl-gene domain."""
    return FeatureDomain(
        species_taxon_id=species_taxon_id,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        genome_assembly="CanFam3.1",
    )


def resource() -> ResourceSnapshot:
    """Return one immutable interaction-resource snapshot."""
    return ResourceSnapshot(
        provider_id="offline-test-provider",
        resource_id="interaction-db",
        resource_release="2026-08",
        retrieved_at=NOW,
        response_sha256="a" * 64,
        source_uri="https://example.invalid/export.tsv",
        license_id="CC-BY-4.0",
        citation_ids=("PMID:test",),
    )


def provenance(
    *,
    feature_domain: FeatureDomain | None = None,
    snapshot: ResourceSnapshot | None = None,
    complete: bool = True,
    input_hash: str | None = None,
    response_checksum: str | None = None,
    provider_id: str | None = None,
) -> QueryProvenance:
    """Return deterministic query provenance for network imports."""
    selected_resource = snapshot or resource()
    return QueryProvenance(
        provider_id=provider_id or selected_resource.provider_id,
        provider_version="test-client-1",
        resources=(selected_resource,),
        domain=feature_domain or domain(),
        retrieved_at=NOW,
        input_hash=input_hash or seeds(feature_domain=feature_domain).content_hash,
        response_checksum=response_checksum,
        query_parameters={"species": 9615},
        software_versions={"test-client": "1"},
        complete=complete,
    )


def seeds(*, feature_domain: FeatureDomain | None = None) -> FeatureCollection:
    """Return requested seed identifiers with one deliberately unmatched seed."""
    return FeatureCollection(
        collection_id="seed-panel-v1",
        domain=feature_domain or domain(),
        feature_ids=("ENSCAFG_A", "ENSCAFG_MISSING"),
        source_snapshot_id="prespecified-seed-snapshot-v1",
    )


def columns() -> InteractionNetworkColumns:
    """Return explicit mappings for the test provider table."""
    return InteractionNetworkColumns(
        source_feature_id="source",
        target_feature_id="target",
        source_provider_id="source_pid",
        target_provider_id="target_pid",
        confidence="combined_score",
        relation_type="relation",
        provider_record_id="record_id",
        direction="direction",
        sign="sign",
        evidence_channels=(
            EvidenceChannelColumn(
                channel_id="experimental",
                column="experimental_score",
                definition="normalized provider experimental-evidence channel",
            ),
            EvidenceChannelColumn(
                channel_id="text_mining",
                column="text_score",
                definition="normalized provider text-mining channel",
            ),
        ),
    )


def frame() -> pd.DataFrame:
    """Return a small network export in intentionally noncanonical row order."""
    return pd.DataFrame(
        {
            "source": ["ENSCAFG_C", "ENSCAFG_B"],
            "target": ["ENSCAFG_A", "ENSCAFG_C"],
            "source_pid": ["pC", "pB"],
            "target_pid": ["pA", "pC"],
            "combined_score": [0.91, 0.82],
            "relation": ["physical_association", "genetic_interaction"],
            "record_id": ["record-2", "record-1"],
            "direction": ["undirected", "undirected"],
            "sign": ["unsigned", "unsigned"],
            "experimental_score": [0.7, 0.6],
            "text_score": [0.4, 0.3],
        }
    )


def load(
    data: pd.DataFrame,
    *,
    feature_domain: FeatureDomain | None = None,
    selected_resource: ResourceSnapshot | None = None,
    query_provenance: QueryProvenance | None = None,
    seed_collection: FeatureCollection | None = None,
    minimum_confidence: float = 0.0,
    self_loop_policy: SelfLoopPolicy = SelfLoopPolicy.ERROR,
    response_truncated: bool = False,
) -> InteractionNetwork:
    """Load a network using coherent defaults."""
    selected_domain = feature_domain or domain()
    selected_resource = selected_resource or resource()
    selected_seeds = seed_collection or seeds(feature_domain=selected_domain)
    selected_provenance = query_provenance or provenance(
        feature_domain=selected_domain,
        snapshot=selected_resource,
        input_hash=selected_seeds.content_hash,
    )
    return load_interaction_network_frame(
        data,
        network_id="canine-network-v1",
        columns=columns(),
        domain=selected_domain,
        resource=selected_resource,
        provenance=selected_provenance,
        seed_features=selected_seeds,
        confidence_definition="provider combined association score rescaled to [0, 1]",
        minimum_confidence=minimum_confidence,
        self_loop_policy=self_loop_policy,
        response_truncated=response_truncated,
    )


def test_reversed_undirected_edges_are_rejected_as_duplicates() -> None:
    """Provider row duplication cannot masquerade as independent support."""
    data = pd.concat(
        [
            frame().iloc[[0]],
            frame()
            .iloc[[0]]
            .rename(
                columns={
                    "source": "target",
                    "target": "source",
                    "source_pid": "target_pid",
                    "target_pid": "source_pid",
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate interaction edge after canonicalization"):
        load(data)


def test_self_loop_policy_is_explicit_and_audited() -> None:
    """Self-loops error, drop with an audit, or remain only by explicit policy."""
    data = frame().iloc[[0]].copy()
    data.loc[:, "target"] = data["source"]
    data.loc[:, "target_pid"] = data["source_pid"]
    with pytest.raises(ValueError, match="contains a self-loop"):
        load(data)

    dropped = load(data, self_loop_policy=SelfLoopPolicy.DROP)
    assert dropped.edges == ()
    assert dropped.dropped_self_loop_count == 1
    assert "interaction_self_loops_dropped:1" in dropped.warnings

    retained = load(data, self_loop_policy=SelfLoopPolicy.ALLOW)
    assert len(retained.edges) == 1
    assert retained.edges[0].source_feature_id == retained.edges[0].target_feature_id


def test_network_rejects_seed_and_query_domain_mismatches() -> None:
    """Species, feature kind, namespace, and assembly remain exact contracts."""
    human = FeatureDomain(
        species_taxon_id=9606,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        genome_assembly="GRCh38",
    )
    with pytest.raises(ValueError, match="seed feature collection domain"):
        load(frame(), seed_collection=seeds(feature_domain=human))
    with pytest.raises(ValueError, match="query provenance domain"):
        load(frame(), query_provenance=provenance(feature_domain=human))


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, float("nan"), float("inf"), "bad"])
def test_provider_confidence_is_finite_and_normalized(bad_confidence: object) -> None:
    """The main provider score must honor its declared normalized scale."""
    data = frame().iloc[[0]].copy()
    data["combined_score"] = pd.Series([bad_confidence], index=data.index, dtype=object)
    with pytest.raises(ValueError, match="confidence"):
        load(data)


def test_direct_edge_model_rejects_invalid_confidence() -> None:
    """The frozen edge contract also validates confidence outside the loader."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        InteractionEdge(
            source_feature_id="A",
            target_feature_id="B",
            source_provider_id="pA",
            target_provider_id="pB",
            confidence=2,
            direction=InteractionDirection.UNDIRECTED,
            sign=InteractionSign.UNSIGNED,
        )


def test_direction_sign_and_provider_evidence_are_preserved() -> None:
    """Explicit directed signed records retain their provider semantics."""
    data = frame().iloc[[0]].copy()
    data.loc[:, "direction"] = "->"
    data.loc[:, "sign"] = "-"
    result = load(data)
    edge = result.edges[0]
    assert edge.direction is InteractionDirection.DIRECTED
    assert edge.sign is InteractionSign.NEGATIVE
    assert edge.relation_type == "physical_association"
    assert edge.provider_record_id == "record-2"
    assert edge.source_provider_id == "pC"
    assert tuple(channel.channel_id for channel in edge.evidence_channels) == (
        "experimental",
        "text_mining",
    )
    assert result.provenance.response_checksum is not None
    assert len(result.provenance.response_checksum) == 64


def test_truncation_is_incomplete_visible_and_not_fusible() -> None:
    """A truncated catalog response can never look like a complete network."""
    result = load(frame(), response_truncated=True)
    assert result.complete is False
    assert result.provenance.complete is False
    assert "interaction_network_response_truncated" in result.warnings
    assert result.fusion_eligibility == "not_fusible"
    assert not hasattr(result, "to_evidence")


def test_seed_and_threshold_audit_is_explicit() -> None:
    """Seeds filtered out by confidence remain visible as unmatched."""
    result = load(frame(), minimum_confidence=0.9)
    assert result.matched_seed_feature_ids == ("ENSCAFG_A",)
    assert result.unmatched_seed_feature_ids == ("ENSCAFG_MISSING",)
    assert result.below_threshold_edge_count == 1
    assert "interaction_edges_below_confidence_threshold:1" in result.warnings
    assert "unmatched_seed_features:1" in result.warnings


def test_network_row_audit_has_no_unexplained_input_rows() -> None:
    """Retained and explicitly filtered records must exhaust the provider table."""
    result = load(frame(), minimum_confidence=0.9)
    assert (
        len(result.edges) + result.below_threshold_edge_count + result.dropped_self_loop_count
        == result.input_row_count
    )
    payload = result.model_dump(mode="python")
    payload["input_row_count"] += 1
    with pytest.raises(ValidationError, match="every input row exactly"):
        InteractionNetwork.model_validate(payload)


def test_import_is_deterministic_across_row_order_and_endpoint_orientation() -> None:
    """Equivalent undirected exports yield byte-for-byte-equivalent model data."""
    first = load(frame())
    reversed_rows = frame().iloc[::-1].reset_index(drop=True)
    reversed_rows[["source", "target"]] = reversed_rows[["target", "source"]].to_numpy()
    reversed_rows[["source_pid", "target_pid"]] = reversed_rows[
        ["target_pid", "source_pid"]
    ].to_numpy()
    second = load(reversed_rows)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert tuple((node.feature_id, node.provider_id) for node in first.nodes) == (
        ("ENSCAFG_A", "pA"),
        ("ENSCAFG_B", "pB"),
        ("ENSCAFG_C", "pC"),
    )
    assert tuple(channel.channel_id for channel in first.edges[0].evidence_channels) == (
        "experimental",
        "text_mining",
    )


def test_resource_must_be_the_exact_query_snapshot() -> None:
    """Network imports cannot silently cite a resource absent from provenance."""
    selected_resource = resource()
    other = selected_resource.model_copy(update={"resource_release": "different-release"})
    with pytest.raises(ValueError, match="present in query provenance resources"):
        load(
            frame(),
            selected_resource=other,
            query_provenance=provenance(snapshot=selected_resource),
        )


def test_query_executor_and_scientific_resource_may_be_different_providers() -> None:
    """A connector may query a separately identified scientific database snapshot."""
    selected_resource = resource()
    result = load(
        frame(),
        selected_resource=selected_resource,
        query_provenance=provenance(
            snapshot=selected_resource,
            provider_id="smarts.bio/string-toolkit",
        ),
    )
    assert result.provenance.provider_id == "smarts.bio/string-toolkit"
    assert result.resource.provider_id == "offline-test-provider"


def test_network_rejects_a_stale_normalized_response_checksum() -> None:
    """The normalized result checksum cannot be reused after imported data change."""
    stale = provenance(response_checksum="d" * 64)
    with pytest.raises(ValueError, match="supplied response checksum"):
        load(frame(), query_provenance=stale)


def test_one_feature_cannot_silently_map_to_multiple_provider_nodes() -> None:
    """Canonical feature identity remains unambiguous within one snapshot."""
    data = frame()
    data.loc[data.index[1], "target_pid"] = "different-provider-C"
    with pytest.raises(ValueError, match="multiple provider IDs"):
        load(data)


def test_query_input_hash_must_bind_the_exact_seed_collection() -> None:
    """Provenance from another seed query cannot be attached to this result."""
    with pytest.raises(ValueError, match="input_hash must match the seed feature collection"):
        load(frame(), query_provenance=provenance(input_hash="f" * 64))


def test_typed_multigraph_preserves_distinct_parallel_relations_and_records() -> None:
    """Parallel edges coexist when relation or provider record identity differs."""
    base = frame().iloc[[0]].copy()
    second_relation = base.copy()
    second_relation.loc[:, "relation"] = "regulates_expression"
    second_relation.loc[:, "record_id"] = "record-3"
    second_provider_record = base.copy()
    second_provider_record.loc[:, "record_id"] = "record-4"
    data = pd.concat([base, second_relation, second_provider_record], ignore_index=True)

    result = load(data)

    assert len(result.edges) == 3
    assert {(edge.relation_type, edge.provider_record_id) for edge in result.edges} == {
        ("physical_association", "record-2"),
        ("physical_association", "record-4"),
        ("regulates_expression", "record-3"),
    }


def test_exact_typed_provider_record_duplicates_are_rejected() -> None:
    """Changing confidence cannot disguise a duplicate typed provider record."""
    duplicate = frame().iloc[[0]].copy()
    duplicate.loc[:, "combined_score"] = 0.55
    with pytest.raises(ValueError, match="duplicate interaction edge"):
        load(pd.concat([frame().iloc[[0]], duplicate], ignore_index=True))


def test_network_round_trip_rejects_relation_or_record_identity_forgery() -> None:
    """Typed multigraph semantics are bound by the normalized response checksum."""
    network = load(frame())
    assert InteractionNetwork.model_validate(network.model_dump(mode="python")) == network

    for field_name, forged_value in (
        ("relation_type", "forged_relation"),
        ("provider_record_id", "forged-record"),
    ):
        payload = network.model_dump(mode="python")
        payload["edges"][0][field_name] = forged_value
        with pytest.raises(ValidationError, match="normalized imported network"):
            InteractionNetwork.model_validate(payload)
