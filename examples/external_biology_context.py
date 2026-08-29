"""Import archived GO- and STRING-shaped results without treating them as efficacy evidence."""

from datetime import UTC, datetime
from hashlib import sha256

import pandas as pd

from rejuvenationkit.genomics import (
    DirectionalRankedSetRequest,
    EvidenceChannelColumn,
    FeatureCollection,
    FeatureDomain,
    FeatureNamespace,
    GenomicFeatureType,
    InteractionNetworkColumns,
    QueryProvenance,
    ResourceSnapshot,
    build_ranked_feature_universe,
    gene_set_collection_from_frame,
    load_interaction_network_frame,
    read_functional_annotations,
    run_directional_ranked_set_analysis,
    run_overrepresentation,
)
from rejuvenationkit.genomics.enrichment import OverrepresentationRequest

RETRIEVED_AT = datetime(2026, 8, 15, tzinfo=UTC)


def _archived_csv_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact deterministic CSV bytes retained as the archived response."""
    archived_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256(archived_bytes).hexdigest()


def functional_context() -> None:
    """Audit functional annotations and run directionless ORA on a measured universe."""
    domain = FeatureDomain(
        species_taxon_id=9606,
        feature_type=GenomicFeatureType.PROTEIN,
        namespace=FeatureNamespace.UNIPROT,
    )
    background_ids = (
        "O75385",
        "P23443",
        "P31749",
        "P42345",
        "P49815",
        "P62942",
        "Q13541",
        "Q14457",
        "Q15382",
        "Q8N122",
        "Q92574",
        "Q9GZQ8",
    )
    annotation_frame = pd.DataFrame(
        [
            ("P42345", "GO:0031929", "TOR signaling", "EXP"),
            ("P62942", "GO:0031929", "TOR signaling", "IDA"),
            ("Q8N122", "GO:0031929", "TOR signaling", "IPI"),
            ("P31749", "GO:0031929", "TOR signaling", "IMP"),
            ("Q15382", "GO:0031929", "TOR signaling", "IDA"),
            ("Q92574", "GO:0031929", "TOR signaling", "IMP"),
            ("P49815", "GO:0031929", "TOR signaling", "IMP"),
            ("O75385", "GO:0006914", "autophagy", "IDA"),
            ("Q14457", "GO:0006914", "autophagy", "IDA"),
            ("Q9GZQ8", "GO:0006914", "autophagy", "IDA"),
            ("P23443", "GO:0031929", "TOR signaling", "IDA"),
            ("Q13541", "GO:0031929", "TOR signaling", "IDA"),
        ],
        columns=["accession", "term_id", "term_name", "evidence_code"],
    ).assign(term_namespace="biological_process", relation="involved_in")
    raw_response_hash = _archived_csv_sha256(annotation_frame)
    resource = ResourceSnapshot(
        provider_id="go-toolkit",
        resource_id="gene-ontology-annotations",
        resource_release="archived-fixture-2026-08-15",
        retrieved_at=RETRIEVED_AT,
        response_sha256=raw_response_hash,
        source_uri="https://geneontology.org/",
    )
    background = FeatureCollection(
        collection_id="measured-proteins",
        domain=domain,
        feature_ids=background_ids,
        source_snapshot_id="proteomics-feature-universe-v1",
    )
    annotation_provenance = QueryProvenance(
        provider_id="go-toolkit",
        provider_version="1.0.0",
        resources=(resource,),
        domain=domain,
        retrieved_at=RETRIEVED_AT,
        input_hash=background.content_hash,
        query_parameters={
            "operation": "annotation",
            "organism": "9606",
            "aspect": "biological_process",
        },
        warnings=(
            "example_uses_archived_provider_shaped_fixture",
            "upstream_database_release_and_license_not_reported_by_provider",
        ),
    )
    annotations = read_functional_annotations(
        annotation_frame,
        query=background,
        provenance=annotation_provenance,
        feature_id_column="accession",
        term_id_column="term_id",
        term_name_column="term_name",
        term_namespace_column="term_namespace",
        relation_column="relation",
        evidence_code_column="evidence_code",
    )
    gene_sets = gene_set_collection_from_frame(
        annotation_frame,
        collection_id="go-biological-process-fixture",
        collection_name="Archived GO biological-process fixture",
        domain=domain,
        resource=resource,
        gene_set_id_column="term_id",
        gene_set_name_column="term_name",
        member_feature_id_column="accession",
    )
    selected = FeatureCollection(
        collection_id="prespecified-response-features",
        domain=domain,
        feature_ids=("P31749", "P42345", "P62942", "Q15382", "Q8N122"),
        source_snapshot_id="held-out-contrast-v1",
    )
    enrichment = run_overrepresentation(
        OverrepresentationRequest(
            selected_features=selected,
            background_universe=background,
            gene_sets=gene_sets,
            minimum_term_size=2,
            maximum_term_size=20,
        ),
        executed_at=RETRIEVED_AT,
    )
    signed_scores = {
        "O75385": -2.8,
        "P23443": 1.4,
        "P31749": 2.1,
        "P42345": 3.6,
        "P49815": 1.0,
        "P62942": 3.0,
        "Q13541": 0.8,
        "Q14457": -2.2,
        "Q15382": 2.5,
        "Q8N122": 3.2,
        "Q92574": 1.8,
        "Q9GZQ8": -1.9,
    }
    ranking_frame = pd.DataFrame(
        sorted(signed_scores.items()),
        columns=["accession", "signed_standardized_effect"],
    )
    ranking_resource = ResourceSnapshot(
        provider_id="archived-upstream-analysis",
        resource_id="synthetic-signed-protein-contrast",
        resource_release="fixture-v1",
        retrieved_at=RETRIEVED_AT,
        response_sha256=_archived_csv_sha256(ranking_frame),
        source_uri="https://example.org/rejuvenationkit/synthetic-ranked-protein-fixture.csv",
    )
    ranked_universe = FeatureCollection(
        collection_id="prespecified-ranked-proteins",
        domain=domain,
        feature_ids=background_ids,
        source_snapshot_id=ranking_resource.snapshot_id,
    )
    ranking = build_ranked_feature_universe(
        signed_scores,
        ranking_id="prespecified-signed-protein-fixture",
        ranking_method="archived_upstream_contrast",
        ranking_metric="signed_standardized_effect",
        higher_score_interpretation=(
            "greater abundance in the declared treated-minus-control contrast"
        ),
        feature_universe=ranked_universe,
        source_resource=ranking_resource,
        provider_id="archived-upstream-analysis",
        provider_version="1.0.0",
        executed_at=RETRIEVED_AT,
        software_versions={"fixture": "1.0.0"},
        warnings=("synthetic_ranked_scores_for_software_demonstration",),
    )
    directional = run_directional_ranked_set_analysis(
        DirectionalRankedSetRequest(
            ranking=ranking,
            gene_sets=gene_sets,
            permutation_count=1_000,
            random_seed=2718,
            minimum_term_size=2,
            maximum_term_size=20,
        ),
        executed_at=RETRIEVED_AT,
    )

    print("Functional annotation coverage")
    print(f"  matched: {len(annotations.matched_feature_ids)}/{len(background.feature_ids)}")
    print(f"  resource snapshot: {resource.snapshot_id[:12]}")
    print("Directionless overrepresentation")
    for term in enrichment.terms:
        print(
            f"  {term.gene_set_name}: overlap={term.selected_overlap_count}/"
            f"{term.background_gene_set_size}, fold={term.fold_enrichment:.2f}, "
            f"adjusted_p={term.adjusted_p_value:.4f}"
        )
    print(f"  fusion eligibility: {enrichment.fusion_eligibility}")
    print("Directional ranked-set context")
    for term in directional.terms:
        print(
            f"  {term.gene_set_name}: score={term.enrichment_score:+.3f}, "
            f"direction={term.direction}, adjusted_p={term.adjusted_p_value:.4f}"
        )
    print(f"  fusion eligibility: {directional.fusion_eligibility}")
    print("  ranking values are synthetic and are not treatment-efficacy evidence")


def interaction_context() -> None:
    """Import an archived STRING-shaped response with channel-level support."""
    network_frame = pd.DataFrame(
        [
            (
                "RPTOR",
                "MTOR",
                "9606.ENSP00000307272",
                "9606.ENSP00000354558",
                0.999,
                0.301,
                0.999,
                0.900,
                0.999,
            ),
            (
                "RPTOR",
                "FKBP1A",
                "9606.ENSP00000307272",
                "9606.ENSP00000383003",
                0.999,
                0.081,
                0.903,
                0.000,
                0.996,
            ),
            (
                "MTOR",
                "FKBP1A",
                "9606.ENSP00000354558",
                "9606.ENSP00000383003",
                0.999,
                0.049,
                0.993,
                0.000,
                0.997,
            ),
        ],
        columns=[
            "preferredName_A",
            "preferredName_B",
            "stringId_A",
            "stringId_B",
            "score",
            "ascore",
            "escore",
            "dscore",
            "tscore",
        ],
    )
    domain = FeatureDomain(
        species_taxon_id=9606,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.STRING_PREFERRED_SYMBOL,
    )
    seeds = FeatureCollection(
        collection_id="prespecified-mtor-network-seeds",
        domain=domain,
        feature_ids=("MTOR", "FKBP1A", "RPTOR"),
        source_snapshot_id="mechanism-panel-v1",
    )
    raw_response_hash = _archived_csv_sha256(network_frame)
    resource = ResourceSnapshot(
        provider_id="string-db",
        resource_id="functional-interactions",
        resource_release="api-snapshot-2026-08-15",
        retrieved_at=RETRIEVED_AT,
        response_sha256=raw_response_hash,
        source_uri="https://string-db.org/",
    )
    provenance = QueryProvenance(
        provider_id="string-db",
        provider_version="1.0.0",
        resources=(resource,),
        domain=domain,
        retrieved_at=RETRIEVED_AT,
        input_hash=seeds.content_hash,
        query_parameters={
            "tool_type": "network",
            "species": "9606",
            "required_score": 700,
            "network_type": "functional",
            "add_white_nodes": 0,
        },
        warnings=("upstream_database_release_and_license_not_reported_by_provider",),
    )
    network = load_interaction_network_frame(
        network_frame,
        network_id="mtor-string-fixture",
        columns=InteractionNetworkColumns(
            source_feature_id="preferredName_A",
            target_feature_id="preferredName_B",
            source_provider_id="stringId_A",
            target_provider_id="stringId_B",
            confidence="score",
            evidence_channels=(
                EvidenceChannelColumn(
                    channel_id="coexpression",
                    column="ascore",
                    definition="STRING coexpression support",
                ),
                EvidenceChannelColumn(
                    channel_id="experiments",
                    column="escore",
                    definition="STRING experimental support",
                ),
                EvidenceChannelColumn(
                    channel_id="databases",
                    column="dscore",
                    definition="STRING curated-database support",
                ),
                EvidenceChannelColumn(
                    channel_id="textmining",
                    column="tscore",
                    definition="STRING text-mining support",
                ),
            ),
        ),
        domain=domain,
        resource=resource,
        provenance=provenance,
        seed_features=seeds,
        confidence_definition="STRING combined association score on [0, 1]",
        minimum_confidence=0.7,
    )

    print("Interaction context")
    print(f"  nodes: {len(network.nodes)}; edges: {len(network.edges)}")
    print(f"  matched seeds: {', '.join(network.matched_seed_feature_ids)}")
    print(f"  fusion eligibility: {network.fusion_eligibility}")


if __name__ == "__main__":
    functional_context()
    interaction_context()
