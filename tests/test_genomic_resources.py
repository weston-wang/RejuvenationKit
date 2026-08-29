from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
    build_feature_collection_hash,
    build_query_hash,
    canonical_sha256,
)
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType


def domain(**updates: object) -> FeatureDomain:
    values: dict[str, object] = {
        "species_taxon_id": 9615,
        "feature_type": GenomicFeatureType.GENE,
        "namespace": FeatureNamespace.ENSEMBL,
        "genome_assembly": "CanFam4",
    }
    values.update(updates)
    return FeatureDomain(**values)


def snapshot(
    *,
    provider_id: str = "reactome",
    resource_id: str = "reactome-pathways",
    resource_release: str = "2026-07",
    response_sha256: str | None = "a" * 64,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id=provider_id,
        resource_id=resource_id,
        resource_release=resource_release,
        retrieved_at=datetime(2026, 8, 14, 12, tzinfo=UTC),
        response_sha256=response_sha256,
        source_uri="https://example.org/archive.tsv",
        license_id="CC0-1.0",
        citation_ids=("PMID:123", "DOI:10.1/example"),
    )


def query(**updates: object) -> QueryProvenance:
    values: dict[str, object] = {
        "provider_id": "gprofiler",
        "provider_version": "e115_eg62_p22",
        "resources": (snapshot(),),
        "domain": domain(),
        "retrieved_at": datetime(2026, 8, 14, 13, tzinfo=UTC),
        "input_hash": "b" * 64,
        "query_parameters": {
            "sources": ("REAC", "GO:BP"),
            "user_threshold": 0.05,
            "ordered": True,
        },
        "software_versions": {"gprofiler-official": "1.0.0"},
    }
    values.update(updates)
    return QueryProvenance(**values)


def test_canonical_sha256_is_mapping_order_independent_and_value_sensitive() -> None:
    first = canonical_sha256({"b": [2, 3], "a": {"value": 1}})
    reordered = canonical_sha256({"a": {"value": 1}, "b": [2, 3]})

    assert first == reordered
    assert first != canonical_sha256({"a": {"value": 2}, "b": [2, 3]})
    with pytest.raises(ValueError, match="finite"):
        canonical_sha256({"invalid": float("nan")})


def test_feature_domain_hash_covers_scientific_compatibility_fields() -> None:
    baseline = domain()

    assert len(baseline.domain_hash) == 64
    assert baseline.domain_hash != domain(species_taxon_id=10090).domain_hash
    assert baseline.domain_hash != domain(genome_assembly="CanFam3.1").domain_hash
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        domain(genome_assembly=" CanFam4")


def test_feature_domain_hash_distinguishes_explicit_identifier_semantics() -> None:
    hgnc_id = domain(namespace=FeatureNamespace.HGNC_ID, genome_assembly=None)
    hgnc_symbol = domain(namespace=FeatureNamespace.HGNC_SYMBOL, genome_assembly=None)
    string_symbol = domain(
        namespace=FeatureNamespace.STRING_PREFERRED_SYMBOL,
        genome_assembly=None,
    )
    string_protein = domain(
        feature_type=GenomicFeatureType.PROTEIN,
        namespace=FeatureNamespace.STRING_PROTEIN,
        genome_assembly=None,
    )

    assert len({hgnc_id.domain_hash, hgnc_symbol.domain_hash}) == 2
    assert hgnc_id.namespace.value == "hgnc_id"
    assert hgnc_symbol.namespace.value == "hgnc_symbol"
    assert string_symbol.namespace.value == "string_preferred_symbol"
    assert string_protein.namespace.value == "string_protein"


@pytest.mark.parametrize(
    ("feature_type", "namespace"),
    [
        (GenomicFeatureType.CPG, FeatureNamespace.UNIPROT),
        (GenomicFeatureType.GENE, FeatureNamespace.STRING_PROTEIN),
        (GenomicFeatureType.PROTEIN, FeatureNamespace.STRING_PREFERRED_SYMBOL),
        (GenomicFeatureType.GENE, FeatureNamespace.RSID),
        (GenomicFeatureType.EMBEDDING_DIMENSION, FeatureNamespace.ENSEMBL),
    ],
)
def test_feature_domain_rejects_incompatible_entity_identifier_pairs(
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
) -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        domain(
            feature_type=feature_type,
            namespace=namespace,
            genome_assembly=None,
        )


def test_feature_domain_allows_named_entities_without_an_assembly() -> None:
    cpg = domain(
        feature_type=GenomicFeatureType.CPG,
        namespace=FeatureNamespace.ILLUMINA_PROBE,
        genome_assembly=None,
    )
    region = domain(
        feature_type=GenomicFeatureType.REGION,
        namespace=FeatureNamespace.CUSTOM,
        genome_assembly=None,
    )
    embedding = domain(
        feature_type=GenomicFeatureType.EMBEDDING_DIMENSION,
        namespace=FeatureNamespace.CUSTOM,
        genome_assembly=None,
    )

    assert cpg.genome_assembly is None
    assert region.genome_assembly is None
    assert embedding.namespace is FeatureNamespace.CUSTOM


def test_feature_domain_requires_assembly_for_coordinate_keyed_namespace() -> None:
    with pytest.raises(ValidationError, match="genome_assembly is required"):
        domain(
            feature_type=GenomicFeatureType.VARIANT,
            namespace=FeatureNamespace.VCF,
            genome_assembly=None,
        )


@pytest.mark.parametrize("release", ["latest", "CURRENT", " default "])
def test_resource_snapshot_rejects_unresolved_releases(release: str) -> None:
    with pytest.raises(ValidationError, match=r"resolved release|surrounding whitespace"):
        snapshot(resource_release=release)


def test_resource_snapshot_rejects_naive_time_bad_hash_and_duplicate_citations() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        ResourceSnapshot(
            provider_id="reactome",
            resource_id="pathways",
            resource_release="2026-07",
            retrieved_at=datetime(2026, 8, 14),
        )
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        snapshot(response_sha256="not-a-sha")
    with pytest.raises(ValidationError, match="unique"):
        snapshot().model_copy(update={"citation_ids": ("PMID:1", "PMID:1")}).model_validate(
            snapshot().model_copy(update={"citation_ids": ("PMID:1", "PMID:1")}).model_dump()
        )


@pytest.mark.parametrize(
    ("source_uri", "message"),
    [
        ("https://alice:password@example.org/archive.tsv", "user credentials"),
        ("https://example.org/archive.tsv?api_token=secret", "secret-like key"),
        ("https://example.org/archive.tsv?filters%5Baccess_key%5D=secret", "secret-like key"),
    ],
)
def test_resource_snapshot_rejects_secrets_in_persisted_uri(
    source_uri: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ResourceSnapshot(
            provider_id="reactome",
            resource_id="pathways",
            resource_release="2026-07",
            retrieved_at=datetime(2026, 8, 14, tzinfo=UTC),
            source_uri=source_uri,
        )

    forged_snapshot = snapshot().model_copy(update={"source_uri": source_uri})
    with pytest.raises(ValidationError, match=message):
        query(resources=(forged_snapshot,))


def test_snapshot_id_is_citation_order_independent_and_resource_sensitive() -> None:
    first = snapshot()
    reordered = first.model_copy(update={"citation_ids": tuple(reversed(first.citation_ids))})

    assert first.snapshot_id == reordered.snapshot_id
    assert first.snapshot_id != snapshot(resource_release="2026-08").snapshot_id
    assert first.snapshot_id != snapshot(response_sha256="c" * 64).snapshot_id


def test_feature_collection_requires_unique_nonempty_ids_and_hashes_as_a_set() -> None:
    source = snapshot()
    first = FeatureCollection(
        collection_id="R-HSA-165159",
        domain=domain(),
        feature_ids=("ENSCAFG2", "ENSCAFG1"),
        source_snapshot_id=source.snapshot_id,
    )
    reordered = first.model_copy(update={"feature_ids": tuple(reversed(first.feature_ids))})

    assert first.content_hash == reordered.content_hash
    assert first.content_hash == build_feature_collection_hash(
        collection_id=first.collection_id,
        domain=first.domain,
        feature_ids=first.feature_ids,
        source_snapshot_id=first.source_snapshot_id,
    )
    assert (
        first.content_hash != first.model_copy(update={"feature_ids": ("ENSCAFG3",)}).content_hash
    )
    assert (
        first.content_hash
        != first.model_copy(update={"domain": domain(species_taxon_id=10090)}).content_hash
    )
    with pytest.raises(ValidationError, match="nonempty"):
        FeatureCollection(
            collection_id="empty",
            domain=domain(),
            feature_ids=(),
            source_snapshot_id=source.snapshot_id,
        )
    with pytest.raises(ValidationError, match="unique"):
        FeatureCollection(
            collection_id="duplicate",
            domain=domain(),
            feature_ids=("g1", "g1"),
            source_snapshot_id=source.snapshot_id,
        )


def test_query_hash_is_core_computed_resource_order_independent_and_parameter_sensitive() -> None:
    first_resource = snapshot()
    second_resource = snapshot(
        provider_id="gene-ontology",
        resource_id="go-basic",
        resource_release="2026-06-17",
        response_sha256="d" * 64,
    )
    first = query(resources=(first_resource, second_resource))
    reordered = query(
        resources=(second_resource, first_resource),
        query_parameters={"ordered": True, "user_threshold": 0.05, "sources": ("REAC", "GO:BP")},
    )

    assert len(first.query_hash) == 64
    assert first.query_hash == reordered.query_hash
    assert first.query_hash == build_query_hash(
        provider_id=first.provider_id,
        provider_version=first.provider_version,
        resources=first.resources,
        domain=first.domain,
        input_hash=first.input_hash,
        query_parameters=first.query_parameters,
        software_versions=first.software_versions,
    )
    assert (
        first.query_hash
        != query(
            resources=(first_resource, second_resource),
            query_parameters={**first.query_parameters, "user_threshold": 0.01},
        ).query_hash
    )
    assert (
        first.query_hash
        != query(resources=(first_resource, second_resource), input_hash="e" * 64).query_hash
    )
    assert (
        first.query_hash
        != query(resources=(snapshot(resource_release="2026-08"), second_resource)).query_hash
    )


def test_query_rejects_unresolved_version_naive_time_bad_hash_and_duplicates() -> None:
    with pytest.raises(ValidationError, match="resolved release"):
        query(provider_version="latest")
    with pytest.raises(ValidationError, match="timezone-aware"):
        query(retrieved_at=datetime(2026, 8, 14))
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        query(input_hash="bad")
    with pytest.raises(ValidationError, match="unique snapshots"):
        query(resources=(snapshot(), snapshot()))
    with pytest.raises(ValidationError, match="unique"):
        query(warnings=("incomplete_mapping", "incomplete_mapping"))


@pytest.mark.parametrize(
    "parameters",
    [
        {"api_key": "do-not-store"},
        {"headers": {"Authorization": "Bearer secret"}},
        {"request": [{"clientSecret": "do-not-store"}]},
        {"access_token": "do-not-store"},
        {"AWS_ACCESS_KEY_ID": "do-not-store"},
        {"secret_name": "do-not-store"},
    ],
)
def test_query_rejects_secret_like_parameter_keys(parameters: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="secret-like"):
        query(query_parameters=parameters)


def test_query_rejects_forged_hash_but_accepts_matching_hash() -> None:
    baseline = query()
    with pytest.raises(ValidationError, match="does not match"):
        query(query_hash="f" * 64)

    reproduced = query(query_hash=baseline.query_hash)
    assert reproduced.query_hash == baseline.query_hash


def test_query_parameters_are_deeply_copied_and_immutable_after_hashing() -> None:
    supplied: dict[str, object] = {
        "filters": {"aspects": ["biological_process"]},
        "limit": 100,
    }
    provenance = query(query_parameters=supplied)
    original_hash = provenance.query_hash

    supplied["limit"] = 5
    nested = supplied["filters"]
    assert isinstance(nested, dict)
    nested["aspects"] = ["molecular_function"]

    assert provenance.query_hash == original_hash
    assert provenance.query_parameters["limit"] == 100
    with pytest.raises(TypeError):
        provenance.query_parameters["limit"] = 5  # type: ignore[index]
    frozen_filters = provenance.query_parameters["filters"]
    assert isinstance(frozen_filters, Mapping)
    with pytest.raises(TypeError):
        frozen_filters["aspects"] = ()  # type: ignore[index]
