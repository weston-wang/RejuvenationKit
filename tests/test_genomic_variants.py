from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.resources import FeatureDomain, QueryProvenance, ResourceSnapshot
from rejuvenationkit.genomics.schemas import (
    FeatureNamespace,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
)
from rejuvenationkit.genomics.variants import (
    VariantAnnotationBatch,
    VariantAnnotationJoin,
    VariantAnnotationRequest,
    VariantKey,
    join_variant_annotations,
    read_variant_annotations,
)


def snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id="ensembl",
        resource_id="variation-consequence",
        resource_release="2026-07",
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        response_sha256="b" * 64,
        license_id="Apache-2.0",
    )


def requested_variants(*, assembly: str = "CanFam4") -> tuple[VariantKey, ...]:
    return (
        VariantKey(
            genome_assembly=assembly,
            chromosome="1",
            position_1_based=11,
            reference_allele="A",
            alternate_allele="G",
        ),
        VariantKey(
            genome_assembly=assembly,
            chromosome="2",
            position_1_based=30,
            reference_allele="C",
            alternate_allele="T",
        ),
        VariantKey(
            genome_assembly=assembly,
            chromosome="4",
            position_1_based=50,
            reference_allele="A",
            alternate_allele="C",
        ),
    )


def provenance(
    *,
    assembly: str = "CanFam4",
    variants: tuple[VariantKey, ...] | None = None,
) -> QueryProvenance:
    resource = snapshot()
    request = VariantAnnotationRequest(
        species_taxon_id=9615,
        variants=variants or requested_variants(assembly=assembly),
        resource=resource,
    )
    return QueryProvenance(
        provider_id="smarts.bio/ensembl",
        provider_version="1.0.0",
        resources=(resource,),
        domain=FeatureDomain(
            species_taxon_id=9615,
            feature_type=GenomicFeatureType.VARIANT,
            namespace=FeatureNamespace.VCF,
            genome_assembly=assembly,
        ),
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        input_hash=request.input_hash,
        query_parameters={"operation": "variation"},
    )


def annotations() -> VariantAnnotationBatch:
    frame = pd.DataFrame(
        {
            "chrom": ["1", "1", "2"],
            "pos": [11, 11, 30],
            "ref": ["A", "A", "C"],
            "alt": ["G", "G", "T"],
            "consequence": ["missense_variant", "splice_region_variant", "intron_variant"],
            "gene": ["ENSCAFG1", "ENSCAFG1", "ENSCAFG2"],
            "transcript": ["ENSCAFT1", "ENSCAFT2", "ENSCAFT3"],
            "canonical": [True, False, True],
            "dog_af": [0.04, 0.04, 0.2],
        }
    )
    return read_variant_annotations(
        frame,
        chromosome_column="chrom",
        position_column="pos",
        reference_column="ref",
        alternate_column="alt",
        consequence_column="consequence",
        genome_assembly="CanFam4",
        species_taxon_id=9615,
        requested_variants=requested_variants(),
        gene_id_column="gene",
        gene_namespace=FeatureNamespace.ENSEMBL,
        transcript_id_column="transcript",
        canonical_transcript_column="canonical",
        population_frequency_columns={"dog-reference": "dog_af"},
        resource=snapshot(),
        provenance=provenance(),
    )


def dosage_matrix(*, assembly: str = "CanFam4") -> GenomicMatrix:
    return GenomicMatrix(
        values=np.asarray([[1.0, 2.0]]),
        samples=(
            GenomicSample(
                sample_id="dog-1",
                subject_id="dog-1",
                tissue="blood",
                species_taxon_id=9615,
            ),
        ),
        features=(
            GenomicFeature(
                feature_id="rs-test",
                feature_type=GenomicFeatureType.VARIANT,
                namespace=FeatureNamespace.VCF,
                genome_assembly=assembly,
                chromosome="1",
                start=10,
                end=11,
                attributes={"reference": "A", "alternate": "G"},
            ),
            GenomicFeature(
                feature_id="not-annotated",
                feature_type=GenomicFeatureType.VARIANT,
                namespace=FeatureNamespace.VCF,
                genome_assembly=assembly,
                chromosome="3",
                start=49,
                end=50,
                attributes={"reference": "G", "alternate": "A"},
            ),
        ),
        scale=MatrixScale.VARIANT_DOSAGE,
        provenance=GenomicMatrixProvenance(source_id="dog-vcf"),
    )


def test_variant_import_groups_transcripts_and_preserves_external_provenance() -> None:
    batch = annotations()

    assert len(batch.records) == 2
    assert len(batch.records[0].consequences) == 2
    assert batch.records[0].population_frequencies == {"dog-reference": 0.04}
    assert batch.resource.resource_release == "2026-07"
    assert batch.unmatched_variant_ids == ("CanFam4:4:50:A>C",)
    assert "variant_equivalence_normalization_not_performed" in batch.warnings
    assert batch.provenance.response_checksum is not None
    assert len(batch.provenance.response_checksum) == 64
    assert batch.fusion_eligibility == "not_fusible"
    assert not hasattr(batch, "to_evidence")


def test_variant_join_is_allele_and_assembly_strict_and_reports_unmatched_records() -> None:
    batch = annotations()
    joined = join_variant_annotations(dosage_matrix(), batch)

    assert [item.matrix_feature_id for item in joined.matches] == ["rs-test"]
    assert joined.unmatched_matrix_feature_ids == ("not-annotated",)
    assert joined.unmatched_annotation_variant_ids == ("CanFam4:2:30:C>T",)
    assert joined.matrix_feature_ids == ("not-annotated", "rs-test")
    assert joined.annotation_variant_ids == ("CanFam4:1:11:A>G", "CanFam4:2:30:C>T")
    assert joined.annotation_response_checksum == batch.provenance.response_checksum
    assert len(joined.join_checksum) == 64
    assert "matrix_variants_without_annotations" in joined.warnings
    assert joined.fusion_eligibility == "not_fusible"

    with pytest.raises(ValueError, match="assemblies"):
        join_variant_annotations(dosage_matrix(assembly="CanFam3.1"), batch)


def test_variant_keys_reject_unsplit_or_identity_alleles() -> None:
    normalized = VariantKey(
        genome_assembly="CanFam4",
        chromosome="1",
        position_1_based=1,
        reference_allele=" a ",
        alternate_allele="g",
    )
    assert normalized.canonical_id == "CanFam4:1:1:A>G"
    with pytest.raises(ValidationError, match="split alternate"):
        VariantKey(
            genome_assembly="CanFam4",
            chromosome="1",
            position_1_based=1,
            reference_allele="A",
            alternate_allele="G,T",
        )
    with pytest.raises(ValidationError, match="must differ"):
        VariantKey(
            genome_assembly="CanFam4",
            chromosome="1",
            position_1_based=1,
            reference_allele="A",
            alternate_allele="A",
        )


@pytest.mark.parametrize("position", [1.9, np.nan, True, "1.9"])
def test_variant_positions_are_never_lossily_coerced(position: object) -> None:
    """Fractional, missing, and boolean coordinates cannot silently match a request."""
    with pytest.raises(ValidationError, match="position"):
        VariantKey(
            genome_assembly="CanFam4",
            chromosome="1",
            position_1_based=cast(Any, position),
            reference_allele="A",
            alternate_allele="G",
        )


def test_variant_import_rejects_conflicting_population_frequency() -> None:
    frame = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [11, 11],
            "ref": ["A", "A"],
            "alt": ["G", "G"],
            "consequence": ["missense_variant", "splice_region_variant"],
            "af": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="conflicting"):
        requested = (requested_variants()[0],)
        read_variant_annotations(
            frame,
            chromosome_column="chrom",
            position_column="pos",
            reference_column="ref",
            alternate_column="alt",
            consequence_column="consequence",
            genome_assembly="CanFam4",
            species_taxon_id=9615,
            requested_variants=requested,
            gene_namespace=None,
            population_frequency_columns={"dog": "af"},
            resource=snapshot(),
            provenance=provenance(variants=requested),
        )


def test_variant_import_represents_a_valid_empty_provider_response() -> None:
    requested = (requested_variants()[0],)
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt", "consequence"])

    batch = read_variant_annotations(
        empty,
        chromosome_column="chrom",
        position_column="pos",
        reference_column="ref",
        alternate_column="alt",
        consequence_column="consequence",
        genome_assembly="CanFam4",
        species_taxon_id=9615,
        requested_variants=requested,
        gene_namespace=None,
        resource=snapshot(),
        provenance=provenance(variants=requested),
    )

    assert batch.records == ()
    assert batch.unmatched_variant_ids == ("CanFam4:1:11:A>G",)
    assert "requested_variants_without_annotations" in batch.warnings


def test_variant_import_rejects_provenance_for_a_different_allele_query() -> None:
    requested = (requested_variants()[0],)
    frame = pd.DataFrame(
        {
            "chrom": ["1"],
            "pos": [11],
            "ref": ["A"],
            "alt": ["G"],
            "consequence": ["missense_variant"],
        }
    )

    with pytest.raises(ValidationError, match="exact allele request"):
        read_variant_annotations(
            frame,
            chromosome_column="chrom",
            position_column="pos",
            reference_column="ref",
            alternate_column="alt",
            consequence_column="consequence",
            genome_assembly="CanFam4",
            species_taxon_id=9615,
            requested_variants=requested,
            gene_namespace=None,
            resource=snapshot(),
            provenance=provenance(),
        )


def test_variant_import_rejects_missing_required_scalar_fields() -> None:
    """Missing contigs, alleles, and consequences are not stringified as 'nan'."""
    requested = (requested_variants()[0],)
    base = pd.DataFrame(
        {
            "chrom": ["1"],
            "pos": [11],
            "ref": ["A"],
            "alt": ["G"],
            "consequence": ["missense_variant"],
        }
    )
    for column in ("chrom", "ref", "alt", "consequence"):
        broken = base.copy()
        broken.loc[0, column] = np.nan
        with pytest.raises((ValueError, ValidationError), match="cannot be missing"):
            read_variant_annotations(
                broken,
                chromosome_column="chrom",
                position_column="pos",
                reference_column="ref",
                alternate_column="alt",
                consequence_column="consequence",
                genome_assembly="CanFam4",
                species_taxon_id=9615,
                requested_variants=requested,
                gene_namespace=None,
                resource=snapshot(),
                provenance=provenance(variants=requested),
            )


def test_variant_join_requires_interval_to_match_reference_allele_length() -> None:
    """A one-base REF cannot annotate a feature spanning a different interval."""
    source = dosage_matrix()
    malformed_features = list(source.features)
    malformed_features[0] = malformed_features[0].model_copy(update={"end": 99})
    malformed = replace(source, features=tuple(malformed_features))
    with pytest.raises(ValueError, match="interval length"):
        join_variant_annotations(malformed, annotations())


def test_variant_normalized_response_is_row_order_invariant_and_bound() -> None:
    """Equivalent provider order yields one checksum; a stale checksum is rejected."""
    first = annotations()
    data = pd.DataFrame(
        {
            "chrom": ["2", "1", "1"],
            "pos": [30, 11, 11],
            "ref": ["C", "A", "A"],
            "alt": ["T", "G", "G"],
            "consequence": ["intron_variant", "splice_region_variant", "missense_variant"],
            "gene": ["ENSCAFG2", "ENSCAFG1", "ENSCAFG1"],
            "transcript": ["ENSCAFT3", "ENSCAFT2", "ENSCAFT1"],
            "canonical": [True, False, True],
            "dog_af": [0.2, 0.04, 0.04],
        }
    )
    reordered = read_variant_annotations(
        data,
        chromosome_column="chrom",
        position_column="pos",
        reference_column="ref",
        alternate_column="alt",
        consequence_column="consequence",
        genome_assembly="CanFam4",
        species_taxon_id=9615,
        requested_variants=requested_variants(),
        gene_id_column="gene",
        gene_namespace=FeatureNamespace.ENSEMBL,
        transcript_id_column="transcript",
        canonical_transcript_column="canonical",
        population_frequency_columns={"dog-reference": "dog_af"},
        resource=snapshot(),
        provenance=provenance(),
    )
    assert reordered.records == first.records
    assert reordered.provenance.response_checksum == first.provenance.response_checksum
    assert join_variant_annotations(dosage_matrix(), reordered).model_dump(
        mode="json"
    ) == join_variant_annotations(dosage_matrix(), first).model_dump(mode="json")

    stale = provenance().model_copy(update={"response_checksum": "f" * 64})
    with pytest.raises(ValueError, match="supplied response checksum"):
        read_variant_annotations(
            data,
            chromosome_column="chrom",
            position_column="pos",
            reference_column="ref",
            alternate_column="alt",
            consequence_column="consequence",
            genome_assembly="CanFam4",
            species_taxon_id=9615,
            requested_variants=requested_variants(),
            gene_id_column="gene",
            gene_namespace=FeatureNamespace.ENSEMBL,
            transcript_id_column="transcript",
            canonical_transcript_column="canonical",
            population_frequency_columns={"dog-reference": "dog_af"},
            resource=snapshot(),
            provenance=stale,
        )


def test_variant_join_round_trip_reconstructs_partitions_and_checksum() -> None:
    """Serialized joins validate both source identities and every audit partition."""
    joined = join_variant_annotations(dosage_matrix(), annotations())
    reproduced = VariantAnnotationJoin.model_validate(joined.model_dump(mode="python"))
    assert reproduced == joined

    missing_partition_member = joined.model_dump(mode="python")
    missing_partition_member["unmatched_matrix_feature_ids"] = ()
    with pytest.raises(ValidationError, match="partition matrix_feature_ids"):
        VariantAnnotationJoin.model_validate(missing_partition_member)

    forged_response = joined.model_dump(mode="python")
    forged_response["annotation_response_checksum"] = "f" * 64
    with pytest.raises(ValidationError, match="join checksum"):
        VariantAnnotationJoin.model_validate(forged_response)

    forged_match = joined.model_dump(mode="python")
    forged_match["matches"][0]["annotation"]["consequences"][0]["impact_label"] = "HIGH"
    with pytest.raises(ValidationError, match="join checksum"):
        VariantAnnotationJoin.model_validate(forged_match)
