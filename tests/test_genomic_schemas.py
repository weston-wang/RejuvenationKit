from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import sparse

from rejuvenationkit.genomics import (
    FeatureNamespace,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
)


def sample(sample_id: str) -> GenomicSample:
    return GenomicSample(
        sample_id=sample_id,
        subject_id=f"subject-{sample_id}",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        tissue="blood",
        species_taxon_id=9615,
        cohort="treated",
    )


def feature(feature_id: str) -> GenomicFeature:
    return GenomicFeature(
        feature_id=feature_id,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
    )


def provenance() -> GenomicMatrixProvenance:
    return GenomicMatrixProvenance(
        source_id="unit-test",
        source_checksum="a" * 64,
        preprocessing=("log-cpm",),
        software_versions={"pipeline": "1.0"},
    )


def test_dense_matrix_alignment_hash_subset_and_missingness() -> None:
    matrix = GenomicMatrix(
        values=np.asarray([[1.0, np.nan], [2.0, 3.0]]),
        samples=(sample("s1"), sample("s2")),
        features=(feature("g1"), feature("g2")),
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=provenance(),
    )

    assert matrix.shape == (2, 2)
    assert matrix.sample_ids == ("s1", "s2")
    assert matrix.feature_ids == ("g1", "g2")
    assert matrix.missing_fraction == 0.25
    assert not matrix.is_sparse
    assert len(matrix.content_hash) == 64
    assert len(matrix.artifact_hash) == 64
    assert not matrix.values.flags.writeable
    subset = matrix.subset_features(("g2",))
    assert subset.shape == (2, 1)
    assert subset.feature_ids == ("g2",)
    assert np.isnan(subset.dense_values()[0, 0])
    with pytest.raises(ValueError, match="absent"):
        matrix.subset_features(("unknown",))
    with pytest.raises(ValueError, match="maximum_cells"):
        matrix.dense_values(maximum_cells=3)
    subjects = matrix.subset_samples(("s2",))
    assert subjects.sample_ids == ("s2",)
    assert subjects.shape == (1, 2)
    with pytest.raises(ValueError, match="samples are absent"):
        matrix.subset_samples(("unknown",))


def test_sparse_matrix_remains_sparse_and_hashes_compressed_content() -> None:
    values = sparse.csr_matrix([[0, 1, 0], [2, 0, 3]], dtype=float)
    matrix = GenomicMatrix(
        values=values,
        samples=(sample("s1"), sample("s2")),
        features=(feature("g1"), feature("g2"), feature("g3")),
        scale=MatrixScale.RAW_COUNTS,
        provenance=provenance(),
    )

    assert matrix.is_sparse
    assert matrix.missing_fraction == 0
    assert matrix.values.nnz == 3
    assert not matrix.values.data.flags.writeable
    subset = matrix.subset_features(("g3", "g1"))
    assert subset.is_sparse
    assert np.array_equal(subset.dense_values(), [[0, 0], [3, 2]])
    assert (
        matrix.content_hash
        == GenomicMatrix(
            values=values,
            samples=matrix.samples,
            features=matrix.features,
            scale=matrix.scale,
            provenance=matrix.provenance,
        ).content_hash
    )


def test_dense_and_normalized_sparse_storage_have_the_same_identity() -> None:
    dense_values = np.asarray([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    # The first row is unsorted and contains duplicate index 1 plus explicit zeros.
    sparse_values = sparse.csr_matrix(
        (
            np.asarray([0.0, 0.25, 0.75, -0.0, 3.0, 2.0]),
            np.asarray([2, 1, 1, 0, 2, 0], dtype=np.int32),
            np.asarray([0, 4, 6], dtype=np.int32),
        ),
        shape=(2, 3),
    )
    metadata = {
        "samples": (sample("s1"), sample("s2")),
        "features": (feature("g1"), feature("g2"), feature("g3")),
        "scale": MatrixScale.NORMALIZED_EXPRESSION,
        "provenance": provenance(),
    }

    dense_matrix = GenomicMatrix(values=dense_values, **metadata)
    sparse_matrix = GenomicMatrix(values=sparse_values, **metadata)

    assert sparse_matrix.values.nnz == 3
    assert sparse_matrix.values.has_sorted_indices
    assert np.array_equal(sparse_matrix.dense_values(), dense_values)
    assert sparse_matrix.content_hash == dense_matrix.content_hash
    assert sparse_matrix.artifact_hash == dense_matrix.artifact_hash


def test_hash_schema_breaks_legacy_dense_sparse_binary_collision() -> None:
    sparse_values = sparse.csr_matrix(
        (
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([0, 3], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
        ),
        shape=(1, 4),
    )
    legacy_sparse_payload = (
        sparse_values.data.tobytes()
        + sparse_values.indices.tobytes()
        + sparse_values.indptr.tobytes()
    )
    dense_values = np.frombuffer(legacy_sparse_payload, dtype=np.dtype(float)).reshape(1, 4).copy()
    assert dense_values.tobytes() == legacy_sparse_payload

    metadata = {
        "samples": (sample("s1"),),
        "features": tuple(
            GenomicFeature(
                feature_id=f"embedding-{index}",
                feature_type=GenomicFeatureType.EMBEDDING_DIMENSION,
                namespace=FeatureNamespace.CUSTOM,
            )
            for index in range(4)
        ),
        "scale": MatrixScale.EMBEDDING,
        "provenance": provenance(),
    }
    dense_matrix = GenomicMatrix(values=dense_values, **metadata)
    sparse_matrix = GenomicMatrix(values=sparse_values, **metadata)

    assert not np.array_equal(dense_matrix.dense_values(), sparse_matrix.dense_values())
    assert dense_matrix.content_hash != sparse_matrix.content_hash
    assert dense_matrix.artifact_hash != sparse_matrix.artifact_hash


def test_hashes_are_deterministic_and_sensitive_to_logical_content() -> None:
    values = np.asarray([[0.0, 1.0], [2.0, np.nan]])
    baseline = GenomicMatrix(
        values=values,
        samples=(sample("s1"), sample("s2")),
        features=(feature("g1"), feature("g2")),
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=provenance(),
    )
    repeated = GenomicMatrix(
        values=np.asfortranarray(values),
        samples=baseline.samples,
        features=baseline.features,
        scale=baseline.scale,
        provenance=baseline.provenance,
    )
    changed_values = values.copy()
    changed_values[0, 1] = 1.5
    changed = GenomicMatrix(
        values=changed_values,
        samples=baseline.samples,
        features=baseline.features,
        scale=baseline.scale,
        provenance=baseline.provenance,
    )

    assert repeated.content_hash == baseline.content_hash
    assert repeated.artifact_hash == baseline.artifact_hash
    assert changed.content_hash != baseline.content_hash
    assert changed.artifact_hash != baseline.artifact_hash


def test_length_delimited_identifiers_prevent_separator_collision() -> None:
    values = np.asarray([[1.0], [2.0]])
    first = GenomicMatrix(
        values=values,
        samples=(sample("a\0b"), sample("c")),
        features=(feature("g1"),),
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=provenance(),
    )
    second = GenomicMatrix(
        values=values,
        samples=(sample("a"), sample("b\0c")),
        features=first.features,
        scale=first.scale,
        provenance=first.provenance,
    )

    assert "\0".join(first.sample_ids) == "\0".join(second.sample_ids)
    assert first.content_hash != second.content_hash


def test_genome_scale_sparse_subset_never_materializes_full_matrix() -> None:
    feature_count = 20_000
    values = sparse.csr_matrix(
        (
            np.asarray([10.0, 20.0, 30.0, 40.0]),
            (np.asarray([0, 1, 2, 3]), np.asarray([7, 1_004, 10_000, 19_999])),
        ),
        shape=(4, feature_count),
    )
    matrix = GenomicMatrix(
        values=values,
        samples=tuple(sample(f"s{index}") for index in range(4)),
        features=tuple(feature(f"g{index}") for index in range(feature_count)),
        scale=MatrixScale.RAW_COUNTS,
        provenance=provenance(),
    )

    subset = matrix.subset_features(("g19999", "g7"))

    assert matrix.is_sparse
    assert subset.is_sparse
    assert subset.shape == (4, 2)
    assert subset.values.nnz == 2
    with pytest.raises(ValueError, match="maximum_cells"):
        matrix.dense_values(maximum_cells=10_000)


@pytest.mark.parametrize(
    ("values", "scale", "message"),
    [
        ([[1.2]], MatrixScale.RAW_COUNTS, "integers"),
        ([[-1.0]], MatrixScale.CPM, "nonnegative"),
        ([[1.1]], MatrixScale.METHYLATION_BETA, r"\[0, 1\]"),
        ([[2.5]], MatrixScale.VARIANT_DOSAGE, r"\[0, 2\]"),
        ([[-0.1]], MatrixScale.PROTEIN_ABUNDANCE, "nonnegative"),
        ([[float("inf")]], MatrixScale.EMBEDDING, "infinity"),
    ],
)
def test_matrix_scale_contracts_reject_invalid_values(
    values: list[list[float]],
    scale: MatrixScale,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GenomicMatrix(
            values=np.asarray(values),
            samples=(sample("s1"),),
            features=(feature("g1"),),
            scale=scale,
            provenance=provenance(),
        )


@pytest.mark.parametrize(
    ("feature_type", "namespace", "scale"),
    [
        (GenomicFeatureType.GENE, FeatureNamespace.ENSEMBL, MatrixScale.VARIANT_DOSAGE),
        (GenomicFeatureType.VARIANT, FeatureNamespace.RSID, MatrixScale.NORMALIZED_EXPRESSION),
        (GenomicFeatureType.CPG, FeatureNamespace.ILLUMINA_PROBE, MatrixScale.RAW_COUNTS),
        (GenomicFeatureType.PROTEIN, FeatureNamespace.UNIPROT, MatrixScale.METHYLATION_BETA),
    ],
)
def test_matrix_scale_must_match_the_biological_feature_type(
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
    scale: MatrixScale,
) -> None:
    typed_feature = GenomicFeature(
        feature_id="feature-1",
        feature_type=feature_type,
        namespace=namespace,
    )
    with pytest.raises(ValueError, match="incompatible with feature types"):
        GenomicMatrix(
            values=np.asarray([[1.0]]),
            samples=(sample("s1"),),
            features=(typed_feature,),
            scale=scale,
            provenance=provenance(),
        )


def test_matrix_rejects_shape_and_identifier_mismatches() -> None:
    with pytest.raises(ValueError, match="shape"):
        GenomicMatrix(
            values=np.ones((2, 2)),
            samples=(sample("s1"),),
            features=(feature("g1"), feature("g2")),
            scale=MatrixScale.NORMALIZED_EXPRESSION,
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="sample identifiers"):
        GenomicMatrix(
            values=np.ones((2, 1)),
            samples=(sample("s1"), sample("s1")),
            features=(feature("g1"),),
            scale=MatrixScale.NORMALIZED_EXPRESSION,
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="feature identifiers"):
        GenomicMatrix(
            values=np.ones((1, 2)),
            samples=(sample("s1"),),
            features=(feature("g1"), feature("g1")),
            scale=MatrixScale.NORMALIZED_EXPRESSION,
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="samples and features"):
        GenomicMatrix(
            values=np.empty((0, 0)),
            samples=(),
            features=(),
            scale=MatrixScale.NORMALIZED_EXPRESSION,
            provenance=provenance(),
        )


def test_sample_and_feature_domain_metadata_are_strict() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        GenomicSample(
            sample_id="s1",
            subject_id="subject-1",
            timestamp=datetime(2026, 1, 1),
            tissue="blood",
            species_taxon_id=9615,
        )
    with pytest.raises(ValidationError, match="supplied together"):
        GenomicFeature(
            feature_id="variant",
            feature_type=GenomicFeatureType.VARIANT,
            namespace=FeatureNamespace.VCF,
            chromosome="1",
        )
    with pytest.raises(ValidationError, match="greater"):
        GenomicFeature(
            feature_id="variant",
            feature_type=GenomicFeatureType.VARIANT,
            namespace=FeatureNamespace.VCF,
            genome_assembly="CanFam4",
            chromosome="1",
            start=10,
            end=10,
        )
    with pytest.raises(ValidationError, match="genome_assembly"):
        GenomicFeature(
            feature_id="variant",
            feature_type=GenomicFeatureType.VARIANT,
            namespace=FeatureNamespace.VCF,
            chromosome="1",
            start=10,
            end=11,
        )


def test_artifact_hash_captures_domain_metadata_while_content_hash_remains_compatible() -> None:
    baseline = GenomicMatrix(
        values=np.asarray([[1.0]]),
        samples=(sample("s1"),),
        features=(feature("g1"),),
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=provenance(),
    )
    reassigned = GenomicMatrix(
        values=baseline.values,
        samples=(baseline.samples[0].model_copy(update={"cohort": "control"}),),
        features=baseline.features,
        scale=baseline.scale,
        provenance=baseline.provenance,
    )
    rescaled = GenomicMatrix(
        values=baseline.values,
        samples=baseline.samples,
        features=baseline.features,
        scale=MatrixScale.LOG_CPM,
        provenance=baseline.provenance,
    )

    assert baseline.content_hash == reassigned.content_hash == rescaled.content_hash
    assert len({baseline.artifact_hash, reassigned.artifact_hash, rescaled.artifact_hash}) == 3
    assert GenomicFeatureType.PROTEIN.value == "protein"
    assert FeatureNamespace.UNIPROT.value == "uniprot"


@pytest.mark.parametrize(
    ("feature_type", "namespace", "assembly"),
    [
        (GenomicFeatureType.GENE, FeatureNamespace.HGNC_ID, None),
        (GenomicFeatureType.GENE, FeatureNamespace.HGNC_SYMBOL, None),
        (GenomicFeatureType.GENE, FeatureNamespace.STRING_PREFERRED_SYMBOL, None),
        (GenomicFeatureType.PROTEIN, FeatureNamespace.STRING_PROTEIN, None),
        (GenomicFeatureType.CPG, FeatureNamespace.ILLUMINA_PROBE, None),
        (GenomicFeatureType.REGION, FeatureNamespace.CUSTOM, None),
        (GenomicFeatureType.GENE, FeatureNamespace.CUSTOM, None),
    ],
)
def test_explicit_feature_namespaces_accept_only_their_intended_domains(
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
    assembly: str | None,
) -> None:
    typed = GenomicFeature(
        feature_id="feature-1",
        feature_type=feature_type,
        namespace=namespace,
        genome_assembly=assembly,
    )

    assert typed.namespace is namespace


@pytest.mark.parametrize(
    ("feature_type", "namespace", "assembly"),
    [
        (GenomicFeatureType.CPG, FeatureNamespace.UNIPROT, "CanFam4"),
        (GenomicFeatureType.GENE, FeatureNamespace.STRING_PROTEIN, None),
        (GenomicFeatureType.PROTEIN, FeatureNamespace.STRING_PREFERRED_SYMBOL, None),
        (GenomicFeatureType.GENE, FeatureNamespace.RSID, None),
        (GenomicFeatureType.EMBEDDING_DIMENSION, FeatureNamespace.ENSEMBL, None),
    ],
)
def test_feature_domains_reject_incompatible_entity_identifier_pairs(
    feature_type: GenomicFeatureType,
    namespace: FeatureNamespace,
    assembly: str | None,
) -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        GenomicFeature(
            feature_id="feature-1",
            feature_type=feature_type,
            namespace=namespace,
            genome_assembly=assembly,
        )


def test_coordinate_keyed_namespace_requires_an_assembly() -> None:
    with pytest.raises(ValidationError, match="genome_assembly is required"):
        GenomicFeature(
            feature_id="feature-1",
            feature_type=GenomicFeatureType.VARIANT,
            namespace=FeatureNamespace.VCF,
        )


def test_legacy_namespaces_remain_readable_but_are_explicitly_ambiguous() -> None:
    legacy_hgnc = GenomicFeature(
        feature_id="MTOR",
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.HGNC,
    )
    legacy_string = GenomicFeature(
        feature_id="9606.ENSP00000354558",
        feature_type=GenomicFeatureType.PROTEIN,
        namespace=FeatureNamespace.STRING,
    )

    assert legacy_hgnc.namespace.is_legacy_ambiguous
    assert legacy_string.namespace.is_legacy_ambiguous
    assert not FeatureNamespace.HGNC_ID.is_legacy_ambiguous
    assert not FeatureNamespace.STRING_PROTEIN.is_legacy_ambiguous


def test_hash_relevant_metadata_mappings_are_copied_frozen_and_serializable() -> None:
    sample_attributes = {"breed": "mixed"}
    feature_attributes: dict[str, str | int | float | bool] = {"source": "assay"}
    software_versions = {"pipeline": "1.0"}
    typed_sample = GenomicSample(
        sample_id="s1",
        subject_id="dog-1",
        tissue="blood",
        species_taxon_id=9615,
        attributes=sample_attributes,
    )
    typed_feature = GenomicFeature(
        feature_id="ENSCAFG1",
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        attributes=feature_attributes,
    )
    typed_provenance = GenomicMatrixProvenance(
        source_id="matrix-v1",
        software_versions=software_versions,
    )
    matrix = GenomicMatrix(
        values=np.asarray([[1.0]]),
        samples=(typed_sample,),
        features=(typed_feature,),
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=typed_provenance,
    )
    original_hash = matrix.artifact_hash

    sample_attributes["breed"] = "changed"
    feature_attributes["source"] = "changed"
    software_versions["pipeline"] = "2.0"

    assert typed_sample.attributes == {"breed": "mixed"}
    assert typed_feature.attributes == {"source": "assay"}
    assert typed_provenance.software_versions == {"pipeline": "1.0"}
    with pytest.raises(TypeError):
        typed_sample.attributes["breed"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        typed_feature.attributes["source"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        typed_provenance.software_versions["pipeline"] = "2.0"  # type: ignore[index]
    assert matrix.artifact_hash == original_hash
    assert typed_sample.model_dump(mode="json")["attributes"] == {"breed": "mixed"}
    assert typed_feature.model_dump(mode="json")["attributes"] == {"source": "assay"}
    assert typed_provenance.model_dump(mode="json")["software_versions"] == {"pipeline": "1.0"}
    assert GenomicSample.model_validate(typed_sample.model_dump()) == typed_sample
    assert GenomicFeature.model_validate(typed_feature.model_dump()) == typed_feature
    assert GenomicMatrixProvenance.model_validate(typed_provenance.model_dump()) == typed_provenance


def test_hash_relevant_metadata_rejects_ambiguous_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        GenomicSample(
            sample_id="s1",
            subject_id="dog-1",
            tissue="blood",
            species_taxon_id=9615,
            attributes={"bad": float("nan")},
        )
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        GenomicFeature(
            feature_id="ENSCAFG1",
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
            attributes={" bad": "value"},
        )
    with pytest.raises(ValidationError, match="whitespace"):
        GenomicMatrixProvenance(
            source_id="matrix-v1",
            software_versions={"pipeline": " latest"},
        )
