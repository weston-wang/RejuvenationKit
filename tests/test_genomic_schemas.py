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


def test_matrix_rejects_shape_and_identifier_mismatches() -> None:
    with pytest.raises(ValueError, match="shape"):
        GenomicMatrix(
            values=np.ones((2, 2)),
            samples=(sample("s1"),),
            features=(feature("g1"), feature("g2")),
            scale=MatrixScale.EMBEDDING,
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="sample identifiers"):
        GenomicMatrix(
            values=np.ones((2, 1)),
            samples=(sample("s1"), sample("s1")),
            features=(feature("g1"),),
            scale=MatrixScale.EMBEDDING,
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="feature identifiers"):
        GenomicMatrix(
            values=np.ones((1, 2)),
            samples=(sample("s1"),),
            features=(feature("g1"), feature("g1")),
            scale=MatrixScale.EMBEDDING,
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="samples and features"):
        GenomicMatrix(
            values=np.empty((0, 0)),
            samples=(),
            features=(),
            scale=MatrixScale.EMBEDDING,
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
