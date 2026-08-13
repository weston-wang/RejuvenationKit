from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rejuvenationkit.genomics import (
    FeatureNamespace,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
    annotate_feature_overlaps,
    beta_to_m_values,
    from_anndata,
    from_expression_frame,
    from_methylation_beta,
    normalize_counts_log_cpm,
    read_vcf_dosage,
)


def samples() -> tuple[GenomicSample, ...]:
    return (
        GenomicSample(
            sample_id="s1",
            subject_id="dog-1",
            tissue="blood",
            species_taxon_id=9615,
            cohort="control",
        ),
        GenomicSample(
            sample_id="s2",
            subject_id="dog-2",
            tissue="blood",
            species_taxon_id=9615,
            cohort="treated",
        ),
    )


def provenance() -> GenomicMatrixProvenance:
    return GenomicMatrixProvenance(source_id="adapter-test")


def test_expression_and_methylation_frame_adapters() -> None:
    frame = pd.DataFrame([[1, 2], [3, 4]], index=["s1", "s2"], columns=["g1", "g2"])
    expression = from_expression_frame(
        frame,
        samples=samples(),
        namespace=FeatureNamespace.ENSEMBL,
        scale=MatrixScale.RAW_COUNTS,
        provenance=provenance(),
    )
    assert expression.shape == (2, 2)
    assert expression.features[0].feature_type is GenomicFeatureType.GENE

    beta = from_methylation_beta(
        pd.DataFrame([[0.0, 0.5], [1.0, 0.25]], index=["s1", "s2"], columns=["cg1", "cg2"]),
        samples=samples(),
        provenance=provenance(),
        genome_assembly="CanFam4",
    )
    m_values = beta_to_m_values(beta, epsilon=0.01)
    assert m_values.scale is MatrixScale.METHYLATION_M
    assert np.isfinite(m_values.dense_values()).all()
    assert m_values.dense_values()[0, 0] < 0
    assert m_values.provenance.preprocessing[-1] == "beta_to_m:0.01"
    with pytest.raises(ValueError, match="methylation_beta"):
        beta_to_m_values(expression)
    with pytest.raises(ValueError, match="epsilon"):
        beta_to_m_values(beta, epsilon=0.5)


def test_log_cpm_normalization_uses_full_library_and_limits_output_features() -> None:
    frame = pd.DataFrame(
        [[10, 90, 0], [20, 0, 80]],
        index=["s1", "s2"],
        columns=["g1", "g2", "g3"],
    )
    raw = from_expression_frame(
        frame,
        samples=samples(),
        namespace=FeatureNamespace.ENSEMBL,
        scale=MatrixScale.RAW_COUNTS,
        provenance=provenance(),
    )
    normalized = normalize_counts_log_cpm(raw, feature_ids=("g1",))
    assert normalized.shape == (2, 1)
    assert normalized.scale is MatrixScale.LOG_CPM
    assert normalized.dense_values()[0, 0] == pytest.approx(np.log2(100_000 + 1))
    assert normalized.dense_values()[1, 0] == pytest.approx(np.log2(200_000 + 1))
    assert "library_size_log2_cpm" in normalized.provenance.preprocessing[-1]
    with pytest.raises(ValueError, match="raw_counts"):
        normalize_counts_log_cpm(normalized)
    with pytest.raises(ValueError, match="pseudocount"):
        normalize_counts_log_cpm(raw, pseudocount=0)
    with pytest.raises(ValueError, match="maximum_output_cells"):
        normalize_counts_log_cpm(raw, maximum_output_cells=5)


def test_log_cpm_selects_from_sparse_genome_scale_matrix_before_densifying() -> None:
    feature_count = 20_000
    values = sparse.csr_matrix(
        (
            np.asarray([10.0, 90.0, 20.0, 80.0]),
            (np.asarray([0, 0, 1, 1]), np.asarray([5, 19_999, 5, 10_000])),
        ),
        shape=(2, feature_count),
    )
    raw = GenomicMatrix(
        values=values,
        samples=samples(),
        features=tuple(
            GenomicFeature(
                feature_id=f"g{index}",
                feature_type=GenomicFeatureType.GENE,
                namespace=FeatureNamespace.ENSEMBL,
            )
            for index in range(feature_count)
        ),
        scale=MatrixScale.RAW_COUNTS,
        provenance=provenance(),
    )

    normalized = normalize_counts_log_cpm(
        raw,
        feature_ids=("g5", "g19999"),
        maximum_output_cells=4,
    )

    assert raw.is_sparse
    assert normalized.shape == (2, 2)
    assert normalized.feature_ids == ("g5", "g19999")
    assert normalized.dense_values()[0, 0] == pytest.approx(np.log2(100_000 + 1))
    assert normalized.dense_values()[1, 0] == pytest.approx(np.log2(200_000 + 1))


def test_expression_frame_requires_exact_sample_order_and_unique_features() -> None:
    reversed_frame = pd.DataFrame([[1], [2]], index=["s2", "s1"], columns=["g1"])
    with pytest.raises(ValueError, match="row order"):
        from_expression_frame(
            reversed_frame,
            samples=samples(),
            namespace=FeatureNamespace.ENSEMBL,
            scale=MatrixScale.RAW_COUNTS,
            provenance=provenance(),
        )
    duplicate = pd.DataFrame([[1, 2], [2, 3]], index=["s1", "s2"], columns=["g1", "g1"])
    with pytest.raises(ValueError, match="unique"):
        from_expression_frame(
            duplicate,
            samples=samples(),
            namespace=FeatureNamespace.ENSEMBL,
            scale=MatrixScale.RAW_COUNTS,
            provenance=provenance(),
        )


class FakeAnnData:
    def __init__(self) -> None:
        self.X = sparse.csr_matrix([[1, 0], [0, 2]])
        self.obs_names = ("s1", "s2")
        self.var_names = ("g1", "g2")
        self.obs = pd.DataFrame(
            {
                "subject": ["dog-1", "dog-2"],
                "tissue": ["blood", "blood"],
                "taxon": [9615, 9615],
                "cohort": ["control", "treated"],
                "batch": ["b1", "b2"],
            },
            index=self.obs_names,
        )
        self.var = pd.DataFrame(index=self.var_names)
        self.layers = {"counts": self.X.copy()}


def test_anndata_duck_typed_adapter_preserves_sparse_storage() -> None:
    matrix = from_anndata(
        FakeAnnData(),
        layer="counts",
        scale=MatrixScale.RAW_COUNTS,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        provenance=provenance(),
        tissue_column="tissue",
        species_taxon_id_column="taxon",
        subject_id_column="subject",
        cohort_column="cohort",
        batch_id_column="batch",
    )
    assert matrix.is_sparse
    assert matrix.samples[1].cohort == "treated"
    assert matrix.samples[0].batch_id == "b1"
    with pytest.raises(ValueError, match="layer"):
        from_anndata(
            FakeAnnData(),
            layer="missing",
            scale=MatrixScale.RAW_COUNTS,
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
            provenance=provenance(),
            tissue_column="tissue",
            species_taxon_id_column="taxon",
            subject_id_column="subject",
        )

    misaligned = FakeAnnData()
    misaligned.var.index = pd.Index(["g2", "g1"])
    with pytest.raises(ValueError, match="var index"):
        from_anndata(
            misaligned,
            layer="counts",
            scale=MatrixScale.RAW_COUNTS,
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
            provenance=provenance(),
            tissue_column="tissue",
            species_taxon_id_column="taxon",
            subject_id_column="subject",
        )


def test_vcf_adapter_reads_biallelic_dosage_and_missing_calls(tmp_path: Path) -> None:
    path = tmp_path / "small.vcf"
    path.write_text(
        """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t11\trs1\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/1
1\t20\t.\tC\tT\t.\tPASS\t.\tGT\t./.\t0/0
""",
        encoding="utf-8",
    )
    matrix = read_vcf_dosage(
        path,
        species_taxon_id=9615,
        tissue="blood",
        genome_assembly="CanFam4",
        provenance=provenance(),
        cohort_by_sample={"s1": "control", "s2": "treated"},
    )
    assert matrix.scale is MatrixScale.VARIANT_DOSAGE
    assert matrix.feature_ids == ("rs1", "1:20:C:T")
    assert matrix.features[0].start == 10
    assert matrix.dense_values()[0, 0] == 1
    assert matrix.dense_values()[1, 0] == 2
    assert np.isnan(matrix.dense_values()[0, 1])


def test_vcf_adapter_rejects_multiallelic_records(tmp_path: Path) -> None:
    path = tmp_path / "multi.vcf"
    path.write_text(
        """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
1\t11\t.\tA\tG,T\t.\tPASS\t.\tGT\t0/1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="biallelic"):
        read_vcf_dosage(
            path,
            species_taxon_id=9615,
            tissue="blood",
            genome_assembly="CanFam4",
            provenance=provenance(),
        )


def test_vcf_adapter_rejects_partially_missing_genotype(tmp_path: Path) -> None:
    path = tmp_path / "partial.vcf"
    path.write_text(
        """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
1\t11\t.\tA\tG\t.\tPASS\t.\tGT\t0/.
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="partially missing"):
        read_vcf_dosage(
            path,
            species_taxon_id=9615,
            tissue="blood",
            genome_assembly="CanFam4",
            provenance=provenance(),
        )


def test_bioframe_overlap_adapter_maps_coordinate_features(tmp_path: Path) -> None:
    path = tmp_path / "one.vcf"
    path.write_text(
        """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
1\t11\trs1\tA\tG\t.\tPASS\t.\tGT\t0/1
""",
        encoding="utf-8",
    )
    matrix = read_vcf_dosage(
        path,
        species_taxon_id=9615,
        tissue="blood",
        genome_assembly="CanFam4",
        provenance=provenance(),
    )
    annotations = pd.DataFrame({"chrom": ["1"], "start": [5], "end": [15], "gene_id": ["ENSCAFG1"]})
    overlaps = annotate_feature_overlaps(
        matrix,
        annotations,
        annotation_id_column="gene_id",
        annotation_genome_assembly="CanFam4",
    )
    assert overlaps.loc[0, "feature_id"] == "rs1"
    assert overlaps.loc[0, "gene_id"] == "ENSCAFG1"
    with pytest.raises(ValueError, match="columns"):
        annotate_feature_overlaps(
            matrix,
            annotations.drop(columns="gene_id"),
            annotation_id_column="gene_id",
            annotation_genome_assembly="CanFam4",
        )
    with pytest.raises(ValueError, match="assembly"):
        annotate_feature_overlaps(
            matrix,
            annotations,
            annotation_id_column="gene_id",
            annotation_genome_assembly="CanFam3.1",
        )
