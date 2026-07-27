"""Unit tests for GEO matrix and annotation parsing."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from human_rapamycin_geo import _collapse_to_genes, _read_annotation, _read_series_matrix


def test_geo_parsers_and_gene_collapse(tmp_path: Path) -> None:
    """Parse representative GEO formats and average duplicate gene probes."""
    matrix = tmp_path / "matrix.gz"
    with gzip.open(matrix, "wt", encoding="utf-8") as output:
        output.write('!Sample_title\t"control"\t"rapamycin"\n')
        output.write("!series_matrix_table_begin\n")
        output.write('"ID_REF"\t"GSM1"\t"GSM2"\n')
        output.write('"probe1"\t1\t3\n"probe2"\t3\t5\n')
        output.write("!series_matrix_table_end\n")
    annotation = tmp_path / "annotation.gz"
    with gzip.open(annotation, "wt", encoding="utf-8") as output:
        output.write("^Annotation\n!platform_table_begin\n")
        output.write("ID\tGene symbol\nprobe1\tMTOR\nprobe2\tMTOR\n")

    titles, expression = _read_series_matrix(matrix)
    mapping = _read_annotation(annotation)
    genes = _collapse_to_genes(expression, mapping)

    assert titles.to_dict() == {"GSM1": "control", "GSM2": "rapamycin"}
    np.testing.assert_allclose(genes.loc["MTOR"], [2, 4])
