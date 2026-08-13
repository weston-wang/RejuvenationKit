# Genome-scale data model

Phase 2 accepts genome-scale assay outputs without turning millions of measurements into
`Observation` objects. The `GenomicMatrix` contract keeps a samples-by-features dense or CSR sparse
matrix aligned to typed sample and feature metadata.

```text
upstream assay pipeline
          │
          ├─ counts / normalized expression
          ├─ methylation beta or M values
          ├─ VCF/BCF dosage
          └─ precomputed sequence embeddings
          │
     GenomicMatrix
          │
          ├─ signatures / pathways ── uncertainty ─┐
          ├─ held-out target calibrator ───────────┼─ EvidenceEstimate
          └─ baseline genomic moderators          │  (only after calibration)
                                                   └─ covariance-aware fusion
```

## Matrix contract

Every matrix declares:

- ordered sample and feature identifiers;
- species taxonomy ID, tissue, assay, cohort, subject, and batch metadata;
- feature namespace, type, and optional genome assembly and coordinates;
- numerical scale such as raw counts, normalized expression, methylation beta, M-value, variant
  dosage, or embedding;
- preprocessing, source identifier, optional external-file checksum, software versions, and
  reference resources; and
- a content hash computed over aligned identifiers and numerical storage.

Dense matrices may encode missing measurements as `NaN`; infinity is rejected. Sparse matrices
interpret absent entries as measured zero and therefore cannot silently use sparse absence for
missingness. The API preserves CSR storage during feature subsetting and guards dense
materialization with `maximum_cells`.

Scale-specific validation rejects fractional or negative raw counts, negative CPM/TPM values,
methylation beta values outside `[0, 1]`, and diploid dosages outside `[0, 2]`.

## Adapters and libraries

The base installation uses SciPy CSR matrices. Optional extras add established file and annotation
tools:

```bash
python -m pip install "rejuvenationkit[genomics,hts]"
```

- `from_anndata(...)` uses an AnnData-like structural interface and does not require AnnData at
  import time. Callers explicitly select `X` or a named layer and map required `obs` columns.
- `read_vcf_dosage(...)` uses `pysam.VariantFile` from the `hts` extra. The baseline accepts
  biallelic diploid records, records assembly and coordinates, maps missing genotypes to `NaN`, and
  rejects multiallelic or non-diploid simplification.
- `from_methylation_beta(...)` validates normalized beta matrices. `beta_to_m_values(...)` records
  the clipping epsilon in preprocessing provenance. Raw IDAT/bisulfite processing remains the job
  of assay-specific pipelines.
- `annotate_feature_overlaps(...)` uses `bioframe` from the `genomics` extra for pandas-native
  interval overlaps and requires an annotation assembly that matches the matrix.
- `normalize_counts_log_cpm(...)` calculates library size from every raw-count feature but can emit
  only requested signature features, avoiding dense expansion of a complete sparse transcriptome.

RejuvenationKit does not replace base calling, read alignment, variant normalization, DESeq2,
edgeR, limma, or array preprocessing. It validates their curated outputs and constructs auditable
evidence for cross-assay decisions.

## Domain compatibility

Species, tissue, feature namespace, matrix scale, and genome assembly are scientific inputs rather
than cosmetic labels. A human blood calibration is not applied to canine liver by default. Feature
signatures require an exact namespace and species match; cross-species use needs a separately
versioned ortholog map and validation artifact.

The canonical taxonomy IDs used in examples are `9606` for human, `10090` for mouse, and `9615` for
dog. Multi-species sequence-model pretraining is not evidence that a model is calibrated in dogs.
