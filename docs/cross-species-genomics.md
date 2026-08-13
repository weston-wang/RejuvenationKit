# Cross-species genomic signatures

Human, mouse, and dog feature identifiers are not interchangeable. `OrthologMap` and
`translate_signature(...)` provide a local, versioned translation boundary that reports what was
mapped, lost, duplicated, or selected.

Every map declares:

- source and target NCBI taxonomy IDs;
- source and target feature namespaces;
- mapping resource identifier and version; and
- one or more target features, confidence, relationship, and evidence IDs for each assertion.

Translation defaults to records whose relationship is exactly `ortholog`. Other relationship
types require an explicit `allowed_relationships` policy and are recorded in the result.

The translation result reports source mapping retention separately from post-aggregation absolute
weight, so many-to-one cancellation cannot masquerade as full signal retention. It also records
missing features, ambiguous one-to-many features, target collisions, dropped features, allowed
relationship types, and every selected source-target pair. It rejects translations below
`minimum_retained_weight_fraction`.

## One-to-many policies

- `ERROR` is the default and requires a scientific decision for every ambiguity.
- `DROP_AMBIGUOUS` removes one-to-many features and records the loss.
- `SELECT_HIGHEST_CONFIDENCE` chooses the highest-confidence target with stable identifier
  tie-breaking.
- `DISTRIBUTE_WEIGHT` divides one source weight across its selected targets, preserving the source
  feature's total signed contribution instead of duplicating it.

Mapping a human signature into dog identifiers does not validate it in dogs. The translated
signature receives a new ID/version and incorporates the map resource into provenance. It must
still pass canine tissue, assay, feature-coverage, calibration, and external-validation checks.

Remote resource clients are intentionally absent from the core path. Research teams should export
and archive a licensed mapping table—such as a versioned Ensembl Compara or HCOP result—then build
an immutable `OrthologMap`. This avoids a live service silently changing a published analysis.
