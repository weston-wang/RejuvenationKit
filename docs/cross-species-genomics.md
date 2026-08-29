# Cross-species genomic signatures

Human, mouse, and dog feature identifiers are not interchangeable. `OrthologMap` and
`translate_signature(...)` provide a local, versioned translation boundary that reports what was
mapped, lost, duplicated, or selected.

Every map declares:

- exact source and target `FeatureDomain` objects, including NCBI taxonomy IDs and namespaces;
- the exact source `FeatureCollection` that was queried;
- a raw `ResourceSnapshot`, query provenance, and a separate normalized-import checksum;
- mapping resource identifier and resolved version; and
- one or more target features, relationship and evidence IDs for each assertion.

Confidence is optional rather than silently defaulting to perfect support. When present, its
provider-specific definition and scale must be declared. Confidence thresholding and
`SELECT_HIGHEST_CONFIDENCE` fail closed if confidence semantics or any required record value are
missing; an exact highest-confidence tie also fails closed instead of using an arbitrary target.

Translation defaults to records whose relationship is exactly `ortholog`. Other relationship
types require an explicit `allowed_relationships` policy and are recorded in the result.

The translation result reports source mapping retention separately from post-aggregation absolute
weight, so many-to-one cancellation cannot masquerade as full signal retention. It also records
missing features, ambiguous one-to-many features, target collisions, dropped features, allowed
relationship types, every selected source-target pair, actual translated absolute weight, and the
mapping snapshot, query, and content hashes. It rejects translations below
`minimum_retained_weight_fraction`, preserves the source tissue instead of allowing a relabel, and
is explicitly `not_fusible` because an ortholog assertion is not subject-level efficacy evidence.

## One-to-many policies

- `ERROR` is the default and requires a scientific decision for every ambiguity.
- `DROP_AMBIGUOUS` removes one-to-many features and records the loss.
- `SELECT_HIGHEST_CONFIDENCE` chooses a unique highest-confidence target and fails closed on an
  exact tie so an arbitrary identifier cannot determine the biology.
- `DISTRIBUTE_WEIGHT` divides one source weight across its selected targets, preserving the source
  feature's total signed contribution instead of duplicating it.

Mapping a human signature into dog identifiers does not validate it in dogs. The translated
signature receives a new ID/version and incorporates the exact map snapshot, query, and normalized
content hashes into provenance. It must still pass canine tissue, assay, feature-coverage,
calibration, and external-validation checks.

Remote resource clients are intentionally absent from the core path. Research teams should export
and archive a licensed mapping table—such as a versioned Ensembl Compara or HCOP result—then build
an immutable `OrthologMap`. This avoids a live service silently changing a published analysis.
