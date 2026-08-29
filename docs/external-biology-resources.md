# External biology resources

RejuvenationKit can preserve biological context acquired from external services without making a
live service part of the scientific analysis. The core package is **offline and provider-neutral**:
save a response or export, hash it, describe the exact resource and query, and then pass the saved
table through an explicit importer.

Live plugins are optional acquisition paths. They are not required to install, test, or rerun the
core analysis, and a plugin response is not automatically evidence that an intervention worked.

```text
optional plugin / provider export / local reference file
                         │
                  archive raw response
                         │
        ResourceSnapshot + FeatureDomain + SHA-256
                         │
       FeatureCollection ──────── QueryProvenance
                         │          query parameters
                         │          resource snapshots
                         │          input/query/response hashes
                         ▼
       ┌────────────────────────────────────────────┐
       │ offline typed import and local analysis    │
       │ annotations • ORA • networks • variants   │
       │ sequencing manifests • ortholog maps      │
       └────────────────────────────────────────────┘
                         │
              descriptive context artifacts
                  `not_fusible` by design
                         │
       separately validated subject-level model
              with an effect and uncertainty
                         │
                         ▼
           EvidenceEstimate → Phase 2 fusion
```

## Frozen snapshot and query workflow

The contracts in `rejuvenationkit.genomics.resources` establish the reproducibility boundary:

1. `FeatureDomain` declares the species taxonomy ID, feature type, identifier namespace, and
   optional genome assembly. Domain equality is exact.
2. `ResourceSnapshot` records a provider ID, resource ID, resolved release or acquisition-snapshot
   label, timezone-aware retrieval time, and optional raw archived-response SHA-256, source URI,
   license ID, and citations. The unresolved aliases `latest`, `current`, and `default` are
   rejected.
3. `FeatureCollection` binds a nonempty unique feature set to one domain and source snapshot. Its
   content hash is independent of input feature order.
4. `QueryProvenance` binds the provider/tool version, resource snapshots, domain, input hash,
   canonical query parameters, software versions, retrieval time, normalized typed-result
   checksum, completeness, and warnings. Secret-like parameter keys are rejected recursively, and
   persisted resource URIs reject embedded credentials and secret-like query keys.
5. The core computes the snapshot, feature-collection, and query hashes. Query hashes are sensitive
   to parameters, inputs, domains, software versions, and exact resource snapshots while remaining
   independent of resource tuple order.

The two response hashes have different jobs. `ResourceSnapshot.response_sha256` identifies the
exact archived provider bytes. `QueryProvenance.response_checksum` identifies the deterministic,
normalized typed artifact produced by an importer. Importers compute and validate the latter; it
must not be populated with a DataFrame-record hash or raw-file hash. Keep the raw archive to prove
origin and the normalized checksum to detect changed parsing, filtering, or typed content.

Archive the raw provider response alongside its hash. If a tool does not report the upstream
database release, do not infer one from the plugin version. Use an explicit acquisition-snapshot
identifier and add a warning such as `upstream_database_release_not_reported`. Likewise, populate
`license_id` only when the license was actually supplied or independently verified.

The importers enforce the domain, input-hash, and resource-snapshot constraints relevant to their
artifact types. Modules with explicit truncated/incomplete response semantics keep that state
visible. This makes a cached response an inspectable artifact instead of an undocumented
intermediate table.

## Implemented interpretation layers

| Module | Implemented behavior | Scientific boundary |
|---|---|---|
| `resources` | Frozen feature domains, resource snapshots, feature collections, query provenance, and canonical hashes | Describes acquisition and identity; contains no biological effect estimator |
| `annotations` | Imports functional feature-to-term associations with relation, evidence code, qualifiers, source record IDs, deterministic deduplication, and matched/unmatched query coverage | `FunctionalAnnotationResult.fusion_eligibility` is `not_fusible` |
| `enrichment` | Imports versioned gene/protein sets with a normalized collection checksum and explicit membership policy, then runs local, directionless one-sided hypergeometric overrepresentation analysis against an explicit measured background, with term-size audit and Benjamini–Hochberg, Bonferroni, or no correction | An enrichment p-value is not a treatment-effect estimate or standard error; `OverrepresentationResult` is `not_fusible` |
| `ranked_sets` | Runs a separate directional weighted running-sum analysis over a fixed, prespecified feature ranking, with membership permutations, plus-one p-values, local multiplicity, coverage audits, reconstruction validation, and deterministic hashes | A directional set score is exploratory pathway context, not a calibrated treatment endpoint; `DirectionalRankedSetResult` is `not_fusible` |
| `networks` | Imports saved interaction multigraphs through explicit endpoint, relation type, provider record, provider-ID, confidence, direction, sign, and evidence-channel mappings; exhaustively accounts for retained, thresholded, and dropped-self-loop rows and records matched seeds and truncation | Network confidence and channel scores describe provider records, not subjects; `InteractionNetwork` is `not_fusible` |
| `variants` | Imports saved VEP-shaped, allele-specific consequences and frequencies, then integrity-binds strict joins to dosage matrices by assembly, chromosome, integral position, reference, alternate, reference-span length, matrix feature identity, annotation response, and normalized join checksum | Matching is deliberately exact and does not left-align or minimize equivalent indels; no pathogenicity conclusion or efficacy conversion is made, and annotation batches/joins are `not_fusible` |
| `sequencing` | Binds a saved SRA-shaped study/sample/experiment/run manifest to an exact request; rejects orphan entities; preserves multiple files per run with file role and declared checksum algorithm; and distinguishes absent, incomplete, complete-unverified, and verified sample-to-subject mappings | Discovery metadata is not an analyzed assay; only an explicitly verified complete crosswalk reports an independent-subject count, and the manifest is `not_fusible` |
| `orthologs` | Imports a saved Ensembl/HCOP-shaped table with exact source/target domains, source query, raw snapshot, normalized checksum, optional confidence semantics, and structured translation-loss/collision audits | Translation is `not_fusible`, does not validate the signature in the target species, and does not translate arbitrary networks or annotation results |
| protein adapter | `from_protein_abundance_frame(...)` creates a typed linear `PROTEIN_ABUNDANCE` or normalized `LOG_PROTEIN_ABUNDANCE` matrix using UniProt, explicit STRING-protein, or custom identifiers | Protein abundance becomes evidence only after a separately validated subject-level estimator—such as explicitly configured held-out target calibration—produces an effect and uncertainty |
| `exports` | Wraps already acquired GO, STRING, Ensembl, or SRA bytes in a release-, license-, pagination-, connector-, request-, and SHA-256-bound offline archive | The helper does not contact a provider or infer a missing release/license; raw-byte identity remains separate from normalized importer identity |
| `chunked` | Streams annotation, network, or variant tables into bounded NDJSON or optional Parquet partitions with disk-backed duplicate detection, resumable checkpoints, atomic finalization, and deterministic manifests | A chunked store is an integrity-preserving staging artifact, remains `not_fusible`, and still requires the corresponding typed importer and scientific analysis |

The ORA implementation remains directionless. Directional analysis is intentionally a different
contract in `ranked_sets`; it is a pre-ranked membership-permutation procedure, not GSVA, pathway
topology inference, or causal network propagation. The network implementation imports and audits a
snapshot; it does not infer causal effects from graph structure.

## Provider export archives

`build_go_export(...)`, `build_string_export(...)`, `build_ensembl_export(...)`, and
`build_sra_export(...)` accept bytes already retrieved by a caller. They require a resolved upstream
release, an SPDX expression or explicit unknown-license warning, timezone-aware acquisition time,
connector version, secret-free request parameters, exact raw SHA-256, and a complete pagination
audit. `save_export_archive(...)` writes an atomic ZIP without overwriting an existing target;
`load_export_archive(...)` revalidates both metadata and bytes.

For a paginated acquisition, the `raw_bytes` argument is the byte-for-byte concatenation of page
payloads in receipt order. Every receipt records that page's byte count and SHA-256; construction,
save, and load split the concatenation at those declared boundaries and revalidate every page.
Concatenating parsed rows, reserialized JSON, or any other transformed representation does not
satisfy this raw-page contract. Archives written now use format v2; the loader accepts a v1 archive
only when its bytes also satisfy these exact page-boundary checks.

These helpers are acquisition receipts, not network clients. `ProviderExportEnvelope` converts to
`ResourceSnapshot` and `QueryProvenance`, leaving the normalized response checksum unset for the
typed importer to compute. This preserves the distinction between archived provider bytes and the
later normalized result.

## Bounded-memory catalog staging

`write_chunked_annotations(...)`, `write_chunked_network(...)`, and
`write_chunked_variants(...)` consume DataFrame chunks and write fixed-row partitions without
concatenating the full catalog. Cross-partition keys are checked through a temporary disk-backed
index. A durable checkpoint records only complete partitions; `resume=True` requires the caller to
restart at the recorded source-row offset. Finalization is atomic and will not replace an existing
destination.

NDJSON is available with the core dependencies. Parquet and lazy Parquet reads require:

```bash
python -m pip install "rejuvenationkit[arrow]"
```

Lazy Parquet reads honor pandas schema metadata for object-backed text columns. This keeps a file
written from an `object` string column distinct from one explicitly written with pandas
`StringDtype`, instead of allowing the installed pandas version's default string inference to
change the chunk schema.

Every manifest binds the feature domain, resource snapshot, query provenance, schema, duplicate
policy, partition receipts, row accounting, writer versions, and logical store hash. Partitioned
storage solves scale and recovery; it does not by itself normalize provider semantics or turn a
catalog row into efficacy evidence.

Store roots, resumable staging roots, manifests, partition directories, and partition files must be
real local path entries rather than symbolic links; readers and resume operations fail closed on a
symlink. The parent directory remains a caller-controlled trust boundary, so do not stage a study
artifact in a directory writable by untrusted users.

## Exact fusion boundary

The following are functional context and cannot be fused directly:

- annotation evidence codes or the number of annotated features;
- enrichment p-values, adjusted p-values, overlap counts, or fold enrichment;
- provider interaction confidence or individual evidence-channel scores;
- variant consequences or population frequencies;
- SRA record counts, runs, experiments, or public-study search hits; and
- ortholog confidence or retained mapping weight.

These values do not estimate a common treatment effect and do not provide subject-level standard
errors. Annotation, enrichment, network, variant, and sequencing results therefore expose
`fusion_eligibility="not_fusible"` and do not provide `to_evidence()`. Ortholog translation remains
a signature-definition operation and still requires target-species validation before the translated
signature can support an estimate.

External context can inform a prespecified signature, candidate moderator, safety hypothesis, or
assay plan. To cross the fusion boundary, a separate study analysis must:

1. define the exact estimand, population, time contrast, unit, and direction;
2. measure or predict it for independent subjects in the correct species, tissue, and assay domain;
3. estimate a treatment contrast or use held-out target calibration;
4. quantify subject-level uncertainty and material covariance; and
5. produce an `EvidenceEstimate` with calibration and analysis provenance.

Only estimates with the same complete `Estimand` may then enter covariance-aware or hierarchical
fusion. See [covariance-aware fusion](covariance-aware-fusion.md) and
[genomic signatures](genomic-signatures.md).

## Optional connected acquisition plugins

The connected smarts.bio catalog observed during the integration audit contained:

| Plugin | Catalog version | Possible acquisition role |
|---|---:|---|
| Ensembl | 1.0.0 | Identifier, homology, or genome context to archive before import |
| go-toolkit | 1.0.0 | Functional terms and associations to normalize into annotation or gene-set contracts |
| string-db | 1.0.0 | Interaction records and channel scores to archive and import as a network snapshot |
| uniprot-toolkit | 1.0.0 | Protein records and identifier context for protein-domain workflows |
| ncbi-sra | 1.0.0 | Public sequencing-study discovery and manifest acquisition |
| ebi-search | 1.0.0 | Search across indexed EBI resources before obtaining a saved source artifact |
| biograph | 1.0.0 | Graph-oriented biological context for later normalization into a supported contract |

These are plugin/connector versions, not claims about the release or license of Ensembl, Gene
Ontology, STRING, UniProt, SRA, an EBI index, `biograph`, or any returned dataset. Record a database
release or license only when the provider response or another authoritative artifact supplies it.

## Observations from live-tool probes

The integration audit deliberately tested live acquisition without making it a runtime dependency:

- A STRING response returned individual channel scores in addition to a combined score. The
  network importer therefore retains explicitly named channels—such as coexpression, experimental,
  curated-database, and text-mining support—instead of collapsing them into an unexplained edge.
- One Ensembl homology probe returned HTTP 500.
- One Gene Ontology probe returned HTTP 400.
- One SRA query for canine rapamycin RNA-seq returned zero results.

These are observations about those exact calls, not evidence that the providers are generally
unavailable or that no relevant study exists. Query syntax, indexing, metadata, service state, and
accession hierarchy can all change the result. The core therefore consumes archived tables and
checksums; no scientific rerun depends on a successful live call.

## Canine limitations

The organism lists exposed by the audited GO and STRING catalog tools did not include dog taxonomy
ID `9615`. This does **not** establish that their upstream databases contain no canine records, but
it means direct canine support was not verified through those acquisition paths.

Do not submit dog identifiers under a human or mouse organism setting and do not relabel a returned
human/mouse result as canine. Where a gene-signature workflow permits cross-species translation:

1. archive a versioned ortholog export;
2. declare exact source/target domains and the source feature query;
3. build it with `read_ortholog_map(...)`, passing its raw `ResourceSnapshot`, query provenance,
   and the declared `source_query=FeatureCollection` explicitly;
4. call `translate_signature(...)` with an explicit ambiguity policy;
5. inspect missing, ambiguous, dropped, collided, and translated-weight diagnostics; and
6. rescore and externally validate the translated signature in the intended canine tissue, assay,
   population, and feature namespace.

The current ortholog API translates `GeneSignature` definitions. It does not automatically
translate interaction graphs, functional-annotation results, variant consequences, or provider
confidence values. For those artifacts, use an actually canine-compatible saved resource or retain
the human/mouse result only as clearly labeled hypothesis context. See
[cross-species genomics](cross-species-genomics.md).

## SRA hierarchy and independent subjects

An SRA-like flat table repeats metadata across several distinct entities:

```text
study
  └─ biological sample / BioSample
       └─ experiment / library
            └─ sequencing run
                 ├─ read 1 / technical file + checksum algorithm
                 └─ read 2 or other companion technical file + checksum algorithm
```

A run is not automatically a biological sample, and a BioSample is not automatically an
independent animal. `SequencingManifestRequest` first binds the study, domain, raw resource
snapshot, and selection parameters to the provenance input hash.
`ExternalSequencingStudyManifest` then preserves the hierarchy while
`SubjectMappingDeclaration` records whether the sample-to-subject crosswalk is absent, incomplete,
complete but unverified, or verified—and records its basis and source. Call
`require_subject_mapping()` before an analysis that needs independent subjects. Unless the mapping
is both complete and explicitly verified, `independent_subject_count` is `None`; do not substitute
the run, experiment, or sample count.

The zero-result canine rapamycin query observed in the audit is therefore only a search result. It
is neither a negative biological finding nor permission to synthesize sample assignments.

## Runnable offline example

The example uses archived GO- and STRING-shaped fixtures. It calculates raw fixture hashes;
constructs resource and query provenance; lets the importers bind normalized result checksums;
imports functional annotations; runs directionless ORA and a separate directional ranked-set
fixture; and preserves STRING-style evidence channels:

```bash
python examples/external_biology_context.py
```

The current reference run reports complete annotation coverage for the small fixture, two ORA
terms, two directional ranked-set terms, and a three-node/three-edge interaction snapshot. All
three analysis-result types report `fusion eligibility: not_fusible`. The signed ranking is
synthetic, and every value is a software demonstration rather than measured rejuvenation or safety
evidence.

The typed pandas importers remain the final semantic boundary for selected records. Full catalogs
can now be staged and audited through the bounded-memory chunked store, then streamed partition by
partition into a provider-specific selection/normalization workflow. Parquet support is optional
and lazy; the default NDJSON path does not add a core dependency.

## Reproducibility checklist

Before using an external biological resource in a study artifact, verify that:

- the raw response or export is archived and hashed;
- the retrieval timestamp is timezone-aware;
- the provider/tool version is distinguished from the upstream database release;
- unknown release and license information is reported as unknown rather than inferred;
- the exact taxonomy ID, feature type, namespace, and assembly are declared;
- the query feature collection and its order-independent hash are retained;
- query parameters contain no credentials or tokens;
- resource snapshots and query hashes reproduce exactly;
- raw archived-byte hashes and normalized typed-result hashes are recorded separately;
- truncation, unmatched identifiers, filtering, and incomplete results remain visible;
- identifier or ortholog translation loss is audited; and
- descriptive context is kept outside Phase 2 fusion until independently calibrated subject-level
  evidence exists.
