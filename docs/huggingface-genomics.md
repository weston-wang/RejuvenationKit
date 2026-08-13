# Hugging Face genome-model policy

Hugging Face models are optional upstream feature encoders. Their embeddings are not biological
age, treatment response, pathogenicity, or evidence that rejuvenation occurred.

Install model support separately:

```bash
python -m pip install "rejuvenationkit[genome-hf]"
```

`HFGenomeEmbedder` requires:

- a model ID and full 40-character model/tokenizer commit revision;
- a recorded weights license;
- an explicit Transformers auto-model class compatible with the checkpoint;
- an explicit hidden layer, token pooling method, strand policy, and base-window policy;
- a maximum token count with no silent truncation;
- an explicit device and `trust_remote_code` choice; and
- local-cache-only operation when required by the environment.

Mean pooling excludes padding and special tokens. Long sequences default to an error; explicitly
enabled chunking records window size and overlap, then weights chunk embeddings by newly covered
bases so overlap and short tail chunks are not overrepresented. Forward/reverse-complement
averaging is available and recorded. Provenance also records model class, token limit, batch size, device, remote-code
choice, and cache-only policy, plus package versions and hashes of input sequences—not the raw
sequences themselves.

Model code and weights can carry different security and licensing constraints. Models using
`trust_remote_code=True` execute repository code and should be reviewed and pinned. The following
are unverified candidate checkpoints until a separately pinned smoke test passes with the declared
model class:

- DNABERT-2 is a plausible Apache-2.0 research adapter target but requires remote model code;
- Nucleotide Transformer v2 is multi-species and experimentally useful, but its model-card license
  is CC-BY-NC-SA-4.0 and is not a commercial default;
- HyenaDNA offers small BSD-licensed long-context checkpoints but is human-reference trained and
  also uses custom code; and
- large/gated models are inappropriate for ordinary CI.

RejuvenationKit ordinary tests inject a fake backend and never download weights. Production
deployments should verify model-card terms, pin an immutable revision, cache approved artifacts,
and run a separate opt-in smoke test.

## Calibration boundary

Germline sequence is time-invariant and usually acts as a baseline moderator, stratification
variable, or functional prior. An embedding can enter efficacy fusion only after a held-out model
maps it to the same declared estimand as other evidence and quantifies prediction uncertainty over
independent subjects. Variation across embedding dimensions is not an uncertainty estimate.
