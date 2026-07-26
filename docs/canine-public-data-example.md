# Public canine longitudinal-data example

The `public_dog_aging_project.py` example downloads individual pet-dog blood chemistry
measurements from the public
[Dog Aging Project dataset](https://doi.org/10.7910/DVN/8C63KB) hosted by Harvard
Dataverse. The release contains 972 dogs and as many as four `precision_N` sampling
waves. The example uses the first two waves.

```bash
python examples/public_dog_aging_project.py \
  --output-dir analysis/dog-aging-project
```

It downloads and caches the source archive, validates dog and visit identifiers,
creates typed longitudinal subjects and observations, checks expected-visit
missingness, and summarizes paired chemistry changes among dogs measured at both
waves.

## Normalized visit timing

The released chemistry table labels visits as `precision_1`, `precision_2`, and so
on, but does not provide specimen collection dates. The adapter therefore places
the waves 365 days apart on a clearly labeled normalized timeline. These timestamps
support visit-level QC; they are not asserted clinic dates.

## Scientific interpretation

This is real canine longitudinal data, but it is an observational biomarker dataset.
It contains no rapamycin treatment assignment. It is useful for testing ingestion,
expected-visit missingness, paired-measurement handling, and later biological-state
models. It cannot estimate rapamycin efficacy or establish rejuvenation.

The published six-month canine rapamycin trial is relevant context, but its article
does not expose the complete individual-level clinical dataset as a machine-readable
public download. RejuvenationKit should not reconstruct individual dogs from group
means or figures.
