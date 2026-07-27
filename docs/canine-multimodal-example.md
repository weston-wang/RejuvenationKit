# Public canine multimodal example

`public_dog_multimodal.py` uses aligned longitudinal measurements from two genuinely different
Dog Aging Project assay families:

- clinical chemistry: globulins and potassium;
- technical-adjusted metabolomics: ethanolamine and 1/3-methylhistidine.

The channels share anonymized dog identifiers and `precision_N` visit waves. The detector learns
their joint longitudinal covariance from a deterministic calibration split, then scores complete
held-out dogs.

Install the optional R-data reader and run:

```bash
python -m pip install -e ".[public-data]"
python examples/public_dog_multimodal.py \
  --output-dir analysis/dog-multimodal
```

To produce the standard DSP diagnostic figures:

```bash
python -m pip install -e ".[public-data,visualization]"
python examples/public_dog_multimodal.py \
  --output-dir analysis/dog-multimodal \
  --plots
```

Covariance whitening makes the detection statistic invariant to ordinary linear rescaling of
channels, allowing reported clinical values and technical-adjusted metabolite abundances to enter
one joint innovation vector. It does not make the modalities biologically interchangeable: the
signed channel-level changes and whitened innovations remain available for interpretation.

This observational dataset contains no intervention assignment. A detected dog has an unusual
joint two-modality change relative to the fitted reference population; it is not evidence of
rejuvenation, deterioration, or rapamycin response.
