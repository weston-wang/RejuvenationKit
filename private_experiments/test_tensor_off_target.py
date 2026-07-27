"""Tests for private pathway-resolved tensor factorization."""

from __future__ import annotations

import numpy as np
from tensor_off_target import MarkerAnnotation, TensorResponseModel


def test_tensor_model_separates_target_and_off_target_factors() -> None:
    """Recover independent mTOR/autophagy and metabolic safety responses."""
    random = np.random.default_rng(4)
    reference = random.normal(0, 0.25, size=(80, 5, 8))
    treated = random.normal(0, 0.25, size=(80, 5, 8))
    target_course = np.array([0, 0.8, 1.4, 1.7, 1.8])
    safety_course = np.array([0, 0.0, 0.3, 0.9, 1.5])
    treated[:, :, :4] += target_course[None, :, None]
    treated[:, :, 4:] += safety_course[None, :, None]
    annotations = tuple(
        MarkerAnnotation(
            name=f"marker-{index}",
            domain="transcriptome" if index < 4 else "blood",
            pathway="mTOR_autophagy" if index < 4 else "glucose_lipid_safety",
            intended_target=index < 4,
        )
        for index in range(8)
    )
    model = TensorResponseModel(rank=2).fit(reference, treated, annotations)
    factors = model.factors()

    assert {factor.dominant_pathway for factor in factors} == {
        "mTOR_autophagy",
        "glucose_lipid_safety",
    }
    assert any(not factor.off_target for factor in factors)
    assert any(factor.off_target for factor in factors)
    assert len(model.transform(treated)) == 80
