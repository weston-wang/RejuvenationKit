"""Validation for the private epigenetic Volterra experiment."""

from __future__ import annotations

import numpy as np
import pytest
from volterra_epigenetics import EpigeneticVolterraModel, VolterraConfig


def _synthetic_animals() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    random = np.random.default_rng(8)
    exposures = tuple(random.uniform(0, 1.5, size=(80, 1)) for _ in range(12))
    responses = []
    for exposure in exposures:
        dose = exposure[:, 0]
        marker = np.zeros(len(dose))
        for time in range(1, len(dose)):
            marker[time] = (
                3.0
                + 1.8 * dose[time]
                + 0.7 * dose[time - 1]
                - 1.1 * dose[time] ** 2
                + random.normal(0, 0.05)
            )
        responses.append(marker[:, None])
    return exposures, tuple(responses)


def test_volterra_recovers_nonlinear_marker_response() -> None:
    """Recover known linear memory and quadratic saturation kernels."""
    exposures, responses = _synthetic_animals()
    model = EpigeneticVolterraModel(
        VolterraConfig(memory=2, linear_penalty=1e-5, quadratic_penalty=1e-5)
    ).fit(exposures[:10], responses[:10])

    prediction = model.predict(exposures[10:])[0]
    target = responses[10][1:]
    rmse = float(np.sqrt(np.mean(np.square(prediction - target))))

    assert rmse < 0.08
    assert model.intercept_[0] == pytest.approx(3.0, abs=0.03)
    assert model.linear_kernel_[0, 0, 0] == pytest.approx(1.8, abs=0.06)
    assert model.linear_kernel_[0, 0, 1] == pytest.approx(0.7, abs=0.06)
    assert model.quadratic_kernel_[0, 0, 0, 0, 0] == pytest.approx(-1.1, abs=0.06)


def test_decomposition_sums_to_prediction() -> None:
    """Ensure interpretable components exactly reconstruct predictions."""
    exposures, responses = _synthetic_animals()
    model = EpigeneticVolterraModel(VolterraConfig(memory=2)).fit(exposures, responses)
    component = model.decompose((exposures[0],))[0]
    prediction = model.predict((exposures[0],))[0]

    np.testing.assert_allclose(component.total, prediction)
    assert np.abs(component.quadratic).max() > 0


def test_validation_and_unfitted_errors() -> None:
    """Reject invalid configurations and prediction before fitting."""
    model = EpigeneticVolterraModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict((np.ones((5, 1)),))
    with pytest.raises(ValueError, match="matching"):
        model.fit((), ())
    with pytest.raises(ValueError, match="positive"):
        VolterraConfig(memory=0)
