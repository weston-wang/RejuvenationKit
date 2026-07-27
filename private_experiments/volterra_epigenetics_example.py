"""Fit a nonlinear exposure-memory model to synthetic epigenetic markers."""

from __future__ import annotations

import numpy as np
from volterra_epigenetics import EpigeneticVolterraModel, VolterraConfig


def main() -> None:
    """Demonstrate saturation and delayed epigenetic response components."""
    random = np.random.default_rng(42)
    exposures = []
    responses = []
    for _ in range(20):
        dose = random.uniform(0, 1.5, size=60)
        inflammation = random.normal(0, 0.5, size=60)
        inputs = np.column_stack((dose, inflammation))
        methylation_age_delta = np.zeros(60)
        inflammatory_cpg_score = np.zeros(60)
        for time in range(2, 60):
            methylation_age_delta[time] = (
                -1.5 * dose[time]
                - 0.8 * dose[time - 1]
                + 0.9 * dose[time] ** 2
                + 0.6 * dose[time - 1] * inflammation[time]
                + random.normal(0, 0.08)
            )
            inflammatory_cpg_score[time] = (
                1.2 * inflammation[time]
                - 0.5 * dose[time - 2]
                + 0.7 * dose[time] * inflammation[time]
                + random.normal(0, 0.08)
            )
        exposures.append(inputs)
        responses.append(np.column_stack((methylation_age_delta, inflammatory_cpg_score)))

    model = EpigeneticVolterraModel(
        VolterraConfig(memory=3, linear_penalty=1e-3, quadratic_penalty=1e-2)
    ).fit(tuple(exposures[:16]), tuple(responses[:16]))
    predictions = model.predict(tuple(exposures[16:]))
    targets = tuple(response[2:] for response in responses[16:])
    rmse = np.sqrt(
        np.mean(
            np.vstack(
                [
                    np.square(prediction - target)
                    for prediction, target in zip(predictions, targets, strict=True)
                ]
            ),
            axis=0,
        )
    )
    components = model.decompose((exposures[16],))[0]
    assert model.linear_kernel_ is not None
    assert model.quadratic_kernel_ is not None

    print("PRIVATE SYNTHETIC EPIGENETIC VOLTERRA EXPERIMENT")
    print("Markers: methylation_age_delta, inflammatory_cpg_score")
    print(f"Held-out RMSE: {rmse[0]:.4f}, {rmse[1]:.4f}")
    print(f"Linear kernel shape: {model.linear_kernel_.shape}")
    print(f"Quadratic kernel shape: {model.quadratic_kernel_.shape}")
    print(
        "First held-out animal mean absolute contribution — "
        f"linear: {np.abs(components.linear).mean():.4f}; "
        f"quadratic: {np.abs(components.quadratic).mean():.4f}"
    )
    print(
        "Interpretation: the quadratic contribution exposes saturation and exposure-context "
        "interactions that a linear distributed-lag model cannot represent."
    )


if __name__ == "__main__":
    main()
