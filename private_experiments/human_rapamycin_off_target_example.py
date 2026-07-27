"""Couple tensor factors and Volterra dynamics in a PEARL-shaped simulation."""

from __future__ import annotations

import numpy as np
from tensor_off_target import MarkerAnnotation, TensorResponseModel
from volterra_epigenetics import EpigeneticVolterraModel, VolterraConfig

MARKERS = (
    MarkerAnnotation("RPS6KB1", "transcriptome", "mTORC1", True),
    MarkerAnnotation("EIF4EBP1", "transcriptome", "mTORC1", True),
    MarkerAnnotation("ULK1", "transcriptome", "autophagy", True),
    MarkerAnnotation("ATG5", "transcriptome", "autophagy", True),
    MarkerAnnotation("NFKB1", "transcriptome", "inflammaging"),
    MarkerAnnotation("CCL2", "transcriptome", "inflammaging"),
    MarkerAnnotation("CXCL10", "transcriptome", "inflammaging"),
    MarkerAnnotation("RICTOR", "transcriptome", "mTORC2"),
    MarkerAnnotation("AKT1", "transcriptome", "mTORC2"),
    MarkerAnnotation("RBC", "bloodwork", "hematology_safety"),
    MarkerAnnotation("BUN", "bloodwork", "renal_safety"),
    MarkerAnnotation("HbA1c", "bloodwork", "glucose_safety"),
    MarkerAnnotation("carbon_dioxide", "bloodwork", "electrolyte_safety"),
    MarkerAnnotation("calcium", "bloodwork", "electrolyte_safety"),
    MarkerAnnotation("LDL", "bloodwork", "lipid_safety"),
    MarkerAnnotation("triglycerides", "bloodwork", "lipid_safety"),
)


def main() -> None:
    """Run a transparent synthetic proof of off-target identifiability."""
    random = np.random.default_rng(2026)
    visits = 5
    placebo = random.normal(0, 0.35, size=(39, visits, len(MARKERS)))
    treated = random.normal(0, 0.35, size=(75, visits, len(MARKERS)))
    dose = np.concatenate((np.full(40, 0.5), np.full(35, 1.0)))
    exposure = np.zeros((75, visits, 1))
    exposure[:, 1:, 0] = dose[:, None]

    target = np.array([0.0, -0.7, -1.1, -1.25, -1.3])
    autophagy = -0.8 * target
    inflammatory = np.array([0.0, -0.15, -0.35, -0.45, -0.5])
    late_safety = np.array([0.0, 0.02, 0.12, 0.35, 0.65])
    for subject in range(len(treated)):
        scale = dose[subject]
        treated[subject, :, 0:2] += scale * target[:, None]
        treated[subject, :, 2:4] += scale * autophagy[:, None]
        treated[subject, :, 4:7] += scale * inflammatory[:, None]
        treated[subject, :, 7:9] += scale**2 * late_safety[:, None]
        treated[subject, :, 9:] += scale**2 * late_safety[:, None]

    tensor = TensorResponseModel(rank=4).fit(placebo, treated, MARKERS)
    factors = tensor.factors(top_markers=4)
    factor_trajectories = tensor.transform(treated)
    volterra = EpigeneticVolterraModel(
        VolterraConfig(memory=2, linear_penalty=0.05, quadratic_penalty=0.10)
    ).fit(
        tuple(exposure),
        factor_trajectories,
    )
    components = volterra.decompose(tuple(exposure))
    nonlinear_energy = np.mean(
        np.vstack([np.square(component.quadratic) for component in components]),
        axis=0,
    )

    print("SYNTHETIC PEARL-SHAPED TENSOR-VOLTERRA DEMONSTRATION")
    print("No participant-level PEARL molecular data are used.")
    for factor in factors:
        print(
            f"Factor {factor.index}: explained={factor.explained_fraction:.1%}; "
            f"pathway={factor.dominant_pathway}; off_target={factor.off_target}; "
            f"target_energy={factor.target_energy_fraction:.1%}; "
            f"nonlinear_energy={nonlinear_energy[factor.index]:.4f}; "
            f"markers={', '.join(factor.top_markers)}"
        )
    print(
        "\nInterpretation: tensor loadings localize coordinated pathways while the quadratic "
        "Volterra contribution identifies dose-dependent nonlinear factors. A factor dominated "
        "by mTORC2 or blood-safety markers is an off-target candidate, not proof of toxicity."
    )


if __name__ == "__main__":
    main()
