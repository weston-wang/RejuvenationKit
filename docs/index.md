# RejuvenationKit

RejuvenationKit provides typed, composable tools for quality control, analysis-readiness
profiling, DSP change detection, randomized longitudinal inference, multimodal evidence fusion,
latent-state estimation, and factorial combination analysis.

Phases 1 through 4 are implemented as an integrated alpha. The
[one-command audit](study-audit.md) has been validated end to end on
[public longitudinal canine data](dap-audit-case-study.md). Phase 2 adds
[uncertainty-aware multimodal fusion](multimodal-fusion.md),
[covariance-aware evidence fusion](covariance-aware-fusion.md), and a
[genome-scale data and signature layer](genomic-data-model.md). It now also includes an
[offline external-biology resource layer](external-biology-resources.md) for frozen annotations,
gene sets, overrepresentation analysis, interaction networks, variant context, ortholog maps, and
public sequencing-study manifests. See the
[Phase 2 architecture and research use cases](phase-2-architecture-and-use-cases.md) for the full
decision flow. Phase 3 adds [irregular-time latent-state estimation](phase-3-state.md), while
Phase 4 adds [subject-level factorial combination analysis](phase-4-combinations.md) and explicit
bridges between longitudinal measurements, calibrated evidence, state trajectories, and declared
endpoints. The [four-phase workflow](four-phase-workflow.md) runs any configured subset behind one
serialized Phase 1 QC gate and publishes a checksum-verified bundle without inventing conversions
between phases.
