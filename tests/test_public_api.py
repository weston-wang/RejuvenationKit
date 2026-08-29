"""Regression tests for the documented top-level public API."""

import rejuvenationkit
from rejuvenationkit import genomics


def test_cross_phase_public_api_exports_are_available() -> None:
    """Keep integration and provenance contracts importable from package roots."""
    root_names = (
        "FoldCalibrationProvenance",
        "LongitudinalExclusionCount",
        "RandomizedInferenceProvenance",
        "RejuvenationWorkflowRunner",
        "StateEstimationReport",
        "SubjectEndpointBatch",
    )
    genomic_names = (
        "DirectionalRankedSetResult",
        "GenomicCalibrationDomain",
        "SignatureSampleScore",
    )

    assert all(hasattr(rejuvenationkit, name) for name in root_names)
    assert all(name in rejuvenationkit.__all__ for name in root_names)
    assert all(hasattr(genomics, name) for name in genomic_names)
    assert all(name in genomics.__all__ for name in genomic_names)


def test_every_declared_public_export_resolves_once() -> None:
    """Prevent stale or duplicate names in either package-root export manifest."""
    assert len(rejuvenationkit.__all__) == len(set(rejuvenationkit.__all__))
    assert len(genomics.__all__) == len(set(genomics.__all__))
    assert all(hasattr(rejuvenationkit, name) for name in rejuvenationkit.__all__)
    assert all(hasattr(genomics, name) for name in genomics.__all__)
