from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from rejuvenationkit.longitudinal import (
    AggregationPolicy,
    LongitudinalAlignmentError,
    LongitudinalChannel,
    LongitudinalExclusionReason,
    LongitudinalExtraction,
    VisitAlignedVector,
    VisitVectorExtraction,
    complete_visit_vectors,
    extract_visit_aligned_values,
)
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject

START = datetime(2026, 1, 1, tzinfo=UTC)
VISIT = ExpectedVisit(
    visit_id="week-1",
    scheduled_at=START + timedelta(days=7),
    window_before=timedelta(days=2),
    window_after=timedelta(days=2),
    required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
)


def observation(
    *,
    day: int,
    value: float,
    modality: Modality = Modality.CLINICAL,
    unit: str = "g/dL",
    replicate_id: str | None = None,
) -> Observation:
    return Observation(
        subject_id="dog-1",
        timestamp=START + timedelta(days=day),
        modality=modality,
        feature="albumin",
        value=value,
        unit=unit,
        replicate_id=replicate_id,
    )


def study(*rows: Observation, anchored: bool = True) -> Study:
    return Study(
        study_id="alignment",
        subjects=(
            Subject(
                subject_id="dog-1",
                cohort="treated",
                anchors={"dose": START} if anchored else {},
            ),
        ),
        observations=rows,
    )


def test_mean_aggregation_preserves_exact_channel_indices_and_observed_time() -> None:
    source = study(
        observation(day=6, value=2.0, replicate_id="a"),
        observation(day=8, value=4.0, replicate_id="b"),
    )

    extraction = extract_visit_aligned_values(
        source,
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin"),),
    )

    assert extraction.channels == (
        LongitudinalChannel(
            feature="albumin",
            modality=Modality.CLINICAL,
            unit="g/dL",
            aggregation_policy=AggregationPolicy.MEAN,
        ),
    )
    aligned = extraction.values[0]
    assert aligned.value == 3.0
    assert aligned.selected_observation_indices == (0, 1)
    assert aligned.effective_timestamp == START + timedelta(days=7)
    assert aligned.scheduled_timestamp == START + timedelta(days=7)

    vectors = complete_visit_vectors(extraction)
    assert vectors.vectors[0].values == (3.0,)
    assert vectors.vectors[0].selected_observation_indices == (0, 1)


def test_closest_to_schedule_averages_only_replicates_at_selected_time() -> None:
    source = study(
        observation(day=6, value=100.0),
        observation(day=8, value=2.0, replicate_id="a"),
        observation(day=8, value=4.0, replicate_id="b"),
    )
    channel = LongitudinalChannel(
        feature="albumin",
        modality=Modality.CLINICAL,
        unit="g/dL",
        aggregation_policy=AggregationPolicy.CLOSEST_TO_SCHEDULE,
    )

    aligned = extract_visit_aligned_values(
        source,
        visits=(VISIT,),
        channels=(channel,),
    ).values[0]

    # Days 6 and 8 are tied; the documented deterministic tie break chooses
    # the earlier timestamp and therefore does not let replicate count decide.
    assert aligned.value == 100.0
    assert aligned.selected_observation_indices == (0,)
    assert aligned.effective_timestamp == START + timedelta(days=6)


def test_wildcard_modality_ambiguity_fails_with_structured_provenance() -> None:
    source = study(
        observation(day=7, value=3.0),
        observation(day=7, value=3.0, modality=Modality.PROTEOMICS),
    )

    with pytest.raises(LongitudinalAlignmentError) as caught:
        extract_visit_aligned_values(
            source,
            visits=(VISIT,),
            channels=(VisitFeature(feature="albumin"),),
        )

    exclusion = caught.value.exclusions[0]
    assert exclusion.reason is LongitudinalExclusionReason.AMBIGUOUS_MODALITY
    assert exclusion.observation_indices == (0, 1)
    assert exclusion.observed_modalities == (Modality.CLINICAL, Modality.PROTEOMICS)


def test_wildcard_and_exact_requests_cannot_alias_one_resolved_channel() -> None:
    source = study(observation(day=7, value=3.0))

    with pytest.raises(ValueError, match="cannot overlap"):
        extract_visit_aligned_values(
            source,
            visits=(VISIT,),
            channels=(
                VisitFeature(feature="albumin"),
                VisitFeature(feature="albumin", modality=Modality.CLINICAL),
            ),
        )


def test_distinct_exact_modalities_for_one_feature_remain_valid() -> None:
    source = study(
        observation(day=7, value=3.0),
        observation(day=7, value=4.0, modality=Modality.PROTEOMICS),
    )

    extraction = extract_visit_aligned_values(
        source,
        visits=(VISIT,),
        channels=(
            VisitFeature(feature="albumin", modality=Modality.CLINICAL),
            VisitFeature(feature="albumin", modality=Modality.PROTEOMICS),
        ),
    )

    assert tuple(channel.modality for channel in extraction.channels if channel is not None) == (
        Modality.CLINICAL,
        Modality.PROTEOMICS,
    )


def test_serialized_extractions_and_vectors_reject_duplicate_exact_channels() -> None:
    channel = LongitudinalChannel(
        feature="albumin",
        modality=Modality.CLINICAL,
        unit="g/dL",
    )

    with pytest.raises(ValidationError, match="resolved longitudinal channels must be unique"):
        LongitudinalExtraction(
            study_id="alignment",
            visit_ids=(VISIT.visit_id,),
            subject_ids=("dog-1",),
            channels=(channel, channel),
        )

    with pytest.raises(ValidationError, match="visit vector channels must be unique"):
        VisitAlignedVector(
            subject_id="dog-1",
            visit_id=VISIT.visit_id,
            channels=(channel, channel),
            values=(3.0, 3.0),
            effective_timestamp=START + timedelta(days=7),
            selected_observation_indices=(0,),
        )


def test_longitudinal_extraction_json_round_trip_preserves_bound_axes() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )

    assert LongitudinalExtraction.model_validate_json(extraction.model_dump_json()) == extraction


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("subject_id", "dog-outside-axis", "outside the declared extraction axes"),
        ("visit_id", "visit-outside-axis", "outside the declared extraction axes"),
        ("channel_index", 1, "channel index lies outside the declared axis"),
    ),
)
def test_serialized_longitudinal_values_must_belong_to_declared_axes(
    field: str,
    value: str | int,
    message: str,
) -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    payload = extraction.model_dump(mode="python")
    forged_value = dict(payload["values"][0])
    forged_value[field] = value
    payload["values"] = (forged_value,)

    with pytest.raises(ValidationError, match=message):
        LongitudinalExtraction.model_validate(payload)


def test_serialized_longitudinal_values_must_match_declared_channel_identity() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    payload = extraction.model_dump(mode="python")
    forged_value = dict(payload["values"][0])
    forged_value["channel"] = LongitudinalChannel(
        feature="albumin",
        modality=Modality.CLINICAL,
        unit="g/L",
    )
    payload["values"] = (forged_value,)

    with pytest.raises(ValidationError, match="does not match the declared channel"):
        LongitudinalExtraction.model_validate(payload)


def test_serialized_longitudinal_value_keys_must_be_unique() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    payload = extraction.model_dump(mode="python")
    payload["values"] = (payload["values"][0], payload["values"][0])

    with pytest.raises(ValidationError, match="value keys must be unique"):
        LongitudinalExtraction.model_validate(payload)


def test_serialized_longitudinal_exclusions_must_belong_to_declared_axes() -> None:
    channel = LongitudinalChannel(
        feature="albumin",
        modality=Modality.CLINICAL,
        unit="g/dL",
    )
    extraction = extract_visit_aligned_values(
        study(),
        visits=(VISIT,),
        channels=(channel,),
    )
    payload = extraction.model_dump(mode="python")
    forged_exclusion = dict(payload["exclusions"][0])
    forged_exclusion["subject_id"] = "dog-outside-axis"
    payload["exclusions"] = (forged_exclusion,)

    with pytest.raises(ValidationError, match="exclusion subject lies outside"):
        LongitudinalExtraction.model_validate(payload)


def test_serialized_longitudinal_values_cannot_reuse_one_source_row_across_channels() -> None:
    extraction = extract_visit_aligned_values(
        study(
            observation(day=7, value=3.0),
            observation(day=7, value=4.0, modality=Modality.PROTEOMICS),
        ),
        visits=(VISIT,),
        channels=(
            VisitFeature(feature="albumin", modality=Modality.CLINICAL),
            VisitFeature(feature="albumin", modality=Modality.PROTEOMICS),
        ),
    )
    payload = extraction.model_dump(mode="python")
    second = dict(payload["values"][1])
    second["selected_observation_indices"] = payload["values"][0]["selected_observation_indices"]
    payload["values"] = (payload["values"][0], second)

    with pytest.raises(ValidationError, match="reuses an observation across values"):
        LongitudinalExtraction.model_validate(payload)


def test_visit_vector_extraction_json_round_trip_preserves_shared_axes() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    vectors = complete_visit_vectors(extraction)

    assert VisitVectorExtraction.model_validate_json(vectors.model_dump_json()) == vectors


def test_serialized_visit_vector_keys_must_be_unique() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    vectors = complete_visit_vectors(extraction)
    payload = vectors.model_dump(mode="python")
    payload["vectors"] = (payload["vectors"][0], payload["vectors"][0])

    with pytest.raises(ValidationError, match="subject/visit keys must be unique"):
        VisitVectorExtraction.model_validate(payload)


def test_serialized_visit_vectors_must_share_one_exact_channel_axis() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    vectors = complete_visit_vectors(extraction)
    payload = vectors.model_dump(mode="python")
    second = dict(payload["vectors"][0])
    second["visit_id"] = "week-2"
    second["channels"] = (
        LongitudinalChannel(
            feature="albumin",
            modality=Modality.CLINICAL,
            unit="g/L",
        ),
    )
    second["selected_observation_indices"] = (1,)
    payload["vectors"] = (payload["vectors"][0], second)

    with pytest.raises(ValidationError, match="one exact shared axis"):
        VisitVectorExtraction.model_validate(payload)


def test_serialized_visit_vectors_cannot_reuse_source_rows_across_visits() -> None:
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    vectors = complete_visit_vectors(extraction)
    payload = vectors.model_dump(mode="python")
    second = dict(payload["vectors"][0])
    second["visit_id"] = "week-2"
    payload["vectors"] = (payload["vectors"][0], second)

    with pytest.raises(ValidationError, match="reuse a source observation across visits"):
        VisitVectorExtraction.model_validate(payload)


def test_mixed_units_fail_instead_of_being_averaged() -> None:
    source = study(
        observation(day=6, value=3.0, unit="g/dL"),
        observation(day=8, value=30.0, unit="g/L"),
    )

    with pytest.raises(LongitudinalAlignmentError) as caught:
        extract_visit_aligned_values(
            source,
            visits=(VISIT,),
            channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
        )

    exclusion = caught.value.exclusions[0]
    assert exclusion.reason is LongitudinalExclusionReason.MIXED_UNITS
    assert exclusion.observed_units == ("g/L", "g/dL")


def test_explicit_unit_mismatch_fails_closed() -> None:
    source = study(observation(day=7, value=3.0, unit="g/L"))
    channel = LongitudinalChannel(
        feature="albumin",
        modality=Modality.CLINICAL,
        unit="g/dL",
    )

    with pytest.raises(LongitudinalAlignmentError) as caught:
        extract_visit_aligned_values(source, visits=(VISIT,), channels=(channel,))

    assert caught.value.exclusions[0].reason is LongitudinalExclusionReason.UNIT_MISMATCH


@pytest.mark.parametrize(
    "policy",
    (
        AggregationPolicy.MEAN,
        AggregationPolicy.MEDIAN,
        AggregationPolicy.CLOSEST_TO_SCHEDULE,
    ),
)
def test_nonfinite_exact_rows_exclude_entire_visit_channel(
    policy: AggregationPolicy,
) -> None:
    source = study(
        observation(day=6, value=2.0, replicate_id="finite"),
        observation(day=7, value=float("nan"), replicate_id="nan"),
        observation(day=8, value=float("inf"), replicate_id="infinite"),
    )
    channel = LongitudinalChannel(
        feature="albumin",
        modality=Modality.CLINICAL,
        unit="g/dL",
        aggregation_policy=policy,
    )

    extraction = extract_visit_aligned_values(
        source,
        visits=(VISIT,),
        channels=(channel,),
    )
    vectors = complete_visit_vectors(extraction)

    assert extraction.values == ()
    nonfinite = tuple(
        item
        for item in extraction.exclusions
        if item.reason is LongitudinalExclusionReason.NONFINITE_OBSERVATION
    )
    assert len(nonfinite) == 1
    assert nonfinite[0].observation_indices == (1, 2)
    assert nonfinite[0].observed_modalities == (Modality.CLINICAL,)
    assert nonfinite[0].observed_units == ("g/dL",)
    assert vectors.vectors == ()
    assert vectors.exclusions[-1].reason is LongitudinalExclusionReason.INCOMPLETE_VISIT_VECTOR
    assert vectors.exclusions[-1].missing_channel_indices == (0,)


def test_wildcard_channel_does_not_hide_nonfinite_exact_replicate() -> None:
    extraction = extract_visit_aligned_values(
        study(
            observation(day=6, value=2.0),
            observation(day=8, value=float("-inf")),
        ),
        visits=(VISIT,),
        channels=(VisitFeature(feature="albumin"),),
    )

    assert extraction.values == ()
    assert len(extraction.exclusions) == 1
    exclusion = extraction.exclusions[0]
    assert exclusion.reason is LongitudinalExclusionReason.NONFINITE_OBSERVATION
    assert exclusion.observation_indices == (1,)
    assert exclusion.observed_modalities == (Modality.CLINICAL,)
    assert exclusion.observed_units == ("g/dL",)


def test_missing_anchor_and_incomplete_vector_are_structured() -> None:
    relative = ExpectedVisit(
        visit_id="after-dose",
        anchor_id="dose",
        offset=timedelta(days=7),
        required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    extraction = extract_visit_aligned_values(
        study(anchored=False),
        visits=(relative,),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    vectors = complete_visit_vectors(extraction)

    assert extraction.values == ()
    assert extraction.exclusions[0].reason is LongitudinalExclusionReason.MISSING_ANCHOR
    assert vectors.vectors == ()
    assert vectors.exclusions[-1].reason is LongitudinalExclusionReason.INCOMPLETE_VISIT_VECTOR
    assert vectors.exclusions[-1].missing_channel_indices == (0,)


def test_overlapping_visit_windows_cannot_reuse_one_source_observation() -> None:
    first = ExpectedVisit(
        visit_id="day-6",
        scheduled_at=START + timedelta(days=6),
        window_after=timedelta(days=2),
        required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    second = ExpectedVisit(
        visit_id="day-8",
        scheduled_at=START + timedelta(days=8),
        window_before=timedelta(days=2),
        required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )

    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0)),
        visits=(first, second),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    vectors = complete_visit_vectors(extraction)

    assert extraction.values == ()
    reuse = tuple(
        item
        for item in extraction.exclusions
        if item.reason is LongitudinalExclusionReason.REUSED_OBSERVATION_ACROSS_VISITS
    )
    assert tuple(item.visit_id for item in reuse) == ("day-6", "day-8")
    assert all(item.observation_indices == (0,) for item in reuse)
    assert vectors.vectors == ()


def test_requested_visit_order_excludes_nonchronological_observed_values() -> None:
    late = ExpectedVisit(
        visit_id="declared-first",
        scheduled_at=START + timedelta(days=7),
        required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    early = ExpectedVisit(
        visit_id="declared-second",
        scheduled_at=START,
        required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )

    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0), observation(day=0, value=2.0)),
        visits=(late, early),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )

    assert tuple(value.visit_id for value in extraction.values) == ("declared-first",)
    chronology = tuple(
        item
        for item in extraction.exclusions
        if item.reason is LongitudinalExclusionReason.NONCHRONOLOGICAL_OBSERVED_TIME
    )
    assert len(chronology) == 1
    assert chronology[0].visit_id == "declared-second"
    assert chronology[0].observation_indices == (0, 1)


def test_serialized_extraction_rejects_nonchronological_observed_values() -> None:
    follow_up = ExpectedVisit(
        visit_id="week-2",
        scheduled_at=START + timedelta(days=14),
        required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    extraction = extract_visit_aligned_values(
        study(observation(day=7, value=3.0), observation(day=14, value=4.0)),
        visits=(VISIT, follow_up),
        channels=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
    )
    payload = extraction.model_dump(mode="python")
    forged_follow_up = dict(payload["values"][1])
    forged_follow_up["effective_timestamp"] = START
    payload["values"] = (payload["values"][0], forged_follow_up)

    with pytest.raises(ValidationError, match="nonchronological observed visits"):
        LongitudinalExtraction.model_validate(payload)
