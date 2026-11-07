"""AI2C persistent player-identity foundation contracts.

Synthetic AI1 artifacts only: no detector, weights, video, or network. Frames
are sampled at 5 fps and every box is 60 px tall, so a 60 px foot-point move is
one body height.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from footballai_v2.execution.ai1.artifacts import (
    TRACKLETS_SCHEMA,
    BoundingBox,
    TrackObservation,
    TrackerProvenance,
    Tracklet,
    TrackletsArtifact,
    sha256_bytes,
    tracklets_to_parquet_bytes,
)
from footballai_v2.execution.ai2a.association import CannotLink, TeamEvidence
from footballai_v2.execution.ai2a.model import (
    GROUND_TRUTH_SCHEMA,
    AnnotationStatus,
    IdentityGroundTruthArtifact,
    PlayerIdentityId,
    ReviewState,
    SourceReference,
    TeamLabel,
    TrackletAnnotation,
    TrackletId,
    tracklet_id,
)
from footballai_v2.execution.ai2a.review import (
    REVIEW_SAMPLES_SCHEMA,
    ReviewSample,
    ReviewSamplesArtifact,
    select_representative_observations,
)
from footballai_v2.execution.ai2c.artifact import (
    PLAYER_IDENTITIES_SCHEMA,
    AssignmentStatus,
    LinkSource,
    player_identities_from_json_bytes,
    player_identities_to_json_bytes,
)
from footballai_v2.execution.ai2c.corrections import (
    CORRECTIONS_SCHEMA,
    CorrectionAction,
    CorrectionEntry,
    IdentityCorrections,
    corrections_from_json_bytes,
    corrections_to_json_bytes,
)
from footballai_v2.execution.ai2c.evaluation import evaluate_identities
from footballai_v2.execution.ai2c.inputs import (
    InputKind,
    SourceDescriptor,
    summaries_from_legacy_raw_tracks,
    summaries_from_review_samples,
    summaries_from_tracklets,
)
from footballai_v2.execution.ai2c.observations import iter_player_observations
from footballai_v2.execution.ai2c.resolver import (
    SELECTED_TEMPORAL_KINEMATIC,
    ResolverConfig,
    ResolverMode,
    resolve_identities,
)


TRACKLETS_SHA = "c" * 64
REPOSITORY = Path(__file__).resolve().parents[2]


def track(track_id: int, start: int, end: int, x: float = 100, x_end: float | None = None) -> Tracklet:
    x_end = x if x_end is None else x_end
    frames = range(start, end + 1)
    span = max(1, end - start)
    return Tracklet(
        track_id=track_id,
        class_id=0,
        semantic_class="person",
        observations=tuple(
            TrackObservation(
                frame_index=frame,
                timestamp_seconds=frame / 5,
                bbox=BoundingBox(
                    x + (x_end - x) * (frame - start) / span, 10,
                    x + (x_end - x) * (frame - start) / span + 20, 70,
                ),
                confidence=0.9,
            )
            for frame in frames
        ),
    )


def artifact_of(*items: Tracklet) -> TrackletsArtifact:
    return TrackletsArtifact(
        TRACKLETS_SCHEMA,
        TrackerProvenance("ByteTrack", "test", {"track_buffer": 30}, "detections", "d" * 64, "bounded-test"),
        tuple(items),
    )


def source() -> SourceDescriptor:
    return SourceDescriptor(
        source_id="bounded-clip",
        input_kind=InputKind.TRACKLETS,
        input_sha256=TRACKLETS_SHA,
        tracklets_sha256=TRACKLETS_SHA,
        detections_sha256="d" * 64,
        video_sha256=None,
    )


def resolve(*items: Tracklet, **kwargs):
    """Resolve with the library defaults (review_only unless a config is passed)."""
    if kwargs.get("corrections") is not None and "evidence_inputs" not in kwargs:
        kwargs["evidence_inputs"] = (corrections_input(kwargs["corrections"]),)
    return resolve_identities(summaries_from_tracklets(artifact_of(*items)), source(), **kwargs)


def stitch(*items: Tracklet, **kwargs):
    """Resolve with the explicitly selected experimental temporal_kinematic mode."""
    kwargs.setdefault("config", SELECTED_TEMPORAL_KINEMATIC)
    return resolve(*items, **kwargs)


def status_of(result, track_id: int) -> AssignmentStatus:
    return next(item for item in result.assignments if item.tracklet_id == tracklet_id(track_id)).status


def assignment(result, track_id: int):
    return next(item for item in result.assignments if item.tracklet_id == tracklet_id(track_id))


def team(label: TeamLabel, confidence: float = 0.9) -> TeamEvidence:
    return TeamEvidence(label, confidence, "jersey_color_v1")


def entry(sequence, action, *ids, revokes=None, reason="reviewed crops"):
    return CorrectionEntry(
        sequence=sequence,
        action=action,
        tracklet_ids=tuple(tracklet_id(item) for item in ids),
        reviewer="analyst-1",
        reviewed_at="2026-11-06T10:00:00Z",
        reason=reason,
        revokes_sequence=revokes,
    )


def corrections(*entries) -> IdentityCorrections:
    return IdentityCorrections(CORRECTIONS_SCHEMA, TRACKLETS_SHA, tuple(entries))


def corrections_input(log: IdentityCorrections):
    from footballai_v2.execution.ai2c.artifact import EvidenceInput

    return EvidenceInput("corrections", CORRECTIONS_SCHEMA, sha256_bytes(corrections_to_json_bytes(log)))


# --- Inputs -----------------------------------------------------------------


def test_tracklet_summaries_keep_exact_extents_and_endpoint_boxes():
    (summary,) = summaries_from_tracklets(artifact_of(track(3, 2, 6, x=100, x_end=140)))

    assert summary.tracklet_id == tracklet_id(3)
    assert (summary.start_frame, summary.end_frame, summary.observation_count) == (2, 6, 5)
    assert summary.first_bbox.x1 == 100 and summary.last_bbox.x1 == 140


def test_review_sample_reconstruction_matches_the_authoritative_summary():
    tracklets = artifact_of(track(1, 0, 0), track(2, 3, 4), track(3, 6, 12, x=50, x_end=80))
    samples = []
    for item in tracklets.tracklets:
        for selected in select_representative_observations(item):
            box = selected.observation.bbox
            samples.append(ReviewSample(
                tracklet_id(item.track_id), selected.label, selected.observation.frame_index,
                selected.observation.timestamp_seconds, (box.x1, box.y1, box.x2, box.y2),
                f"crops/{item.track_id}/{selected.label}.jpg", "e" * 64,
            ))
    review = ReviewSamplesArtifact(
        REVIEW_SAMPLES_SCHEMA,
        SourceReference("bounded-clip", "a" * 64, 20, "d" * 64, TRACKLETS_SHA),
        "early_middle_late/v1",
        tuple(samples),
    )

    reconstructed = summaries_from_review_samples(review)
    exact = summaries_from_tracklets(tracklets)

    for left, right in zip(reconstructed, exact):
        assert (left.start_frame, left.end_frame, left.first_bbox, left.last_bbox) == (
            right.start_frame, right.end_frame, right.first_bbox, right.last_bbox,
        )
    assert [item.observation_count for item in reconstructed] == [1, 2, None]
    assert all(item.semantic_class is None for item in reconstructed)


def test_legacy_raw_tracks_stream_first_and_last_rows_per_track():
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "frame_idx": [12, 0, 6, 6], "time_sec": [0.4, 0.0, 0.2, 0.2], "track_id": [1, 1, 1, 2],
        "cx": [30.0, 10.0, 20.0, 5.0], "cy": [50.0] * 4, "w": [10.0] * 4, "h": [40.0] * 4,
        "conf": [0.9] * 4,
    })
    buffer = io.BytesIO()
    pq.write_table(table, buffer)

    first, second = summaries_from_legacy_raw_tracks(buffer.getvalue(), batch_size=1)

    assert (first.start_frame, first.end_frame, first.observation_count) == (0, 12, 3)
    assert first.first_bbox.center_x == 10 and first.last_bbox.center_x == 30
    assert second.first_bbox.x1 == 0  # clipped at the image edge, never negative


def test_source_descriptor_requires_tracklet_checksums_except_for_legacy_input():
    with pytest.raises(ValueError):
        SourceDescriptor("clip", InputKind.REVIEW_SAMPLES, "a" * 64, None, None, None)
    with pytest.raises(ValueError):
        SourceDescriptor("clip", InputKind.TRACKLETS, "a" * 64, "b" * 64, None, None)
    SourceDescriptor("clip", InputKind.LEGACY_RAW_TRACKS, "a" * 64, None, None, None)


# --- Association and hard constraints ---------------------------------------


def test_clean_short_gap_successor_becomes_a_provisional_identity_not_a_verified_one():
    result = stitch(track(1, 0, 4, x=100), track(2, 7, 10, x=110))

    (identity,) = result.identities
    assert identity.player_identity_id == PlayerIdentityId("PID_000001")
    assert identity.status is AssignmentStatus.PROVISIONAL
    assert identity.tracklet_ids == (tracklet_id(1), tracklet_id(2))
    (link,) = result.links
    assert link.source is LinkSource.AUTOMATIC and 0 <= link.association_score <= 1
    assert result.resolver["association_score_semantics"].startswith("1 - gated cost; uncalibrated")
    assert "appearance_embedding" in result.resolver["evidence_channels_unavailable"]


def test_isolated_tracklet_is_unresolved_and_never_promoted_to_an_identity():
    result = resolve(track(1, 0, 4))

    assert result.identities == ()
    only = assignment(result, 1)
    assert only.status is AssignmentStatus.UNRESOLVED and only.player_identity_id is None
    assert only.reason == "no_accepted_candidate"


def test_temporally_overlapping_tracklets_are_never_merged():
    result = stitch(track(1, 0, 4, x=100), track(2, 4, 8, x=100))

    assert result.identities == ()


def test_explicit_cannot_link_blocks_an_otherwise_perfect_link():
    blocked = CannotLink(tracklet_id(1), tracklet_id(2), "simultaneous_spatially_distinct", 0, 0.0)

    result = stitch(track(1, 0, 4), track(2, 7, 10), cannot_links=(blocked,))

    assert result.identities == ()


def test_confident_team_conflict_blocks_but_weak_team_evidence_does_not():
    tracks = (track(1, 0, 4), track(2, 7, 10))
    confident = {tracklet_id(1): team(TeamLabel.TEAM_A), tracklet_id(2): team(TeamLabel.TEAM_B)}
    weak = {tracklet_id(1): team(TeamLabel.TEAM_A), tracklet_id(2): team(TeamLabel.TEAM_B, 0.5)}

    assert stitch(*tracks, team_evidence=confident).identities == ()
    assert len(stitch(*tracks, team_evidence=weak).identities) == 1


def test_gap_beyond_the_window_is_not_linked():
    assert stitch(track(1, 0, 4), track(2, 15, 18)).identities == ()


def test_kinematic_gate_rejects_implausible_jumps_that_temporal_only_accepts():
    tracks = (track(1, 0, 4, x=100), track(2, 7, 10, x=400))

    assert stitch(*tracks).identities == ()
    temporal = resolve(*tracks, config=ResolverConfig(mode=ResolverMode.TEMPORAL_ONLY))
    assert len(temporal.identities) == 1


def test_near_equal_successors_cause_an_abstention_not_a_guess():
    result = stitch(track(1, 0, 4, x=100), track(2, 7, 10, x=110), track(3, 7, 10, x=112))

    assert result.identities == ()
    (abstention,) = result.abstentions
    assert abstention.reason == "ambiguous_successor"
    assert {item for item, _ in abstention.candidates} == {tracklet_id(2), tracklet_id(3)}
    assert assignment(result, 1).reason == "abstained_ambiguous_or_conflicting"


def test_only_mutual_best_pairs_are_linked():
    result = stitch(track(1, 0, 4, x=170), track(4, 0, 4, x=130), track(2, 7, 10, x=128))

    (identity,) = result.identities
    assert identity.tracklet_ids == (tracklet_id(4), tracklet_id(2))
    assert any(item.reason == "not_mutual_best" and item.tracklet_id == tracklet_id(1) for item in result.abstentions)


def test_transitive_merge_is_rechecked_against_every_group_member():
    teams = {tracklet_id(1): team(TeamLabel.TEAM_A), tracklet_id(3): team(TeamLabel.TEAM_B)}

    result = stitch(track(1, 0, 4), track(2, 7, 10), track(3, 13, 16), team_evidence=teams)

    (identity,) = result.identities
    assert identity.tracklet_ids == (tracklet_id(1), tracklet_id(2))
    assert identity.team is TeamLabel.TEAM_A
    assert status_of(result, 3) is AssignmentStatus.UNRESOLVED
    assert any(item.reason == "group_constraint_conflict" for item in result.abstentions)


def test_suspected_shot_cut_blocks_automatic_links_across_it():
    burst = [track(100 + index, 6, 8, x=300 + 40 * index) for index in range(8)]

    result = stitch(track(1, 0, 4), track(2, 7, 10), *burst)

    assert result.summary["suspected_shot_cut_frames"] == [6]
    assert status_of(result, 1) is AssignmentStatus.UNRESOLVED
    no_barrier = ResolverConfig(mode=ResolverMode.TEMPORAL_KINEMATIC, shot_cut_min_births=None)
    disabled = stitch(track(1, 0, 4), track(2, 7, 10), *burst, config=no_barrier)
    assert status_of(disabled, 1) is AssignmentStatus.PROVISIONAL


def test_review_only_mode_makes_no_automatic_association():
    result = resolve(track(1, 0, 4), track(2, 7, 10), config=ResolverConfig(mode=ResolverMode.REVIEW_ONLY))

    assert result.links == () and result.summary["candidate_pairs_in_gap_window"] == 0


def test_resolution_is_deterministic_regardless_of_input_order():
    items = [track(5, 0, 4), track(2, 7, 10), track(9, 13, 16, x=300), track(1, 20, 22, x=50)]
    config = SELECTED_TEMPORAL_KINEMATIC
    forward = resolve_identities(summaries_from_tracklets(artifact_of(*items)), source(), config=config)
    backward = resolve_identities(
        tuple(reversed(summaries_from_tracklets(artifact_of(*items)))), source(), config=config
    )
    assert forward.links  # the experimental mode actually associates something here

    assert player_identities_to_json_bytes(forward) == player_identities_to_json_bytes(backward)


def test_resolver_rejects_evidence_for_unknown_tracklets():
    with pytest.raises(ValueError, match="unknown"):
        resolve(track(1, 0, 4), team_evidence={tracklet_id(7): team(TeamLabel.TEAM_A)})


# --- Artifact contract --------------------------------------------------------


def test_player_identities_artifact_round_trips_canonically():
    result = stitch(track(1, 0, 4), track(2, 7, 10), track(3, 30, 32))
    content = player_identities_to_json_bytes(result)

    decoded = player_identities_from_json_bytes(content)

    assert decoded == result
    assert player_identities_to_json_bytes(decoded) == content
    assert json.loads(content)["schema"] == PLAYER_IDENTITIES_SCHEMA
    assert "created_at" not in content.decode()


@pytest.mark.parametrize("tamper", ["claim_identity", "overlap", "gap", "orphan_link"])
def test_tampered_artifacts_are_rejected(tamper):
    value = json.loads(player_identities_to_json_bytes(stitch(track(1, 0, 4), track(2, 7, 10), track(3, 30, 32))))
    if tamper == "claim_identity":
        value["assignments"][2].update(player_identity_id="PID_000001", status="provisional")
    elif tamper == "overlap":
        value["assignments"][1]["start_frame"] = 3
    elif tamper == "gap":
        value["links"][0]["gap_frames"] = 9
    else:
        value["links"].append({**value["links"][0], "earlier": "tracklet_000003"})

    with pytest.raises(ValueError):
        player_identities_from_json_bytes(json.dumps(value).encode())


# --- Human corrections ----------------------------------------------------------


def test_must_link_creates_a_confirmed_identity_with_audit_history():
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 3))

    result = resolve(track(1, 0, 4), track(3, 40, 44, x=600), corrections=log)

    (identity,) = result.identities
    assert identity.status is AssignmentStatus.CONFIRMED and identity.association_score_min is None
    assert result.links[0].source is LinkSource.HUMAN and result.links[0].correction_sequence == 1
    assert result.corrections_applied[0]["outcome"] == "applied"
    assert result.corrections_applied[0]["reviewer"] == "analyst-1"


def test_must_link_cannot_override_temporal_overlap():
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 2))

    with pytest.raises(ValueError, match="temporal_overlap"):
        resolve(track(1, 0, 4), track(2, 3, 8), corrections=log)


def test_human_must_link_may_override_weak_team_colour_evidence():
    teams = {tracklet_id(1): team(TeamLabel.TEAM_A), tracklet_id(2): team(TeamLabel.TEAM_B)}
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 2))

    result = resolve(track(1, 0, 4), track(2, 30, 34), team_evidence=teams, corrections=log)

    assert result.identities[0].team is TeamLabel.UNKNOWN


def test_cannot_link_and_exclude_corrections_constrain_automatic_association():
    blocked = stitch(track(1, 0, 4), track(2, 7, 10), corrections=corrections(entry(1, CorrectionAction.CANNOT_LINK, 1, 2)))
    excluded = stitch(track(1, 0, 4), track(2, 7, 10), corrections=corrections(entry(1, CorrectionAction.EXCLUDE, 2, reason="match official")))

    assert blocked.identities == ()
    assert status_of(excluded, 2) is AssignmentStatus.EXCLUDED
    assert assignment(excluded, 2).reason == "human_exclude:correction_1"


def test_revoke_withdraws_a_decision_without_rewriting_history():
    log = corrections(
        entry(1, CorrectionAction.CANNOT_LINK, 1, 2),
        entry(2, CorrectionAction.REVOKE, revokes=1, reason="wrong pair selected"),
    )

    result = stitch(track(1, 0, 4), track(2, 7, 10), corrections=log)

    assert len(result.identities) == 1
    assert [item["outcome"] for item in result.corrections_applied] == ["revoked", "recorded"]
    assert corrections_from_json_bytes(corrections_to_json_bytes(log)) == log


@pytest.mark.parametrize("entries", [
    (entry(2, CorrectionAction.EXCLUDE, 1),),
    (entry(1, CorrectionAction.MUST_LINK, 1, 2), entry(2, CorrectionAction.CANNOT_LINK, 1, 2)),
    (entry(1, CorrectionAction.EXCLUDE, 1), entry(2, CorrectionAction.MUST_LINK, 1, 2)),
    (entry(1, CorrectionAction.EXCLUDE, 1), entry(2, CorrectionAction.REVOKE, revokes=1), entry(3, CorrectionAction.REVOKE, revokes=2)),
])
def test_invalid_correction_logs_are_rejected(entries):
    with pytest.raises(ValueError):
        corrections(*entries)


def test_corrections_are_pinned_to_one_tracklets_artifact():
    log = IdentityCorrections(CORRECTIONS_SCHEMA, "f" * 64, (entry(1, CorrectionAction.EXCLUDE, 1),))

    with pytest.raises(ValueError, match="pinned"):
        resolve(track(1, 0, 4), corrections=log)


def test_correction_entries_require_reason_reviewer_and_utc_time():
    with pytest.raises(ValueError):
        entry(1, CorrectionAction.EXCLUDE, 1, reason=" ")
    with pytest.raises(ValueError):
        CorrectionEntry(1, CorrectionAction.EXCLUDE, (tracklet_id(1),), "Jane Doe", "2026-11-06T10:00:00Z", "x")
    with pytest.raises(ValueError):
        CorrectionEntry(1, CorrectionAction.EXCLUDE, (tracklet_id(1),), "a1", "2026-11-06 10:00", "x")


# --- Evaluation -------------------------------------------------------------------


def label(track_id, identity, start, end, *, team_label=TeamLabel.TEAM_A, role="PLAYER", status=AnnotationStatus.CONFIRMED):
    return TrackletAnnotation(
        tracklet_id(track_id), PlayerIdentityId(identity) if identity else None, team_label, role,
        start, end, start / 5, end / 5, 0.9 if identity else None, status, "reviewer", "manual", "",
        ReviewState.REVIEWED,
    )


def truth(*annotations) -> IdentityGroundTruthArtifact:
    return IdentityGroundTruthArtifact(
        GROUND_TRUTH_SCHEMA, SourceReference("bounded-clip", "a" * 64, 20, "d" * 64, TRACKLETS_SHA), annotations,
    )


def test_evaluation_counts_true_and_false_merges_with_denominators():
    result = stitch(track(1, 0, 4), track(2, 7, 10), track(3, 30, 32), track(4, 35, 37))
    ground = truth(
        label(1, "TEAM_A_01", 0, 4), label(2, "TEAM_A_02", 7, 10),
        label(3, "TEAM_A_03", 30, 32), label(4, "TEAM_A_03", 35, 37),
    )

    metrics = evaluate_identities(result, ground)

    assert metrics["pairwise"]["false_positive"] == 1  # 1+2 merged but different players
    assert metrics["pairwise"]["true_positive"] == 1
    assert metrics["false_merges"]["pairs"][0]["ground_truth"] == ["TEAM_A_01", "TEAM_A_02"]
    assert metrics["scope"] == {
        "identities": ["TEAM_A_01", "TEAM_A_02", "TEAM_A_03"],
        "confirmed_labelled_tracklets": 4, "positive_pairs": 1, "negative_pairs": 5,
    }
    assert metrics["temporal_overlap_violations"] == 0
    assert set(metrics["not_measured"]) == {"HOTA", "IDF1", "ID_switches"}


def test_precision_is_null_rather_than_zero_without_predicted_pairs():
    result = resolve(track(1, 0, 4), track(2, 30, 34))

    metrics = evaluate_identities(result, truth(label(1, "TEAM_A_01", 0, 4), label(2, "TEAM_A_01", 30, 34)))

    assert metrics["pairwise"]["precision"] is None and metrics["pairwise"]["recall"] == 0.0


def test_scoped_evaluation_and_non_player_contamination():
    result = stitch(track(1, 0, 4), track(2, 7, 10), track(3, 30, 32))
    ground = truth(
        label(1, "TEAM_A_01", 0, 4),
        label(2, None, 7, 10, team_label=TeamLabel.UNKNOWN, role="STAFF", status=AnnotationStatus.UNKNOWN),
        label(3, "TEAM_A_02", 30, 32),
    )

    metrics = evaluate_identities(result, ground, identity_scope=[PlayerIdentityId("TEAM_A_01")])

    assert metrics["scope"]["confirmed_labelled_tracklets"] == 1
    assert metrics["labelled_non_player_contamination_pairs"] == 1
    with pytest.raises(ValueError, match="unlabelled"):
        evaluate_identities(result, ground, identity_scope=[PlayerIdentityId("TEAM_B_09")])


def test_evaluation_refuses_ground_truth_from_another_tracklets_artifact():
    other = IdentityGroundTruthArtifact(
        GROUND_TRUTH_SCHEMA, SourceReference("x", "a" * 64, 20, "d" * 64, "9" * 64), (),
    )

    with pytest.raises(ValueError, match="pinned"):
        evaluate_identities(resolve(track(1, 0, 4)), other)


# --- Downstream observation contract --------------------------------------------


def test_observations_default_to_human_confirmed_identities_only():
    tracklets = artifact_of(track(1, 0, 4), track(2, 7, 10), track(3, 40, 42), track(4, 60, 61), track(5, 80, 81))
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 4, 5))
    result = resolve_identities(
        summaries_from_tracklets(tracklets), source(), config=SELECTED_TEMPORAL_KINEMATIC,
        corrections=log, evidence_inputs=(corrections_input(log),),
    )
    digest = sha256_bytes(player_identities_to_json_bytes(result))

    confirmed = list(iter_player_observations(tracklets, TRACKLETS_SHA, result, digest))
    provisional = list(iter_player_observations(tracklets, TRACKLETS_SHA, result, digest, include_provisional=True))
    everything = list(iter_player_observations(
        tracklets, TRACKLETS_SHA, result, digest, include_provisional=True, include_unresolved=True,
    ))

    assert {item.identity_status for item in confirmed} == {AssignmentStatus.CONFIRMED}
    assert len(confirmed) == 4
    assert len(provisional) == 4 + 9
    assert len(everything) == 4 + 9 + 3
    assert all(item.pitch_xy_m is None and item.pitch_calibration == "uncalibrated" for item in everything)
    assert all(item.source_identities_sha256 == digest for item in everything)


def test_observations_refuse_identities_from_other_tracklets():
    tracklets = artifact_of(track(1, 0, 4))

    with pytest.raises(ValueError, match="different tracklets"):
        list(iter_player_observations(tracklets, "9" * 64, resolve(track(1, 0, 4)), "0" * 64))


# --- CLI, isolation and committed evidence ------------------------------------------


def test_cli_resolves_cached_tracklets_and_refuses_to_overwrite(tmp_path):
    from footballai_v2.execution.ai2c.cli import main

    parquet = tmp_path / "tracklets.parquet"
    parquet.write_bytes(tracklets_to_parquet_bytes(artifact_of(track(1, 0, 4), track(2, 7, 10))))
    output = tmp_path / "identities.json"
    args = [
        "resolve", "--tracklets", str(parquet), "--source-id", "bounded-clip",
        "--mode", "temporal_kinematic", "--output", str(output),
    ]

    assert main(args) == 0
    result = player_identities_from_json_bytes(output.read_bytes())
    assert result.source.tracklets_sha256 == sha256_bytes(parquet.read_bytes())
    assert len(result.identities) == 1
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(args)


def test_ai2c_import_does_not_load_detector_or_deep_learning_frameworks():
    code = (
        "import sys; import footballai_v2.execution.ai2c.cli; "
        "assert not {'ultralytics', 'torch', 'cv2'} & set(sys.modules), sorted(sys.modules)"
    )

    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


AI2A_DIR = REPOSITORY / "evaluation" / "ai2a" / "ai1-bounded-18s-validation"
AI2C_DIR = REPOSITORY / "evaluation" / "ai2c"


def test_committed_ai2a_clip_identities_reproduce_byte_for_byte():
    from footballai_v2.execution.ai2a.review import review_samples_from_json_bytes
    from footballai_v2.execution.ai2c.evidence import ai2a_evidence_from_evaluation_bytes
    from footballai_v2.execution.ai2c.inputs import source_for_review_samples

    content = (AI2A_DIR / "review-samples.json").read_bytes()
    review = review_samples_from_json_bytes(content)
    clip_source = source_for_review_samples(content, review)
    teams, links, inputs = ai2a_evidence_from_evaluation_bytes(
        (AI2A_DIR / "identity-evaluation.json").read_bytes(), tracklets_sha256=clip_source.tracklets_sha256
    )

    result = resolve_identities(
        summaries_from_review_samples(review), clip_source, config=SELECTED_TEMPORAL_KINEMATIC,
        team_evidence=teams, cannot_links=links, evidence_inputs=inputs,
    )

    committed = (AI2C_DIR / "ai1-bounded-18s-validation" / "player-identities.json").read_bytes()
    assert player_identities_to_json_bytes(result) == committed


def test_committed_benchmark_report_is_consistent_with_the_contract():
    report = json.loads((AI2C_DIR / "identity-resolution-evaluation.json").read_bytes())

    assert report["schema"] == "footballai.identity-resolution-evaluation/v1"
    assert ResolverConfig.from_dict(report["selected_configuration"]) == SELECTED_TEMPORAL_KINEMATIC
    assert not set(report["protocol"]["development_identities"]) & set(report["protocol"]["evaluation_identities"])
    for result in report["results"].values():
        for scope in ("development", "evaluation", "all_labelled_diagnostic"):
            assert result[scope]["temporal_overlap_violations"] == 0
            assert result[scope]["cross_team_violations"] == 0
    for path in sorted(AI2C_DIR.glob("*/player-identities.json")):
        player_identities_from_json_bytes(path.read_bytes())


# --- Safety gate: automatic association is experimental and opt-in -----------------
#
# The AI2C benchmark found zero held-out true positives and one certain false
# merge (STAFF_03 -> TEAM_A_08) on a second clip, so automatic association must
# never run unless explicitly requested, and must never surface as verified.

EXPERIMENTAL_MODES = (ResolverMode.TEMPORAL_ONLY, ResolverMode.TEMPORAL_KINEMATIC)


def test_resolver_config_defaults_to_review_only():
    assert ResolverConfig().mode is ResolverMode.REVIEW_ONLY


def test_resolver_without_an_explicit_mode_creates_no_automatic_identities():
    # A clean short-gap successor that temporal_kinematic would link.
    result = resolve_identities(summaries_from_tracklets(artifact_of(track(1, 0, 4), track(2, 7, 10))), source())

    assert result.identities == () and result.links == ()
    assert {item.status for item in result.assignments} == {AssignmentStatus.UNRESOLVED}
    assert all(item.player_identity_id is None for item in result.assignments)
    assert result.resolver["configuration"]["mode"] == "review_only"
    assert result.summary["automatic_links"] == 0
    assert result.summary["candidate_pairs_in_gap_window"] == 0
    assert "temporal_gap" not in result.resolver["evidence_channels_used"]


@pytest.mark.parametrize("mode", EXPERIMENTAL_MODES)
def test_experimental_modes_link_only_when_selected_and_stay_provisional(mode):
    result = resolve(track(1, 0, 4), track(2, 7, 10), config=ResolverConfig(mode=mode))

    (identity,) = result.identities
    assert identity.status is AssignmentStatus.PROVISIONAL
    assert {item.status for item in result.assignments} == {AssignmentStatus.PROVISIONAL}
    assert result.summary["confirmed_identities"] == 0
    assert result.resolver["configuration"]["mode"] == mode.value


def test_selected_experimental_configuration_is_explicit_and_matches_the_committed_benchmark():
    from footballai_v2.execution.ai2c.benchmark import BASELINES
    from footballai_v2.execution.ai2c.resolver import SELECTED_TEMPORAL_KINEMATIC

    report = json.loads((AI2C_DIR / "identity-resolution-evaluation.json").read_bytes())

    assert SELECTED_TEMPORAL_KINEMATIC.mode is ResolverMode.TEMPORAL_KINEMATIC
    assert SELECTED_TEMPORAL_KINEMATIC.to_dict() == report["selected_configuration"]
    assert report["results"]["temporal_kinematic_selected"]["configuration"] == report["selected_configuration"]
    for name, config in BASELINES.items():
        assert report["results"][name]["configuration"] == config.to_dict()


def test_human_must_link_is_confirmed_under_the_safe_default_and_the_rest_stays_unknown():
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 3))
    tracks = artifact_of(track(1, 0, 4), track(2, 7, 10), track(3, 40, 44, x=600))

    result = resolve_identities(
        summaries_from_tracklets(tracks), source(), corrections=log, evidence_inputs=(corrections_input(log),),
    )

    (identity,) = result.identities
    assert identity.status is AssignmentStatus.CONFIRMED
    assert identity.tracklet_ids == (tracklet_id(1), tracklet_id(3))
    assert status_of(result, 2) is AssignmentStatus.UNRESOLVED
    assert assignment(result, 2).player_identity_id is None


def test_a_human_link_never_launders_an_automatic_link_into_a_confirmed_identity():
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 2, 3))
    tracks = artifact_of(track(1, 0, 4), track(2, 7, 10), track(3, 40, 44, x=600))

    result = resolve_identities(
        summaries_from_tracklets(tracks), source(), config=ResolverConfig(mode=ResolverMode.TEMPORAL_KINEMATIC),
        corrections=log, evidence_inputs=(corrections_input(log),),
    )

    (identity,) = result.identities
    assert (identity.automatic_links, identity.human_links) == (1, 1)
    assert identity.status is AssignmentStatus.PROVISIONAL
    assert {status_of(result, item) for item in (1, 2, 3)} == {AssignmentStatus.PROVISIONAL}


def test_resolver_requires_provenance_for_the_corrections_it_applies():
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 3))
    tracks = summaries_from_tracklets(artifact_of(track(1, 0, 4), track(3, 40, 44, x=600)))

    with pytest.raises(ValueError, match="corrections evidence input"):
        resolve_identities(tracks, source(), corrections=log)
    with pytest.raises(ValueError, match="corrections evidence input"):
        resolve_identities(tracks, source(), evidence_inputs=(corrections_input(log),))


def test_confirmed_identities_disclose_that_reviewer_handles_are_unauthenticated():
    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 3))
    tracks = summaries_from_tracklets(artifact_of(track(1, 0, 4), track(3, 40, 44, x=600)))

    reviewed = resolve_identities(tracks, source(), corrections=log, evidence_inputs=(corrections_input(log),))
    automatic = resolve_identities(tracks, source())

    assert any("not authenticated" in item for item in reviewed.limitations)
    assert not any("not authenticated" in item for item in automatic.limitations)


def _forge_confirmed(value: dict, *, keep_counts: bool = False) -> dict:
    """Relabel the first automatic link as a human one, as a tampering reviewer might."""
    link = value["links"][0]
    link.update(source="human", correction_sequence=1, cost=None, association_score=None)
    if not keep_counts:
        identity = value["identities"][0]
        identity.update(status="confirmed", automatic_links=0, human_links=1, association_score_min=None)
        for item in value["assignments"]:
            if item["player_identity_id"] == identity["player_identity_id"]:
                item["status"] = "confirmed"
    return value


@pytest.mark.parametrize(("tamper", "reason"), [
    ("forge_confirmed", "require a corrections evidence input"),
    ("source_count_mismatch", "link count disagrees"),
    ("duplicate_link", "do not connect"),
    ("fake_audit_pair", "does not match correction 1"),
])
def test_confirmed_status_cannot_be_forged_without_traceable_correction_evidence(tamper, reason):
    experimental = ResolverConfig(mode=ResolverMode.TEMPORAL_KINEMATIC)
    if tamper == "duplicate_link":
        value = json.loads(player_identities_to_json_bytes(
            resolve(track(1, 0, 4), track(2, 7, 10), track(3, 13, 16), config=experimental)
        ))
        assert len(value["identities"][0]["tracklet_ids"]) == 3
        value["links"][1] = dict(value["links"][0])
    else:
        value = json.loads(player_identities_to_json_bytes(resolve(track(1, 0, 4), track(2, 7, 10), config=experimental)))
        value = _forge_confirmed(value, keep_counts=tamper == "source_count_mismatch")
        if tamper == "fake_audit_pair":
            value["evidence_inputs"].append({"role": "corrections", "schema": CORRECTIONS_SCHEMA, "sha256": "e" * 64})
            value["corrections_applied"].append({
                "sequence": 1, "action": "must_link", "outcome": "applied",
                "tracklet_ids": ["tracklet_000001", "tracklet_000009"], "revokes_sequence": None,
                "reviewer": "analyst-1", "reviewed_at": "2026-11-06T10:00:00Z",
            })

    with pytest.raises(ValueError, match=reason):
        player_identities_from_json_bytes(json.dumps(value).encode())


def test_human_exclusions_must_cite_an_applied_exclude_correction():
    log = corrections(entry(1, CorrectionAction.EXCLUDE, 2, reason="match official"))
    excluded = json.loads(player_identities_to_json_bytes(resolve(track(1, 0, 4), track(2, 7, 10), corrections=log)))
    assert player_identities_from_json_bytes(json.dumps(excluded).encode())  # the genuine artifact loads

    forged = json.loads(player_identities_to_json_bytes(resolve(track(1, 0, 4), track(2, 7, 10))))
    forged["assignments"][0].update(status="excluded", reason="human_exclude:correction_7")
    with pytest.raises(ValueError, match="corrections evidence input"):
        player_identities_from_json_bytes(json.dumps(forged).encode())

    excluded["assignments"][0].update(status="excluded", reason="human_exclude:correction_1")
    with pytest.raises(ValueError, match="does not cite an applied exclude"):
        player_identities_from_json_bytes(json.dumps(excluded).encode())


def test_malformed_audit_entries_raise_value_errors_even_when_built_directly():
    from dataclasses import replace

    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 3))
    result = resolve(track(1, 0, 4), track(3, 40, 44, x=600), corrections=log)
    broken = dict(result.corrections_applied[0], sequence=[1])

    with pytest.raises(ValueError, match="sequences must be integers"):
        replace(result, corrections_applied=(broken,))


def test_resolver_verifies_the_corrections_checksum_against_the_applied_log():
    from footballai_v2.execution.ai2c.artifact import EvidenceInput

    log = corrections(entry(1, CorrectionAction.MUST_LINK, 1, 3))
    tracks = summaries_from_tracklets(artifact_of(track(1, 0, 4), track(3, 40, 44, x=600)))

    with pytest.raises(ValueError, match="does not match the canonical bytes"):
        resolve_identities(
            tracks, source(), corrections=log, evidence_inputs=(EvidenceInput("corrections", CORRECTIONS_SCHEMA, "e" * 64),),
        )


def test_cli_accepts_only_canonical_corrections_and_records_their_file_checksum(tmp_path):
    from footballai_v2.execution.ai2c.cli import main

    tracklets = tracklets_to_parquet_bytes(artifact_of(track(1, 0, 4), track(3, 40, 44, x=600)))
    parquet = tmp_path / "tracklets.parquet"
    parquet.write_bytes(tracklets)
    log = IdentityCorrections(CORRECTIONS_SCHEMA, sha256_bytes(tracklets), (entry(1, CorrectionAction.MUST_LINK, 1, 3),))
    canonical = tmp_path / "corrections.json"
    canonical.write_bytes(corrections_to_json_bytes(log))
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(json.loads(canonical.read_bytes())))
    base = ["resolve", "--tracklets", str(parquet), "--source-id", "bounded-clip"]

    with pytest.raises(SystemExit, match="not a canonical"):
        main([*base, "--corrections", str(pretty), "--output", str(tmp_path / "rejected.json")])
    assert main([*base, "--corrections", str(canonical), "--output", str(tmp_path / "identities.json")]) == 0
    result = player_identities_from_json_bytes((tmp_path / "identities.json").read_bytes())
    assert [item.status for item in result.identities] == [AssignmentStatus.CONFIRMED]
    (recorded,) = [item for item in result.evidence_inputs if item.role == "corrections"]
    assert recorded.sha256 == sha256_bytes(canonical.read_bytes())


def test_observations_exclude_automatic_identities_by_default():
    tracklets = artifact_of(track(1, 0, 4), track(2, 7, 10))
    result = resolve_identities(
        summaries_from_tracklets(tracklets), source(), config=ResolverConfig(mode=ResolverMode.TEMPORAL_KINEMATIC),
    )
    digest = sha256_bytes(player_identities_to_json_bytes(result))

    assert list(iter_player_observations(tracklets, TRACKLETS_SHA, result, digest)) == []
    opted_in = list(iter_player_observations(tracklets, TRACKLETS_SHA, result, digest, include_provisional=True))
    assert opted_in and {item.identity_status for item in opted_in} == {AssignmentStatus.PROVISIONAL}


def test_observations_verify_the_identities_checksum():
    tracklets = artifact_of(track(1, 0, 4), track(2, 7, 10))
    result = resolve_identities(summaries_from_tracklets(tracklets), source())

    with pytest.raises(ValueError, match="checksum"):
        list(iter_player_observations(tracklets, TRACKLETS_SHA, result, "0" * 64))


def test_cli_resolve_without_mode_uses_the_review_only_policy(tmp_path):
    from footballai_v2.execution.ai2c.cli import build_parser, main

    parquet = tmp_path / "tracklets.parquet"
    parquet.write_bytes(tracklets_to_parquet_bytes(artifact_of(track(1, 0, 4), track(2, 7, 10))))
    output = tmp_path / "identities.json"

    assert build_parser().parse_args(["resolve", "--tracklets", "x", "--output", "y"]).mode == "review_only"
    assert main(["resolve", "--tracklets", str(parquet), "--source-id", "bounded-clip", "--output", str(output)]) == 0
    result = player_identities_from_json_bytes(output.read_bytes())
    assert result.resolver["configuration"]["mode"] == "review_only"
    assert result.identities == () and result.links == ()


@pytest.mark.parametrize("mode", EXPERIMENTAL_MODES)
def test_cli_experimental_modes_require_explicit_selection_and_warn(tmp_path, capsys, mode):
    from footballai_v2.execution.ai2c.cli import main

    parquet = tmp_path / "tracklets.parquet"
    parquet.write_bytes(tracklets_to_parquet_bytes(artifact_of(track(1, 0, 4), track(2, 7, 10))))
    output = tmp_path / "identities.json"

    assert main([
        "resolve", "--tracklets", str(parquet), "--source-id", "bounded-clip", "--mode", mode.value,
        "--output", str(output),
    ]) == 0
    result = player_identities_from_json_bytes(output.read_bytes())
    assert result.resolver["configuration"]["mode"] == mode.value
    assert {item.status for item in result.identities} == {AssignmentStatus.PROVISIONAL}
    assert "experimental" in capsys.readouterr().err
