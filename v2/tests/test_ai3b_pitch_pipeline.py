"""AI3-B: pitch_calibration run stage, spatial artifacts, worker integration and API.

Every fixture is synthetic (pinhole cameras with a known ground truth, synthetic
AI1 parquet bytes). These tests prove the contracts, bindings and safety gates of
the pipeline integration. They do not measure calibration accuracy on real
broadcast footage.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from fakes.ai1_spatial import (
    CAMERA,
    OTHER_CAMERA,
    PLAYERS,
    RUN_ID,
    VIDEO_SHA,
    ai1_bytes,
    notes,
)
from footballai_v2.execution.ai1.artifacts import sha256_bytes
from footballai_v2.execution.ai3.annotations import AnnotationOrigin, CameraMotion, annotations_to_json_bytes
from footballai_v2.execution.ai3.artifact import pitch_calibration_from_json_bytes
from footballai_v2.execution.ai3.pipeline_stage import (
    DIAGNOSTIC_LABEL,
    NO_ANNOTATIONS_REASON,
    PITCH_SPACE_DIAGNOSTIC_SCHEMA,
    SpatialStageError,
    diagnostic_to_json_bytes,
    load_pitch_annotations_parameter,
    pitch_annotations_parameter,
    pitch_space_view,
    read_projected_observations,
    run_identity_stage,
    run_pitch_calibration_stage,
    verify_ai1_artifacts,
)


def stage(annotations=None, *, frames=range(0, 41), data_origin="synthetic", camera=CAMERA):
    det_bytes, trk_bytes = ai1_bytes(frames, camera=camera)
    ai1 = verify_ai1_artifacts(det_bytes, trk_bytes, video_sha256=VIDEO_SHA, recorded_tracklets_sha256=sha256_bytes(trk_bytes))
    identities = run_identity_stage(ai1, trk_bytes, source_id=f"run-{RUN_ID}")
    pinned = None
    if annotations is not None:
        parameter = pitch_annotations_parameter(
            annotations_to_json_bytes(annotations), video_sha256=VIDEO_SHA, data_origin=data_origin,
        )
        pinned = load_pitch_annotations_parameter(parameter)
    return ai1, identities, run_pitch_calibration_stage(
        ai1, identities, run_id=RUN_ID, data_origin=data_origin, annotations=pinned,
    )


# --- annotation attachment ---------------------------------------------------------------


def test_annotation_parameter_pins_canonical_bytes_and_round_trips():
    content = annotations_to_json_bytes(notes())
    parameter = pitch_annotations_parameter(content, video_sha256=VIDEO_SHA, data_origin="synthetic")
    assert parameter["sha256"] == sha256_bytes(content) == parameter["uploaded_sha256"]
    assert parameter["keyframes"] == 2 and parameter["origin"] == "synthetic"
    loaded, digest = load_pitch_annotations_parameter(json.loads(json.dumps(parameter)))
    assert digest == parameter["sha256"] and annotations_to_json_bytes(loaded) == content


@pytest.mark.parametrize("mutate, code", [
    (lambda a: dataclasses.replace(a, source=dataclasses.replace(a.source, video_sha256=None)), "pitch_annotations_unbound"),
    (lambda a: dataclasses.replace(a, source=dataclasses.replace(a.source, video_sha256="f" * 64)),
     "pitch_annotations_video_mismatch"),
    (lambda a: dataclasses.replace(a, frames=()), "pitch_annotations_no_keyframes"),
    (lambda a: dataclasses.replace(a, source=dataclasses.replace(a.source, tracklets_sha256="1" * 64)),
     "pitch_annotations_bound_to_ai1"),
    (lambda a: dataclasses.replace(a, source=dataclasses.replace(a.source, detections_sha256="2" * 64)),
     "pitch_annotations_bound_to_ai1"),
])
def test_annotation_parameter_refuses_unbound_foreign_or_empty_files(mutate, code):
    with pytest.raises(SpatialStageError) as error:
        pitch_annotations_parameter(annotations_to_json_bytes(mutate(notes())), video_sha256=VIDEO_SHA, data_origin="synthetic")
    assert error.value.code == code


def test_annotation_parameter_refuses_invalid_large_and_synthetic_on_real():
    with pytest.raises(SpatialStageError) as error:
        pitch_annotations_parameter(b"{not json", video_sha256=VIDEO_SHA, data_origin="real")
    assert error.value.code == "invalid_pitch_annotations"
    with pytest.raises(SpatialStageError) as error:
        pitch_annotations_parameter(b" " * (1024 * 1024 + 1), video_sha256=VIDEO_SHA, data_origin="real")
    assert error.value.code == "pitch_annotations_too_large"
    with pytest.raises(SpatialStageError) as error:
        pitch_annotations_parameter(annotations_to_json_bytes(notes()), video_sha256=VIDEO_SHA, data_origin="real")
    assert error.value.code == "pitch_annotations_synthetic"


def test_tampered_annotation_parameter_is_rejected_by_the_worker():
    parameter = pitch_annotations_parameter(annotations_to_json_bytes(notes()), video_sha256=VIDEO_SHA, data_origin="synthetic")
    tampered = json.loads(json.dumps(parameter))
    tampered["document"]["frames"][0]["landmarks"][0]["pixel_x"] += 25.0
    with pytest.raises(SpatialStageError) as error:
        load_pitch_annotations_parameter(tampered)
    assert error.value.code == "pitch_annotations_checksum_mismatch"
    with pytest.raises(SpatialStageError) as error:
        load_pitch_annotations_parameter({**parameter, "extra": 1})
    assert error.value.code == "pitch_annotations_parameter_invalid"


# --- verified AI1 inputs ------------------------------------------------------------------


def test_ai1_verification_checks_checksums_video_binding_and_frame_alignment():
    det_bytes, trk_bytes = ai1_bytes()
    verified = verify_ai1_artifacts(det_bytes, trk_bytes, video_sha256=VIDEO_SHA, recorded_tracklets_sha256=sha256_bytes(trk_bytes))
    assert verified.frames[0] == (0, 0.0) and len(verified.frames) == 41 and verified.frame_size == (1920, 1080)
    cases = [
        (dict(video_sha256="e" * 64), "ai1_video_binding_mismatch"),
        (dict(video_sha256=VIDEO_SHA, recorded_tracklets_sha256="0" * 64), "ai1_tracklets_checksum_mismatch"),
    ]
    for kwargs, code in cases:
        with pytest.raises(SpatialStageError) as error:
            verify_ai1_artifacts(det_bytes, trk_bytes, **kwargs)
        assert error.value.code == code
    other_det, _ = ai1_bytes(range(0, 40))
    with pytest.raises(SpatialStageError) as error:
        verify_ai1_artifacts(other_det, trk_bytes, video_sha256=VIDEO_SHA)
    assert error.value.code == "ai1_tracklets_binding_mismatch"
    with pytest.raises(SpatialStageError) as error:
        verify_ai1_artifacts(b"PAR1junk", trk_bytes, video_sha256=VIDEO_SHA)
    assert error.value.code == "ai1_artifact_unreadable"


def test_ai1_tracklet_timestamps_must_match_sampled_frames():
    det_bytes, trk_bytes = ai1_bytes(shift_timestamp=True)
    with pytest.raises(SpatialStageError) as error:
        verify_ai1_artifacts(det_bytes, trk_bytes, video_sha256=VIDEO_SHA)
    assert error.value.code == "ai1_frame_alignment_mismatch"


def test_identity_stage_is_review_only_and_invents_no_identity():
    _, identities, _ = stage()
    counts = identities.counts()
    assert counts == {"confirmed": 0, "provisional": 0, "unresolved": 3, "excluded": 1}
    assert identities.artifact.resolver["configuration"]["mode"] == "review_only"
    assert not identities.artifact.identities and not identities.artifact.links


# --- pitch_calibration stage ---------------------------------------------------------------


def test_without_annotations_every_frame_is_explicitly_uncalibrated():
    _, _, result = stage(None, data_origin="real")
    assert result.calibration is None and result.calibration_bytes is None and result.observations_bytes is None
    diagnostic = result.diagnostic
    assert diagnostic["schema"] == PITCH_SPACE_DIAGNOSTIC_SCHEMA and diagnostic["label"] == DIAGNOSTIC_LABEL
    assert diagnostic["calibration"] == {
        "state": "not_attempted", "reason": NO_ANNOTATIONS_REASON, "policy_version": "ai3-calibration-policy/v1",
        "summary": None, "keyframes": [], "shots": [], "suspected_cut_frames": [],
    }
    assert diagnostic["pitch"] is None and diagnostic["evidence_status"] == "no_annotations"
    assert diagnostic["frames"]["by_status"] == {"calibrated": 0, "uncalibrated": 41, "invalidated": 0}
    assert set(diagnostic["frame_states"]["reason"]) == {NO_ANNOTATIONS_REASON}
    assert diagnostic["observations"] == {
        "total": 123, "by_projection_status": {"frame_uncalibrated": 123}, "projected": 0,
        "projected_inside_pitch_lines": 0, "projected_extrapolated_beyond_landmarks": 0,
        "distinct_tracklets_projected": 0,
    }
    assert diagnostic["excluded_tracklets_by_class"] == {"sports ball": 1}
    assert diagnostic["identity"]["team_assignment"] == "not_available"


def test_calibrated_frames_project_true_positions_and_nothing_else():
    ai1, identities, result = stage(notes())
    calibration = result.calibration
    assert calibration is not None
    assert pitch_calibration_from_json_bytes(result.calibration_bytes).sha256() == calibration.sha256()
    assert calibration.source.tracklets_sha256 == ai1.tracklets_sha256
    assert calibration.source.video_sha256 == VIDEO_SHA and calibration.source.detections_sha256 == ai1.detections_sha256
    provenance_, table = read_projected_observations(result.observations_bytes)
    assert provenance_["source_calibration_sha256"] == calibration.sha256()
    assert provenance_["source_identities_sha256"] == identities.sha256
    assert provenance_["source_tracklets_sha256"] == ai1.tracklets_sha256 and provenance_["level"] == "tracklet"
    rows = table.to_pylist()
    assert len(rows) == 123  # 3 person tracklets x 41 frames; the ball is excluded
    calibrated = {item.frame_index for item in calibration.frames if item.status.value == "calibrated"}
    assert calibrated == set(range(10, 31))
    for row in rows:
        assert row["identity_status"] == "unresolved" and row["player_identity_id"] is None and row["team"] == "UNKNOWN"
        if row["frame_index"] in calibrated:
            assert row["projection_status"] == "projected"
            truth = PLAYERS[int(row["tracklet_id"][-6:]) - 1]
            assert math.dist((row["pitch_x_m"], row["pitch_y_m"]), truth) < 0.5
        else:
            assert row["projection_status"] == "frame_uncalibrated"
            assert row["pitch_x_m"] is None and row["pitch_y_m"] is None
    assert [(r["frame_index"], r["tracklet_id"]) for r in rows] == sorted((r["frame_index"], r["tracklet_id"]) for r in rows)
    diagnostic = result.diagnostic
    assert diagnostic["frames"]["by_status"] == {"calibrated": 21, "uncalibrated": 20, "invalidated": 0}
    assert diagnostic["observations"]["projected"] == 63 and diagnostic["observations"]["distinct_tracklets_projected"] == 3
    assert diagnostic["evidence"]["calibration_sha256"] == calibration.sha256()
    assert diagnostic["evidence"]["projected_observations_sha256"] == sha256_bytes(result.observations_bytes)
    assert diagnostic["evidence_status"] == "synthetic_annotations" and diagnostic["pitch"]["length_m"] == 105.0


def test_disagreeing_keyframes_invalidate_frames_and_project_nothing_there():
    _, _, result = stage(notes((10, 30), cameras={30: OTHER_CAMERA}))
    states = dict(zip(result.diagnostic["frame_states"]["frame_index"], result.diagnostic["frame_states"]["status"]))
    assert all(states[index] == "invalidated" for index in range(11, 30))
    _, table = read_projected_observations(result.observations_bytes)
    for row in table.to_pylist():
        if 11 <= row["frame_index"] <= 29:
            assert row["projection_status"] == "frame_invalidated" and row["pitch_x_m"] is None
    assert result.diagnostic["frames"]["by_reason"]["static_shot_keyframes_disagree"] == 19


def test_moving_camera_calibrates_only_the_annotated_keyframes():
    _, _, result = stage(notes(motion=CameraMotion.MOVING))
    assert result.diagnostic["frames"]["by_status"]["calibrated"] == 2
    assert result.diagnostic["frames"]["by_reason"]["camera_not_declared_static"] == 39


@pytest.mark.parametrize("kwargs", [dict(size=(3840, 2160)), dict(timestamp_offset=0.1)])
def test_incompatible_annotations_fail_with_a_coded_error(kwargs):
    with pytest.raises(SpatialStageError) as error:
        stage(notes(**kwargs))
    assert error.value.code == "pitch_annotations_incompatible"


def test_synthetic_annotations_never_calibrate_a_real_analysis_at_stage_time():
    det_bytes, trk_bytes = ai1_bytes()
    ai1 = verify_ai1_artifacts(det_bytes, trk_bytes, video_sha256=VIDEO_SHA)
    identities = run_identity_stage(ai1, trk_bytes, source_id="clip")
    pinned = load_pitch_annotations_parameter(
        pitch_annotations_parameter(annotations_to_json_bytes(notes()), video_sha256=VIDEO_SHA, data_origin="synthetic")
    )
    with pytest.raises(SpatialStageError) as error:
        run_pitch_calibration_stage(ai1, identities, run_id=RUN_ID, data_origin="real", annotations=pinned)
    assert error.value.code == "pitch_annotations_synthetic"


def test_manual_annotations_are_reported_as_manual_evidence():
    _, _, result = stage(notes(origin=AnnotationOrigin.MANUAL), data_origin="real")
    assert result.diagnostic["evidence_status"] == "manual_annotations"
    assert result.diagnostic["data_origin"] == "real"


def test_stage_outputs_are_deterministic():
    _, identities_a, a = stage(notes())
    _, identities_b, b = stage(notes())
    assert identities_a.content == identities_b.content
    assert a.calibration_bytes == b.calibration_bytes and a.observations_bytes == b.observations_bytes
    assert diagnostic_to_json_bytes(a.diagnostic) == diagnostic_to_json_bytes(b.diagnostic)


# --- interval-filtered view -----------------------------------------------------------------


def test_pitch_space_view_filters_by_time_and_reports_coverage():
    _, _, result = stage(notes())
    diagnostic = diagnostic_to_json_bytes(result.diagnostic)
    whole = pitch_space_view(diagnostic, result.observations_bytes)
    assert whole["interval"]["analysed_frames"] == 41 and whole["observations"]["projected"] == 63
    assert sum(map(sum, whole["occupancy"]["counts"])) == 63
    assert [s["status"] for s in whole["interval"]["segments"]] == ["uncalibrated", "calibrated", "uncalibrated"]
    # 1.0 s .. 3.0 s covers frames 5..15: 5 uncalibrated (5..9), 6 calibrated (10..15).
    part = pitch_space_view(diagnostic, result.observations_bytes, start_seconds=1.0, end_seconds=3.0)
    assert part["interval"]["analysed_frames"] == 11
    assert part["interval"]["frames_by_status"] == {"calibrated": 6, "uncalibrated": 5, "invalidated": 0}
    assert part["observations"]["total"] == 33 and part["observations"]["projected"] == 18
    assert all(1.0 <= p["timestamp_seconds"] <= 3.0 for p in part["points"])
    assert part["label"] == DIAGNOSTIC_LABEL and part["level"] == "tracklet"
    limited = pitch_space_view(diagnostic, result.observations_bytes, max_points=10)
    assert len(limited["points"]) == 10 and limited["points_truncated"] is True


def test_pitch_space_view_without_calibration_has_no_positions():
    _, _, result = stage(None, data_origin="real")
    view = pitch_space_view(diagnostic_to_json_bytes(result.diagnostic), None, start_seconds=0.0, end_seconds=2.0)
    assert view["occupancy"] is None and view["points"] == [] and view["pitch"] is None
    assert view["interval"]["frames_by_status"]["uncalibrated"] == 11
    assert view["observations"]["projected"] == 0 and view["observations"]["total"] is None


@pytest.mark.parametrize("kwargs", [
    dict(start_seconds=-1.0), dict(start_seconds=float("nan")), dict(start_seconds=5.0, end_seconds=1.0),
    dict(max_points=100000), dict(cell_size_m=0.1),
])
def test_pitch_space_view_rejects_invalid_requests(kwargs):
    _, _, result = stage(None, data_origin="real")
    with pytest.raises(SpatialStageError):
        pitch_space_view(diagnostic_to_json_bytes(result.diagnostic), None, **kwargs)


# --- worker + API integration ------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from fakes.ai1_spatial import VIDEO_BYTES  # noqa: E402
from footballai_v2.api import create_app  # noqa: E402
from footballai_v2.contracts.v1 import ArtifactCategory, StageStatus  # noqa: E402
from footballai_v2.execution.adapters.v1_compat_pipeline import V1CompatPipeline  # noqa: E402
from footballai_v2.execution.adapters.v1_compat_runtime import V1CompatConfig, V1CompatReadiness  # noqa: E402
from footballai_v2.execution.coordinator import AnalysisCoordinator, ExecutionSettings  # noqa: E402
from footballai_v2.execution.errors import ExecutionFailure  # noqa: E402
from footballai_v2.execution.executor import AnalysisExecutor  # noqa: E402
from footballai_v2.execution.generated_artifact import GeneratedArtifact  # noqa: E402


SPATIAL_IDS = {"player-identities", "pitch-calibration", "projected-player-observations", "pitch-space-diagnostic"}


def v1_payloads(det_bytes: bytes, trk_bytes: bytes) -> dict:
    common = {"run_id": "test", "warnings": ["tracklet != player"], "provenance": {}}
    return {
        "detections": GeneratedArtifact(
            "detections", "Cached detections", ArtifactCategory.OTHER, "artifacts/detections.parquet", det_bytes,
            "application/vnd.apache.parquet", "footballai.detections/v1",
        ),
        "tracklets": GeneratedArtifact(
            "tracklets", "Temporary tracklets", ArtifactCategory.TRACKS, "artifacts/tracklets.parquet", trk_bytes,
            "application/vnd.apache.parquet", "footballai.tracklets/v1",
        ),
        "team-summary": {**common, "schema": "footballai.team-summary/v1", "players": {}},
        "track-summary": {**common, "schema": "footballai.track-summary/v1", "players": {}},
        "track-detail": {**common, "schema": "footballai.track-detail/v1", "tracks": {}},
        "workload-advisory": {**common, "schema": "footballai.workload-advisory/v1", "tracks": {}},
        "analysis-diagnostics": {
            **common, "schema": "footballai.analysis-diagnostics/v2", "output_count": 0,
            "detection": {"frames_processed": 41, "detection_count": 164},
            "tracking": {"frames_processed": 41, "total_tracklets": 4, "tracklets_sha256": sha256_bytes(trk_bytes)},
            "stage_timings": {"video_decoding_detection_seconds": 1.0, "tracking_seconds": .5, "diagnostic_computation_seconds": .01},
        },
    }


@pytest.fixture
def worker(tmp_path, monkeypatch):
    settings = ExecutionSettings(tmp_path / "runs", tmp_path / "queue", max_upload_bytes=1024 * 1024, allow_test_profiles=True)
    monkeypatch.setattr(AnalysisCoordinator, "probe_video", lambda self, path, extension: {"duration_seconds": 8.2, "container": "mp4", "size_bytes": path.stat().st_size})
    model = tmp_path / "yolov8m.pt"
    model.write_bytes(b"model")
    ready = V1CompatReadiness("ready", (), (), {"device": "cpu"}, V1CompatConfig(5, 1280, .2, "cpu", "cpu", model, "a" * 64))
    monkeypatch.setattr("footballai_v2.execution.adapters.v1_compat_pipeline.check_v1_compat_readiness", lambda: ready)
    monkeypatch.setattr("footballai_v2.execution.coordinator.check_v1_compat_readiness", lambda: ready)
    calls = {"count": 0, "fail_first": False}
    det_bytes, trk_bytes = ai1_bytes()

    def build(*args, **kwargs):
        calls["count"] += 1
        if calls["fail_first"] and calls["count"] == 1:
            raise ExecutionFailure("v1_subprocess_failed", "V1-compatible processing stopped safely.")
        return v1_payloads(det_bytes, trk_bytes)

    monkeypatch.setattr(V1CompatPipeline, "build_artifacts", build)
    client = TestClient(create_app(settings.run_root, settings=settings, allowed_origins=("http://localhost:5173",)))
    return client, settings, calls


def upload(client, *, annotations: bytes | None = None, origin="synthetic", profile="v1_compat"):
    files = {"video": ("match.mp4", VIDEO_BYTES, "video/mp4")}
    if annotations is not None:
        files["pitch_annotations"] = ("annotations.json", annotations, "application/json")
    return client.post("/api/v1/analyses", files=files, data={"match_name": "Spatial test", "pipeline_profile": profile, "data_origin": origin})


def execute(client, settings):
    queue = AnalysisCoordinator(settings).queue
    job = queue.claim("test-worker")
    assert job
    status = AnalysisExecutor(client.app.state.run_store, client.app.state.object_storage, stage_delay_seconds=0).execute(job, "test-worker")
    queue.complete(job) if status.value in {"succeeded", "partial"} else queue.fail(job)
    return status


def test_worker_without_annotations_publishes_review_only_identities_and_explicit_uncalibrated_state(worker):
    client, settings, _ = worker
    created = upload(client, origin="real").json()
    assert execute(client, settings).value == "succeeded"
    run = client.app.state.run_store.load(created["run_id"])
    assert run.pipeline_version == "v1_compat/2.2.0"
    stages = {stage.stage_name.value: stage for stage in run.stages}
    assert stages["identity_resolution"].status is StageStatus.SUCCEEDED
    assert stages["identity_resolution"].produced_artifact_ids == ("player-identities",)
    assert "UNKNOWN" in stages["identity_resolution"].message
    assert stages["pitch_calibration"].status is StageStatus.SUCCEEDED
    assert "No pitch landmark annotations" in stages["pitch_calibration"].message
    assert "no pitch coordinates were produced" in stages["pitch_calibration"].message
    assert stages["pitch_calibration"].produced_artifact_ids == ("pitch-space-diagnostic",)
    assert stages["pitch_calibration"].performance_metrics["frames_processed"] == 41
    ids = {item.artifact_id for item in run.artifacts}
    assert {"player-identities", "pitch-space-diagnostic"} <= ids and not {"pitch-calibration", "projected-player-observations"} & ids
    for artifact_id in ("player-identities", "pitch-space-diagnostic"):
        assert client.app.state.object_storage.artifact_integrity(run.run_id, artifact_id)
    view = client.get(f"/api/v1/runs/{run.run_id}/pitch-space").json()
    assert view["calibration_state"] == "not_attempted" and view["calibration_reason"] == NO_ANNOTATIONS_REASON
    assert view["points"] == [] and view["occupancy"] is None and view["label"] == DIAGNOSTIC_LABEL
    assert view["interval"]["frames_by_status"] == {"calibrated": 0, "uncalibrated": 41, "invalidated": 0}


def test_worker_calibrates_projects_and_serves_interval_view_with_evidence(worker):
    client, settings, _ = worker
    created = upload(client, annotations=annotations_to_json_bytes(notes())).json()
    assert execute(client, settings).value == "succeeded"
    run = client.app.state.run_store.load(created["run_id"])
    stages = {stage.stage_name.value: stage for stage in run.stages}
    assert stages["pitch_calibration"].status is StageStatus.SUCCEEDED
    assert stages["pitch_calibration"].produced_artifact_ids == ("pitch-calibration", "projected-player-observations", "pitch-space-diagnostic")
    assert "21 calibrated, 20 uncalibrated, 0 invalidated of 41" in stages["pitch_calibration"].message
    assert stages["pitch_calibration"].performance_metrics["output_count"] == 63
    refs = {item.artifact_id: item for item in run.artifacts}
    assert SPATIAL_IDS <= set(refs)
    storage = client.app.state.object_storage
    calibration = pitch_calibration_from_json_bytes(storage.read_artifact_bytes(run.run_id, "pitch-calibration"))
    assert calibration.sha256() == refs["pitch-calibration"].sha256
    assert calibration.source.tracklets_sha256 == refs["tracklets"].sha256
    assert calibration.source.detections_sha256 == refs["detections"].sha256
    assert calibration.source.video_sha256 == run.input.sha256
    diagnostic = json.loads(storage.read_artifact_bytes(run.run_id, "pitch-space-diagnostic"))
    assert diagnostic["evidence"]["identities_sha256"] == refs["player-identities"].sha256
    assert diagnostic["evidence"]["projected_observations_sha256"] == refs["projected-player-observations"].sha256
    assert diagnostic["evidence"]["annotations_sha256"] == run.parameters["pitch_annotations"]["sha256"]
    response = client.get(f"/api/v1/runs/{run.run_id}/pitch-space", params={"start_seconds": 2.0, "end_seconds": 4.0})
    assert response.status_code == 200
    view = response.json()
    assert view["interval"]["frames_by_status"]["calibrated"] == 11
    assert view["observations"]["projected"] == 33 and len(view["points"]) == 33
    for point in view["points"]:
        assert 2.0 <= point["timestamp_seconds"] <= 4.0 and point["identity_status"] == "unresolved"
        assert math.dist((point["x_m"], point["y_m"]), PLAYERS[int(point["tracklet_id"][-6:]) - 1]) < 0.5
    assert {item["artifact_id"] for item in view["artifacts"]} == {"pitch-space-diagnostic", "projected-player-observations"}
    assert client.get(f"/api/v1/runs/{run.run_id}/pitch-space", params={"start_seconds": 4, "end_seconds": 1}).status_code == 422
    assert client.get(f"/api/v1/runs/{run.run_id}/pitch-space", params={"start_seconds": -1}).status_code == 422


def test_upload_refuses_unbound_foreign_synthetic_or_demo_annotations_before_creating_a_run(worker):
    client, _, _ = worker
    unbound = dataclasses.replace(notes(), source=dataclasses.replace(notes().source, video_sha256=None))
    cases = [
        (dict(annotations=annotations_to_json_bytes(unbound)), 422, "pitch_annotations_unbound"),
        (dict(annotations=annotations_to_json_bytes(notes(video_sha="f" * 64))), 422, "pitch_annotations_video_mismatch"),
        (dict(annotations=annotations_to_json_bytes(notes()), origin="real"), 422, "pitch_annotations_synthetic"),
        (dict(annotations=annotations_to_json_bytes(notes()), profile="demo_fast"), 422, "pitch_annotations_unsupported_profile"),
        (dict(annotations=b"{"), 422, "invalid_pitch_annotations"),
        (dict(annotations=b" " * (1024 * 1024 + 1)), 413, "pitch_annotations_too_large"),
    ]
    for kwargs, status, code in cases:
        response = upload(client, **kwargs)
        assert response.status_code == status and response.json()["detail"]["error_code"] == code, (kwargs, response.json())
    assert client.get("/api/v1/runs").json()["runs"] == []


def test_incompatible_annotations_fail_the_attempt_safely_without_publishing(worker):
    client, settings, _ = worker
    created = upload(client, annotations=annotations_to_json_bytes(notes(size=(3840, 2160)))).json()
    assert execute(client, settings).value == "failed"
    run = client.app.state.run_store.load(created["run_id"])
    assert run.failure.error_code == "pitch_annotations_incompatible" and run.failure.retryable is False and run.artifacts == ()
    stages = {stage.stage_name.value: stage for stage in run.stages}
    assert stages["pitch_calibration"].status is StageStatus.FAILED
    assert stages["identity_resolution"].status is StageStatus.SUCCEEDED


def test_retry_keeps_pinned_annotations_and_clone_can_attach_new_ones(worker):
    client, settings, calls = worker
    calls["fail_first"] = True
    first = upload(client, annotations=annotations_to_json_bytes(notes())).json()
    assert execute(client, settings).value == "failed"
    retry = client.post(f"/api/v1/runs/{first['run_id']}/retry").json()
    assert retry["attempt_number"] == 2
    assert execute(client, settings).value == "succeeded"
    store = client.app.state.run_store
    original, retried = store.load(first["run_id"]), store.load(retry["run_id"])
    assert retried.parameters["pitch_annotations"] == original.parameters["pitch_annotations"]
    assert {"pitch-calibration", "projected-player-observations"} <= {item.artifact_id for item in retried.artifacts}
    assert original.artifacts == ()  # the failed attempt stays immutable

    plain = upload(client, origin="synthetic").json()
    assert execute(client, settings).value == "succeeded"
    moving = annotations_to_json_bytes(notes(motion=CameraMotion.MOVING))
    clone = client.post(f"/api/v1/runs/{plain['run_id']}/clone", files={"pitch_annotations": ("a.json", moving, "application/json")})
    assert clone.status_code == 202
    assert execute(client, settings).value == "succeeded"
    cloned = store.load(clone.json()["run_id"])
    assert cloned.parameters["pitch_annotations"]["sha256"] == sha256_bytes(moving)
    assert "pitch_annotations" not in store.load(plain["run_id"]).parameters
    bad = client.post(
        f"/api/v1/runs/{plain['run_id']}/clone",
        files={"pitch_annotations": ("a.json", annotations_to_json_bytes(notes(video_sha="f" * 64)), "application/json")},
    )
    assert bad.status_code == 422 and bad.json()["detail"]["error_code"] == "pitch_annotations_video_mismatch"


def test_identical_attempts_publish_identical_spatial_bytes(worker):
    client, settings, _ = worker
    content = annotations_to_json_bytes(notes())
    a = upload(client, annotations=content).json(); execute(client, settings)
    b = upload(client, annotations=content).json(); execute(client, settings)
    store = client.app.state.run_store
    sha_a = {item.artifact_id: item.sha256 for item in store.load(a["run_id"]).artifacts if item.artifact_id in SPATIAL_IDS}
    sha_b = {item.artifact_id: item.sha256 for item in store.load(b["run_id"]).artifacts if item.artifact_id in SPATIAL_IDS}
    # The calibration depends only on the video, AI1 frames and annotations. Identities,
    # observations and the diagnostic name their run, so they differ between runs
    # (within one run they are deterministic: test_stage_outputs_are_deterministic).
    assert sha_a["pitch-calibration"] == sha_b["pitch-calibration"]
    assert set(sha_a) == set(sha_b) == SPATIAL_IDS


def test_redelivery_after_spatial_stages_rebuilds_and_publishes_them(worker):
    client, settings, calls = worker
    created = upload(client, annotations=annotations_to_json_bytes(notes())).json()
    store = client.app.state.run_store
    run = store.load(created["run_id"])
    started = run.start(stages=run.stages)
    done = tuple(
        dataclasses.replace(
            stage, status=StageStatus.SUCCEEDED, progress_percent=100, started_at=started.started_at,
            finished_at=started.started_at,
        ) if index < 6 else stage
        for index, stage in enumerate(started.stages)
    )
    store.save(started.with_stages(done))  # crash after pitch_calibration, before publication
    assert execute(client, settings).value == "succeeded"
    assert calls["count"] == 1
    assert SPATIAL_IDS <= {item.artifact_id for item in store.load(created["run_id"]).artifacts}


def test_runs_without_a_diagnostic_report_pitch_space_unavailable(worker):
    client, settings, _ = worker
    created = upload(client, profile="demo_fast", origin="evaluation").json()
    execute(client, settings)
    response = client.get(f"/api/v1/runs/{created['run_id']}/pitch-space")
    assert response.status_code == 404 and response.json()["detail"]["error_code"] == "pitch_space_unavailable"


def test_cli_template_binds_to_the_video_only_for_pipeline_attachment(tmp_path):
    from footballai_v2.execution.ai3.annotations import annotations_from_json_bytes
    from footballai_v2.execution.ai3.cli import main

    det_bytes, trk_bytes = ai1_bytes()
    (tmp_path / "det.parquet").write_bytes(det_bytes)
    (tmp_path / "trk.parquet").write_bytes(trk_bytes)
    pitch = ["--length-m", "105", "--width-m", "68", "--markings", "ifab-law-1", "--dimensions-source", "measured"]
    common = [*pitch, "--source-id", "clip", "--detections", str(tmp_path / "det.parquet"), "--tracklets",
              str(tmp_path / "trk.parquet"), "--shot", "s1:0:40:static:bottom_touchline", "--keyframe", "10"]
    assert main(["template", *common, "--bind", "video", "--output", str(tmp_path / "video.json")]) == 0
    video_bound = annotations_from_json_bytes((tmp_path / "video.json").read_bytes())
    assert video_bound.source.video_sha256 == VIDEO_SHA
    assert video_bound.source.detections_sha256 is None and video_bound.source.tracklets_sha256 is None
    assert (video_bound.source.frame_width, video_bound.source.frame_height) == (1920, 1080)
    assert video_bound.frames[0].timestamp_seconds == 2.0
    assert main(["template", *common, "--output", str(tmp_path / "ai1.json")]) == 0
    ai1_bound = (tmp_path / "ai1.json").read_bytes()
    with pytest.raises(SpatialStageError) as error:
        pitch_annotations_parameter(ai1_bound, video_sha256=VIDEO_SHA, data_origin="real")
    assert error.value.code == "pitch_annotations_bound_to_ai1"


def test_pitch_space_view_refuses_observations_the_diagnostic_does_not_name():
    _, _, a = stage(notes())
    _, _, b = stage(notes((10, 20)))
    with pytest.raises(ValueError, match="not the artifact the diagnostic names"):
        pitch_space_view(diagnostic_to_json_bytes(a.diagnostic), b.observations_bytes)


def test_pitch_space_endpoint_rejects_an_unexpected_observations_schema(worker, monkeypatch):
    client, settings, _ = worker
    created = upload(client, annotations=annotations_to_json_bytes(notes())).json()
    execute(client, settings)
    store = client.app.state.run_store
    run = store.load(created["run_id"])
    relabelled = tuple(
        dataclasses.replace(item, schema_version="footballai.other/v1")
        if item.artifact_id == "projected-player-observations" else item
        for item in run.artifacts
    )
    # Terminal manifests are immutable on disk, so present the relabelled record on read.
    original_load = store.load
    monkeypatch.setattr(store, "load", lambda run_id: dataclasses.replace(original_load(run_id), artifacts=relabelled))
    response = client.get(f"/api/v1/runs/{run.run_id}/pitch-space")
    assert response.status_code == 409 and response.json()["detail"]["error_code"] == "artifact_integrity_failed"
