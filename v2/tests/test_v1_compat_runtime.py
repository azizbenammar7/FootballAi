"""Bounded readiness, device, model, adapter, and launcher tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from footballai_v2.contracts.v1 import AnalysisRun, CodeReference, DataOrigin, InputReference, StageExecution, StageName, StageStatus
from footballai_v2.execution.adapters.v1_compat_pipeline import V1CompatPipeline
from footballai_v2.execution.adapters.v1_compat_runtime import (
    DEFAULT_MODEL_PATH,
    DEPENDENCIES,
    V1CompatConfig,
    V1CompatConfigurationError,
    V1CompatReadiness,
    check_v1_compat_readiness,
    configured_model_path,
    resolve_device,
    sha256_file,
    validate_model_file,
)


class FakeAccelerator:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


def fake_torch(*, mps: bool = True, cuda: bool = False):
    return SimpleNamespace(
        __version__="2.test",
        backends=SimpleNamespace(mps=FakeAccelerator(mps)),
        cuda=FakeAccelerator(cuda),
    )


def model_file(path: Path) -> Path:
    path.write_bytes(b"PK\x03\x04" + b"\0" * (1024 * 1024))
    return path


def readiness(tmp_path: Path, *, missing: str | None = None, importer=None, which=None, **kwargs):
    modules = {name: (fake_torch() if name == "torch" else SimpleNamespace(__version__="1.test")) for _, name, _ in DEPENDENCIES}
    return check_v1_compat_readiness(
        model_path=model_file(tmp_path / "yolov8m.pt"),
        find_spec=lambda name: None if name == missing else object(),
        import_module=importer or (lambda name: modules[name]),
        which=which or (lambda name: f"/safe/{name}"),
        python_version=kwargs.pop("python_version", (3, 13)),
        platform_name=kwargs.pop("platform_name", "darwin"),
        machine=kwargs.pop("machine", "arm64"),
        environment=kwargs.pop("environment", {}),
        **kwargs,
    )


def test_all_dependencies_available_and_opencv_maps_to_cv2(tmp_path):
    result = readiness(tmp_path)
    assert result.ready and result.status == "ready"
    assert next(item for item in DEPENDENCIES if item[0] == "OpenCV")[1:] == ("cv2", "opencv-python-headless")
    assert result.runtime["device"] == "mps"


@pytest.mark.parametrize("module_name", [item[1] for item in DEPENDENCIES])
def test_each_dependency_missing_independently(tmp_path, module_name):
    result = readiness(tmp_path, missing=module_name)
    assert result.status == "missing_python_packages"
    assert result.missing_requirements


def test_system_tool_model_python_platform_and_runtime_failures_are_distinct(tmp_path):
    assert readiness(tmp_path, which=lambda name: None if name == "ffprobe" else "/safe/ffmpeg").status == "missing_system_tools"
    assert check_v1_compat_readiness(model_path=tmp_path / "missing.pt", python_version=(3, 13), platform_name="darwin", machine="arm64", find_spec=lambda name: object(), import_module=lambda name: fake_torch() if name == "torch" else SimpleNamespace(__version__="1"), which=lambda name: "/safe/tool").status == "missing_model_weights"
    assert readiness(tmp_path, python_version=(3, 14)).status == "unsupported_python_version"
    assert readiness(tmp_path, platform_name="win32").status == "unsupported_platform"
    result = readiness(tmp_path, importer=lambda name: (_ for _ in ()).throw(RuntimeError("private detail")) if name == "cv2" else (fake_torch() if name == "torch" else SimpleNamespace(__version__="1")))
    assert result.status == "runtime_import_error"
    assert result.runtime_errors == ("OpenCV could not be imported",)


def test_model_path_default_explicit_invalid_checksum_and_public_safety(tmp_path):
    assert configured_model_path({}) == DEFAULT_MODEL_PATH
    explicit = model_file(tmp_path / "yolov8m.pt")
    assert configured_model_path({"FOOTBALLAI_V1_COMPAT_MODEL_PATH": str(explicit)}) == explicit
    assert validate_model_file(explicit)[0] is True
    assert sha256_file(explicit) == hashlib.sha256(explicit.read_bytes()).hexdigest()
    invalid = tmp_path / "invalid.pt"; invalid.write_bytes(b"not a model")
    assert validate_model_file(invalid)[0] is False
    public = readiness(tmp_path).public_dict()
    assert str(tmp_path) not in json.dumps(public)


def test_device_resolution_auto_and_explicit_failures():
    assert resolve_device(fake_torch(mps=True), "auto") == "mps"
    assert resolve_device(fake_torch(mps=False), "auto") == "cpu"
    assert resolve_device(fake_torch(mps=False, cuda=True), "auto") == "cuda"
    with pytest.raises(V1CompatConfigurationError, match="MPS"):
        resolve_device(fake_torch(mps=False), "mps")
    with pytest.raises(V1CompatConfigurationError, match="CUDA"):
        resolve_device(fake_torch(cuda=False), "cuda")


def make_run(input_path: Path, config: V1CompatConfig) -> AnalysisRun:
    stages = tuple(StageExecution(name.value, name, True, StageStatus.QUEUED, 0, 1) for name in StageName)
    return AnalysisRun.new(
        data_origin=DataOrigin.SYNTHETIC,
        input=InputReference("run-input://source.mp4", "a" * 64, "video/mp4"),
        code=CodeReference("https://example.test/repo", "8" * 40),
        pipeline_version="v1_compat/1.0.0",
        parameters={"pipeline_profile": "v1_compat", "v1_compat": config.public_dict()},
        stages=stages,
    )


def test_adapter_isolates_outputs_propagates_model_and_handles_empty_detection(tmp_path, monkeypatch):
    model = model_file(tmp_path / "yolov8m.pt")
    config = V1CompatConfig(1, 320, .25, "auto", "mps", model, sha256_file(model))
    run_dir = tmp_path / "runs" / "run"
    input_path = run_dir / "input" / "source.mp4"
    input_path.parent.mkdir(parents=True); input_path.write_bytes(b"video")
    run = make_run(input_path, config)
    ready = V1CompatReadiness("ready", (), (), {"device": "mps"}, config)
    monkeypatch.setattr("footballai_v2.execution.adapters.v1_compat_pipeline.check_v1_compat_readiness", lambda: ready)
    monkeypatch.setattr("footballai_v2.execution.adapters.v1_compat_pipeline.configured_model_path", lambda: model)
    seen: dict = {}

    def fake_run(command, cwd, log, timeout, cancellation_requested, environment, selected_device):
        seen.update(command=command, cwd=cwd, environment=environment, device=selected_device)
        processed = cwd / "data" / "processed"
        (processed / "tracking_summary.json").write_text(json.dumps({"frames_processed": 2, "detection_rows": 0, "tracked_ids": 0, "max_track_observations": 0, "empty_after_v1_filters": True}))

    monkeypatch.setattr(V1CompatPipeline, "_run", staticmethod(fake_run))
    protected = Path(__file__).resolve().parents[2] / "data" / "processed"
    before = {path: path.stat().st_mtime_ns for path in protected.glob("*")}
    artifacts = V1CompatPipeline().build_artifacts(run, 2, input_path)
    after = {path: path.stat().st_mtime_ns for path in protected.glob("*")}
    assert before == after
    assert seen["command"][0] == __import__("sys").executable
    assert seen["command"][seen["command"].index("--model") + 1] == str(model)
    assert seen["environment"]["YOLO_OFFLINE"] == "true"
    assert seen["device"] == "mps"
    assert artifacts["analysis-diagnostics"]["output_count"] == 0
    assert artifacts["analysis-diagnostics"]["provenance"]["selected_device"] == "mps"
    assert all(str(run_dir) in str(path) for path in (seen["cwd"],))


def test_adapter_rejects_changed_model_without_execution(tmp_path, monkeypatch):
    model = model_file(tmp_path / "yolov8m.pt")
    config = V1CompatConfig(1, 320, .25, "cpu", "cpu", model, sha256_file(model))
    run = make_run(tmp_path / "unused", config)
    model.write_bytes(model.read_bytes() + b"changed")
    monkeypatch.setattr("footballai_v2.execution.adapters.v1_compat_pipeline.configured_model_path", lambda: model)
    from footballai_v2.execution.errors import ExecutionFailure
    with pytest.raises(ExecutionFailure, match="weights"):
        V1CompatPipeline._execution_config(run, config)


def test_subprocess_uses_argument_list_shell_false_and_supports_cancellation(tmp_path, monkeypatch):
    from footballai_v2.execution.errors import CancellationObserved
    seen = {}

    class Process:
        returncode = None
        def poll(self): return None
        def terminate(self): seen["terminated"] = True
        def wait(self, timeout=None): self.returncode = -15; return self.returncode
        def kill(self): seen["killed"] = True

    def popen(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return Process()

    monkeypatch.setattr("footballai_v2.execution.adapters.v1_compat_pipeline.subprocess.Popen", popen)
    with (tmp_path / "log").open("wb") as log, pytest.raises(CancellationObserved):
        V1CompatPipeline._run(["python", "-m", "safe.module"], tmp_path, log, 10, lambda: True, {"SAFE": "1"}, "cpu")
    assert seen["command"] == ["python", "-m", "safe.module"]
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["env"] == {"SAFE": "1"}
    assert seen["terminated"] is True and "killed" not in seen


def test_worker_tracking_source_has_no_model_or_package_auto_download():
    root = Path(__file__).resolve().parents[2]
    tracking = (root / "v2" / "src" / "footballai_v2" / "execution" / "adapters" / "v1_compat_tracking.py").read_text()
    adapter = (root / "v2" / "src" / "footballai_v2" / "execution" / "adapters" / "v1_compat_pipeline.py").read_text()
    assert "YOLO(str(model_path))" in tracking
    assert "YOLO(\"yolov8m.pt\")" not in tracking
    assert "pip install" not in tracking + adapter
    assert '"YOLO_OFFLINE": "true"' in adapter


def test_readiness_cli_returns_nonzero_incomplete_and_zero_when_mocked_complete(monkeypatch):
    from footballai_v2.cli import v1_compat_runtime as cli
    missing = V1CompatReadiness("missing_model_weights", ("weights",), (), {}, None)
    monkeypatch.setattr(cli, "check_v1_compat_readiness", lambda: missing)
    assert cli._check(json_output=True) == 2
    ready = V1CompatReadiness("ready", (), (), {"device": "cpu"}, None)
    monkeypatch.setattr(cli, "check_v1_compat_readiness", lambda: ready)
    assert cli._check(json_output=False) == 0


def test_demo_scripts_pin_one_interpreter_and_propagate_runtime_environment():
    root = Path(__file__).resolve().parents[2]
    demo = (root / "v2" / "dev" / "run_demo.sh").read_text()
    setup = (root / "v2" / "dev" / "setup_v1_compat.sh").read_text()
    makefile = (root / "Makefile").read_text()
    assert 'v2_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"' in demo
    assert demo.count('"$v2_python" -m') >= 3
    for name in ("FOOTBALLAI_V1_COMPAT_MODEL_PATH", "FOOTBALLAI_V1_COMPAT_DEVICE", "FOOTBALLAI_V2_RUN_ROOT", "FOOTBALLAI_V2_QUEUE_ROOT"):
        assert f"export {name}" in demo
    assert "--no-deps -r v2/requirements-v1-compat.txt" in setup
    assert "v2-demo:" in makefile and "v2-demo-v1-compat:" in makefile


def test_repository_tracks_no_model_weights_or_uploaded_videos():
    import subprocess
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run([str(root / "v2" / "dev" / "check_no_tracked_models.sh")], cwd=root, shell=False, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
