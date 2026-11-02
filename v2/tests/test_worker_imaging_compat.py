"""Worker-image compatibility for Pillow 13 and filelock 4.

Normal CI never installs the v1-compat lock, so this module skips there. Run it
inside the built worker image. Everything is synthetic and CPU-only: YOLO is
built from its bundled YAML with random weights, so nothing is downloaded.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
functional = pytest.importorskip("torchvision.transforms.functional")
ultralytics = pytest.importorskip("ultralytics")
filelock = pytest.importorskip("filelock")
pytest.importorskip("matplotlib")
pytest.importorskip("pyarrow")

from footballai_v2.execution.ai1.artifacts import (  # noqa: E402
    TRACKLETS_SCHEMA,
    BoundingBox,
    TrackObservation,
    TrackerProvenance,
    Tracklet,
    TrackletsArtifact,
    detections_from_parquet_bytes,
    detections_to_parquet_bytes,
    has_legacy_frame_labels,
)
from footballai_v2.execution.ai1.detector import YoloDetector  # noqa: E402
from footballai_v2.execution.ai2a.model import SourceReference  # noqa: E402
from footballai_v2.execution.ai2a.review import (  # noqa: E402
    build_review_samples,
    render_contact_sheets,
)

WIDTH, HEIGHT, FRAMES, FPS = 160, 120, 12, 10


def frame(index: int = 0) -> np.ndarray:
    """A green pitch with one red-shirted player block that moves right."""
    pixels = np.full((HEIGHT, WIDTH, 3), (34, 139, 34), dtype=np.uint8)
    x = 20 + index * 4
    pixels[30:90, x : x + 24] = (200, 30, 30)
    return pixels


def write_video(path: Path, frames: int = FRAMES) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(frame(index)[:, :, ::-1])
    writer.release()
    return path


@pytest.fixture(scope="module")
def model():
    return ultralytics.YOLO("yolov8n.yaml")


def test_runtime_carries_the_patched_pillow():
    from PIL import __version__

    assert int(__version__.split(".")[0]) >= 13


def test_pil_load_convert_crop_and_resize_preserve_pixels():
    encoded = io.BytesIO()
    Image.fromarray(frame()).save(encoded, format="PNG")
    encoded.seek(0)
    loaded = Image.open(encoded)
    loaded.load()

    assert loaded.mode == "RGB" and loaded.size == (WIDTH, HEIGHT)
    assert np.array_equal(np.asarray(loaded), frame())
    assert loaded.convert("L").mode == "L"
    assert loaded.convert("RGBA").convert("RGB").tobytes() == loaded.tobytes()

    crop = loaded.crop((20, 30, 44, 90))
    assert crop.size == (24, 60)
    assert crop.getpixel((12, 30)) == (200, 30, 30)
    assert crop.resize((48, 120), Image.Resampling.NEAREST).getpixel((24, 60)) == (200, 30, 30)


def test_jpeg_player_crop_round_trips_between_opencv_and_pillow():
    crop = frame()[30:90, 20:44]
    success, encoded = cv2.imencode(".jpg", crop[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert success

    decoded = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")

    assert decoded.size == (24, 60)
    red, green, blue = decoded.getpixel((12, 30))
    assert red > 150 and green < 80 and blue < 80


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(frame(), id="rgb-uint8"),
        pytest.param(frame()[:, :, 0], id="l-uint8"),
    ],
)
def test_torchvision_pil_tensor_round_trip(array):
    # to_pil_image passes fromarray(mode=...), the argument Pillow 13 restricted.
    image = functional.to_pil_image(array)
    restored = functional.to_pil_image(functional.pil_to_tensor(image))

    assert np.array_equal(np.asarray(restored), array)
    assert functional.resize(image, [60, 80]).size == (80, 60)
    assert functional.to_tensor(image).max() <= 1


def test_torchvision_float_tensor_converts_to_rgb():
    image = functional.to_pil_image(torch.ones((3, 8, 8)) * 0.5)

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (127, 127, 127)


def test_yolo_preprocessing_accepts_pil_and_numpy_and_annotates(model, tmp_path, monkeypatch):
    from matplotlib import font_manager

    # The PIL annotator otherwise downloads Arial from ultralytics.com.
    bundled_font = font_manager.findfont("DejaVu Sans")
    monkeypatch.setattr("ultralytics.utils.plotting.check_font", lambda *_: bundled_font)
    results = model.predict(
        source=[Image.fromarray(frame()), frame()[:, :, ::-1]],
        imgsz=64, conf=0.001, device="cpu", verbose=False,
    )

    assert len(results) == 2
    for result in results:
        assert result.orig_shape == (HEIGHT, WIDTH)
        assert result.plot(pil=True, font_size=8).size == (WIDTH, HEIGHT)
        assert result.plot().shape == (HEIGHT, WIDTH, 3)
    path = tmp_path / "annotated.jpg"
    results[0].save(str(path))
    assert Image.open(path).size == (WIDTH, HEIGHT)


def test_matplotlib_png_output_reopens_with_pillow(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2, 2), dpi=50)
    FigureCanvasAgg(figure)
    figure.add_subplot().imshow(frame())
    png, jpeg = tmp_path / "plot.png", tmp_path / "plot.jpg"
    figure.savefig(png)
    figure.savefig(jpeg)

    assert Image.open(png).size == (100, 100)
    assert Image.open(jpeg).convert("RGB").size == (100, 100)


def test_ai1_detector_writes_a_valid_artifact_from_a_real_yolo_call(model, tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"unused: the factory supplies the model")

    artifact = YoloDetector(
        model_path=weights, model_sha256="a" * 64, model_version="random-weights",
        confidence=0.25, image_size=64, target_fps=5, device="cpu",
        source_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        code_revision="b" * 40, pipeline_version="test", model_factory=lambda _: model,
    ).detect(video)

    assert [item.frame_index for item in artifact.sampled_frames] == [1, 3, 5, 7, 9, 11]
    assert artifact.provenance.source_width == WIDTH
    assert detections_from_parquet_bytes(detections_to_parquet_bytes(artifact)) == artifact


def block_left_edge(pixels: np.ndarray) -> int:
    red = (pixels[..., 2] > 150) & (pixels[..., 1] < 90)
    return int(np.flatnonzero(red.any(axis=0))[0])


@pytest.mark.parametrize(
    ("stride", "expected_frames"),
    [
        pytest.param(1, list(range(13)), id="stride-1"),
        pytest.param(2, [1, 3, 5, 7, 9, 11], id="stride-2-drops-trailing-frame"),
        pytest.param(5, [4, 9], id="stride-5-drops-trailing-frames"),
        pytest.param(6, [5, 11], id="stride-6-drops-trailing-frame"),
    ],
)
def test_ai1_labels_name_the_frames_the_real_loader_gave_yolo(model, tmp_path, stride, expected_frames):
    video = write_video(tmp_path / "clip.mp4", frames=13)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"unused: the factory supplies the model")
    analysed = []

    class RecordingModel:
        def predict(self, **arguments):
            for result in model.predict(**arguments):
                analysed.append(result.orig_img)
                yield result

    artifact = YoloDetector(
        model_path=weights, model_sha256="a" * 64, model_version="random-weights",
        confidence=0.25, image_size=64, target_fps=FPS / stride, device="cpu",
        source_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        code_revision="b" * 40, pipeline_version="test",
        model_factory=lambda _: RecordingModel(),
    ).detect(video)

    labels = [item.frame_index for item in artifact.sampled_frames]
    assert labels == expected_frames
    assert [item.timestamp_seconds for item in artifact.sampled_frames] == pytest.approx(
        [label / FPS for label in labels]
    )
    assert [block_left_edge(pixels) for pixels in analysed] == pytest.approx(
        [20 + label * 4 for label in labels], abs=1
    )
    assert not has_legacy_frame_labels(artifact)

    # AI2A review seeks to the label, so a seek must land on the analysed frame.
    capture = cv2.VideoCapture(str(video))
    for label, pixels in zip(labels, analysed):
        capture.set(cv2.CAP_PROP_POS_FRAMES, label)
        success, sought = capture.read()
        assert success and block_left_edge(sought) == block_left_edge(pixels)
    capture.release()


def test_ai2a_review_crops_and_contact_sheet_open_with_pillow(tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    observations = tuple(
        TrackObservation(
            frame_index=index,
            timestamp_seconds=index / FPS,
            bbox=BoundingBox(20 + index * 4, 30, 44 + index * 4, 90),
            confidence=0.9,
        )
        for index in range(FRAMES)
    )
    tracklets = TrackletsArtifact(
        TRACKLETS_SCHEMA,
        TrackerProvenance(
            tracker_name="ByteTrack",
            tracker_version="test",
            configuration={"track_buffer": 30},
            detections_artifact_id="detections",
            detections_sha256="d" * 64,
            source_run_id="bounded-test",
        ),
        (Tracklet(track_id=1, class_id=0, semantic_class="person", observations=observations),),
    )
    source = SourceReference(
        source_id="bounded-clip",
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        duration_seconds=FRAMES / FPS,
        detections_sha256="b" * 64,
        tracklets_sha256="c" * 64,
    )
    root = tmp_path / "review"

    artifact = build_review_samples(video, tracklets, source, root)
    sheets = render_contact_sheets(artifact, root)

    assert [sample.label for sample in artifact.samples] == ["early", "middle", "late"]
    for sample in artifact.samples:
        crop = Image.open(root / sample.relative_path).convert("RGB")
        assert crop.size == (24, 60)
    assert Image.open(sheets[0]).size == (740, 190)


def test_filelock_4_serves_the_ultralytics_and_torch_call_sites(tmp_path):
    from ultralytics.data import converter  # noqa: F401
    from torch.utils._filelock import FileLock as TorchFileLock

    async def hold() -> bool:
        async with filelock.AsyncFileLock(tmp_path / "async.lock", timeout=0) as lock:
            return lock.is_locked

    assert asyncio.run(hold())
    with TorchFileLock(str(tmp_path / "torch.lock")):
        with pytest.raises(filelock.Timeout):
            filelock.FileLock(tmp_path / "torch.lock", timeout=0).acquire()
