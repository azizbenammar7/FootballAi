"""Deterministic representative crops for manual identity review."""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from footballai_v2.execution.ai1.artifacts import TrackObservation, Tracklet, TrackletsArtifact
from footballai_v2.execution.ai2a.association import TrackletAppearance
from footballai_v2.execution.ai2a.model import (
    IdentityGroundTruthArtifact,
    SourceReference,
    TrackletId,
    tracklet_id,
)


REVIEW_SAMPLES_SCHEMA = "footballai.tracklet-review-samples/v1"


@dataclass(frozen=True, slots=True)
class SelectedObservation:
    label: str
    observation: TrackObservation


def select_representative_observations(
    tracklet: Tracklet,
    *,
    max_samples: int = 3,
) -> tuple[SelectedObservation, ...]:
    """Select early/middle/late observations deterministically and without duplicates."""
    if max_samples < 1 or max_samples > 3:
        raise ValueError("max_samples must be between one and three")
    size = len(tracklet.observations)
    candidates = [("early", 0), ("middle", (size - 1) // 2), ("late", size - 1)]
    selected: list[SelectedObservation] = []
    seen: set[int] = set()
    for label, index in candidates:
        if index in seen:
            continue
        seen.add(index)
        selected.append(SelectedObservation(label, tracklet.observations[index]))
        if len(selected) == max_samples:
            break
    if size == 2 and len(selected) == 2:
        selected[1] = SelectedObservation("late", selected[1].observation)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class ReviewSample:
    tracklet_id: TrackletId
    label: str
    frame_index: int
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.label not in {"early", "middle", "late"}:
            raise ValueError("review sample label is invalid")
        if self.frame_index < 0 or self.timestamp_seconds < 0:
            raise ValueError("review sample frame/time is invalid")
        if len(self.bbox_xyxy) != 4 or not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("review sample bbox is invalid")
        if Path(self.relative_path).is_absolute() or ".." in Path(self.relative_path).parts:
            raise ValueError("review sample path must be relative and bounded")
        if len(self.sha256) != 64 or any(value not in "0123456789abcdef" for value in self.sha256):
            raise ValueError("review sample sha256 is invalid")


@dataclass(frozen=True, slots=True)
class ReviewSamplesArtifact:
    schema: str
    source: SourceReference
    crop_strategy: str
    samples: tuple[ReviewSample, ...]

    def __post_init__(self) -> None:
        if self.schema != REVIEW_SAMPLES_SCHEMA:
            raise ValueError(f"schema must be {REVIEW_SAMPLES_SCHEMA}")
        if self.crop_strategy != "early_middle_late/v1":
            raise ValueError("unsupported crop strategy")
        if not isinstance(self.samples, tuple):
            raise ValueError("samples must be a tuple")
        keys = [(item.tracklet_id, item.label) for item in self.samples]
        if len(keys) != len(set(keys)):
            raise ValueError("review sample keys must be unique")


def review_samples_to_json_bytes(artifact: ReviewSamplesArtifact) -> bytes:
    value = {
        "schema": artifact.schema,
        "source": asdict(artifact.source),
        "crop_strategy": artifact.crop_strategy,
        "samples": [
            {
                **asdict(item),
                "tracklet_id": item.tracklet_id.value,
                "bbox_xyxy": list(item.bbox_xyxy),
            }
            for item in artifact.samples
        ],
    }
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def review_samples_from_json_bytes(content: bytes) -> ReviewSamplesArtifact:
    try:
        value = json.loads(content)
        return ReviewSamplesArtifact(
            schema=value["schema"],
            source=SourceReference(**value["source"]),
            crop_strategy=value["crop_strategy"],
            samples=tuple(
                ReviewSample(
                    tracklet_id=TrackletId(item["tracklet_id"]),
                    label=item["label"],
                    frame_index=item["frame_index"],
                    timestamp_seconds=item["timestamp_seconds"],
                    bbox_xyxy=tuple(item["bbox_xyxy"]),
                    relative_path=item["relative_path"],
                    sha256=item["sha256"],
                )
                for item in value["samples"]
            ),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid review-samples artifact") from exc


def build_review_samples(
    video_path: Path,
    tracklets: TrackletsArtifact,
    source: SourceReference,
    output_dir: Path,
) -> ReviewSamplesArtifact:
    """Extract no more than three deterministic JPEG crops per tracklet."""
    import cv2

    video_sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest()
    if video_sha256 != source.video_sha256:
        raise ValueError("source video checksum does not match evaluation source")
    crop_root = output_dir / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("source video cannot be opened")
    result: list[ReviewSample] = []
    try:
        for tracklet in sorted(tracklets.tracklets, key=lambda item: item.track_id):
            identifier = tracklet_id(tracklet.track_id)
            for selected in select_representative_observations(tracklet):
                item = selected.observation
                capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame_index)
                success, frame = capture.read()
                if not success:
                    raise ValueError(f"could not read source frame {item.frame_index}")
                height, width = frame.shape[:2]
                x1 = max(0, min(width - 1, math.floor(item.bbox.x1)))
                y1 = max(0, min(height - 1, math.floor(item.bbox.y1)))
                x2 = max(x1 + 1, min(width, math.ceil(item.bbox.x2)))
                y2 = max(y1 + 1, min(height, math.ceil(item.bbox.y2)))
                crop = frame[y1:y2, x1:x2]
                success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not success:
                    raise ValueError("could not encode review crop")
                content = encoded.tobytes()
                relative_path = Path("crops") / identifier.value / (
                    f"{selected.label}-f{item.frame_index:06d}.jpg"
                )
                destination = output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                result.append(
                    ReviewSample(
                        tracklet_id=identifier,
                        label=selected.label,
                        frame_index=item.frame_index,
                        timestamp_seconds=item.timestamp_seconds,
                        bbox_xyxy=(item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2),
                        relative_path=relative_path.as_posix(),
                        sha256=hashlib.sha256(content).hexdigest(),
                    )
                )
    finally:
        capture.release()
    return ReviewSamplesArtifact(
        REVIEW_SAMPLES_SCHEMA,
        source,
        "early_middle_late/v1",
        tuple(result),
    )


def appearances_from_review_samples(
    artifact: ReviewSamplesArtifact,
    artifact_root: Path,
) -> dict[TrackletId, TrackletAppearance]:
    """Summarize central upper-body crop HSV values as weak jersey evidence."""
    import cv2
    import numpy as np

    grouped: dict[TrackletId, list[tuple[float, float, float]]] = {}
    for sample in artifact.samples:
        path = artifact_root / sample.relative_path
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != sample.sha256:
            raise ValueError(f"review crop checksum mismatch: {sample.relative_path}")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"review crop cannot be decoded: {sample.relative_path}")
        height, width = image.shape[:2]
        torso = image[
            max(0, int(height * 0.12)) : max(1, int(height * 0.62)),
            max(0, int(width * 0.18)) : max(1, int(width * 0.82)),
        ]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        median = np.median(hsv, axis=0)
        grouped.setdefault(sample.tracklet_id, []).append(tuple(float(value) for value in median))
    return {
        identifier: TrackletAppearance(
            identifier,
            hue=float(sum(item[0] for item in values) / len(values)),
            saturation=float(sum(item[1] for item in values) / len(values)),
            value=float(sum(item[2] for item in values) / len(values)),
        )
        for identifier, values in sorted(grouped.items())
    }


def render_review_sheet(
    tracklets: TrackletsArtifact,
    samples: ReviewSamplesArtifact,
    annotations: IdentityGroundTruthArtifact,
) -> str:
    """Render a dependency-free local HTML sheet; assignments remain CLI-explicit."""
    sample_map: dict[TrackletId, list[ReviewSample]] = {}
    for sample in samples.samples:
        sample_map.setdefault(sample.tracklet_id, []).append(sample)
    annotation_map = {item.tracklet_id: item for item in annotations.annotations}
    rows = []
    for item in sorted(tracklets.tracklets, key=lambda value: value.track_id):
        identifier = tracklet_id(item.track_id)
        annotation = annotation_map.get(identifier)
        images = "".join(
            f'<figure><img src="{html.escape(sample.relative_path)}" loading="lazy">'
            f'<figcaption>{sample.label} · {sample.timestamp_seconds:.3f}s · f{sample.frame_index}</figcaption></figure>'
            for sample in sample_map.get(identifier, [])
        )
        identity = annotation.ground_truth_player_id.value if annotation and annotation.ground_truth_player_id else "—"
        team = annotation.team_label.value if annotation else "UNKNOWN"
        status = annotation.status.value if annotation else "unreviewed"
        rows.append(
            f"<tr><th>{identifier.value}</th><td>{item.start_time_seconds:.3f}–{item.end_time_seconds:.3f}s"
            f"<br>{item.observation_count} observations</td><td class=images>{images}</td>"
            f"<td>{html.escape(team)}<br>{html.escape(identity)}<br>{html.escape(status)}</td></tr>"
        )
    return """<!doctype html><meta charset=utf-8><title>FootballAI AI2A review</title>
<style>body{font:14px system-ui;margin:20px;background:#111;color:#eee}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #444;padding:8px;vertical-align:top}th{white-space:nowrap}.images{display:flex;gap:8px}
figure{margin:0}img{max-width:150px;max-height:190px;background:#222}figcaption{font-size:11px;color:#bbb}</style>
<h1>AI2A tracklet identity review</h1><p>Images are evidence only. Assign labels explicitly with the review CLI.</p>
<table><thead><tr><th>Tracklet</th><th>Extent</th><th>Representative crops</th><th>Annotation</th></tr></thead>
<tbody>""" + "".join(rows) + "</tbody></table>\n"


def render_contact_sheets(
    artifact: ReviewSamplesArtifact,
    artifact_root: Path,
    *,
    tracklets_per_sheet: int = 13,
) -> tuple[Path, ...]:
    """Create bounded PNG pages for rapid local visual review."""
    import cv2
    import numpy as np

    if tracklets_per_sheet < 1:
        raise ValueError("tracklets_per_sheet must be positive")
    grouped: dict[TrackletId, list[ReviewSample]] = {}
    for sample in artifact.samples:
        grouped.setdefault(sample.tracklet_id, []).append(sample)
    identifiers = sorted(grouped)
    paths: list[Path] = []
    row_height, label_width, image_width, sheet_width = 190, 150, 190, 740
    for page_index, start in enumerate(range(0, len(identifiers), tracklets_per_sheet), 1):
        page_ids = identifiers[start : start + tracklets_per_sheet]
        canvas = np.full((row_height * len(page_ids), sheet_width, 3), 245, dtype=np.uint8)
        for row_index, identifier in enumerate(page_ids):
            y = row_index * row_height
            cv2.putText(
                canvas,
                identifier.value,
                (8, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            for sample_index, sample in enumerate(grouped[identifier]):
                content = (artifact_root / sample.relative_path).read_bytes()
                if hashlib.sha256(content).hexdigest() != sample.sha256:
                    raise ValueError(f"review crop checksum mismatch: {sample.relative_path}")
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"review crop cannot be decoded: {sample.relative_path}")
                available_height = row_height - 35
                scale = min(image_width / image.shape[1], available_height / image.shape[0], 1.8)
                resized = cv2.resize(
                    image,
                    (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                )
                x = label_width + sample_index * image_width
                canvas[y + 30 : y + 30 + resized.shape[0], x : x + resized.shape[1]] = resized
                cv2.putText(
                    canvas,
                    f"{sample.label} {sample.timestamp_seconds:.1f}s",
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
            cv2.line(canvas, (0, y + row_height - 1), (sheet_width, y + row_height - 1), (170, 170, 170), 1)
        path = artifact_root / f"contact-sheet-{page_index:02d}.png"
        if not cv2.imwrite(str(path), canvas):
            raise ValueError(f"could not write {path}")
        paths.append(path)
    return tuple(paths)
