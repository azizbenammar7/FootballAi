"""Synthetic pinhole cameras with a known ground-truth homography.

For unit correctness and the offline benchmark only. The scenes are ideal:
planar pitch, no lens distortion, no motion blur, Gaussian click noise. A good
result here shows the mathematics and the gates behave; it says nothing about
accuracy on real broadcast footage.

World frame: x_m and y_m as in ``footballai.pitch-plan/v1``; z points *down*
(x right, y down, right-handed), so a camera above the pitch has negative z.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from footballai_v2.execution.ai3.annotations import (
    AnnotatedFrame,
    AnnotationQuality,
    CorrespondenceRole,
    LandmarkAnnotation,
    ReviewStatus,
)
from footballai_v2.execution.ai3.geometry import PitchGeometry


SYNTHETIC_ANNOTATOR = "synthetic-generator"


@dataclass(frozen=True, slots=True)
class SyntheticCamera:
    x_m: float
    y_m: float
    height_m: float
    target_x_m: float
    target_y_m: float
    focal_px: float
    width: int = 1920
    height: int = 1080
    _pitch_to_image: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rotation = self.rotation()
        centre = np.array([self.x_m, self.y_m, -self.height_m])
        translation = -rotation @ centre
        intrinsics = np.array([[self.focal_px, 0, self.width / 2], [0, self.focal_px, self.height / 2], [0, 0, 1.0]])
        matrix = intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
        matrix.setflags(write=False)
        object.__setattr__(self, "_pitch_to_image", matrix)

    def rotation(self) -> np.ndarray:
        centre = np.array([self.x_m, self.y_m, -self.height_m])
        forward = np.array([self.target_x_m, self.target_y_m, 0.0]) - centre
        forward /= np.linalg.norm(forward)
        down = np.array([0.0, 0.0, 1.0])
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        image_down = np.cross(forward, right)
        return np.vstack([right, image_down, forward])

    def pitch_to_image(self) -> np.ndarray:
        return self._pitch_to_image.copy()

    def image_to_pitch(self) -> np.ndarray:
        return np.linalg.inv(self.pitch_to_image())

    def project(self, x_m: float, y_m: float) -> tuple[float, float] | None:
        """Pixel of a ground point, or ``None`` if it is behind the camera."""
        u, v, w = self._pitch_to_image @ np.array([x_m, y_m, 1.0])
        if w <= 0:
            return None
        return float(u / w), float(v / w)

    def in_frame(self, pixel: tuple[float, float] | None, border_px: float = 0.0) -> bool:
        return pixel is not None and border_px <= pixel[0] <= self.width - border_px and border_px <= pixel[1] <= self.height - border_px


def broadcast_camera(geometry: PitchGeometry, rng: random.Random) -> SyntheticCamera:
    """A main-stand camera on the BOTTOM touchline side with random pan/zoom."""
    return SyntheticCamera(
        x_m=geometry.length_m / 2 + rng.uniform(-10, 10),
        y_m=geometry.width_m + rng.uniform(20, 45),
        height_m=rng.uniform(12, 28),
        target_x_m=rng.uniform(0.15, 0.85) * geometry.length_m,
        target_y_m=rng.uniform(0.3, 0.7) * geometry.width_m,
        focal_px=rng.uniform(900, 2600),
    )


def visible_landmarks(geometry: PitchGeometry, camera: SyntheticCamera, border_px: float = 8.0):
    visible = []
    for landmark_id, (x, y) in geometry.landmarks().items():
        pixel = camera.project(x, y)
        if camera.in_frame(pixel, border_px):
            visible.append((landmark_id, pixel, (x, y)))
    return visible


def synthetic_frame(
    geometry: PitchGeometry,
    camera: SyntheticCamera,
    *,
    frame_index: int,
    timestamp_seconds: float,
    shot_id: str,
    rng: random.Random,
    noise_px: float = 1.0,
    validation_every: int = 3,
    fit_outliers: int = 0,
    swap_left_right: bool = False,
) -> AnnotatedFrame:
    """Annotate every visible landmark with Gaussian click noise.

    Every ``validation_every``-th landmark (sorted by ID, offset 1) is a
    validation point. ``fit_outliers`` fit points are moved to a wrong pixel at
    least 60 px away. ``swap_left_right`` relabels LEFT↔RIGHT landmarks, the
    classic mirrored-annotation mistake.
    """
    visible = visible_landmarks(geometry, camera)
    landmarks = geometry.landmarks()
    entries = []
    for index, (landmark_id, (u, v), _) in enumerate(visible):
        role = CorrespondenceRole.VALIDATION if validation_every and index % validation_every == 1 else CorrespondenceRole.FIT
        x = min(max(u + rng.gauss(0, noise_px), 0.0), camera.width)
        y = min(max(v + rng.gauss(0, noise_px), 0.0), camera.height)
        entries.append([landmark_id, x, y, role])
    fit_positions = [i for i, entry in enumerate(entries) if entry[3] is CorrespondenceRole.FIT]
    for i in rng.sample(fit_positions, min(fit_outliers, len(fit_positions))):
        while True:
            x, y = rng.uniform(0, camera.width), rng.uniform(0, camera.height)
            if math.hypot(x - entries[i][1], y - entries[i][2]) >= 60:
                entries[i][1], entries[i][2] = x, y
                break
    annotations = []
    for landmark_id, x, y, role in entries:
        if swap_left_right:
            mirrored = (
                landmark_id.replace("left", "@").replace("right", "left").replace("@", "right")
            )
            if mirrored in landmarks:
                landmark_id = mirrored
        pitch_x, pitch_y = landmarks[landmark_id]
        annotations.append(LandmarkAnnotation(
            landmark_id, x, y, pitch_x, pitch_y, role, AnnotationQuality.CLEAR, ReviewStatus.ACCEPTED, SYNTHETIC_ANNOTATOR,
        ))
    unique = {}
    for item in annotations:
        unique.setdefault(item.landmark_id, item)
    return AnnotatedFrame(frame_index, timestamp_seconds, shot_id, tuple(unique.values()))


def field_error_m(
    estimated_image_to_pitch: np.ndarray,
    camera: SyntheticCamera,
    geometry: PitchGeometry,
    step_m: float = 2.0,
) -> np.ndarray:
    """True error of an estimate at a grid of pitch points visible in the frame."""
    xs = np.arange(0.0, geometry.length_m + 1e-9, step_m)
    ys = np.arange(0.0, geometry.width_m + 1e-9, step_m)
    grid = np.array([(x, y, 1.0) for x in xs for y in ys])
    image = grid @ camera.pitch_to_image().T
    in_front = image[:, 2] > 0
    pixels = image[:, :2] / np.where(in_front, image[:, 2], 1.0)[:, None]
    visible = (
        in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] <= camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= camera.height)
    )
    mapped = np.column_stack([pixels[visible], np.ones(int(visible.sum()))]) @ np.asarray(estimated_image_to_pitch).T
    w = mapped[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.hypot(mapped[:, 0] / w - grid[visible, 0], mapped[:, 1] / w - grid[visible, 1])
    error[~np.isfinite(error) | (w <= 0)] = np.inf
    return error
