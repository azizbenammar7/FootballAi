"""Numerical core: normalised DLT homography and deterministic RANSAC.

The transform maps **source image pixels → pitch metres**. Pixel coordinates
use the AI1 bounding-box convention: continuous, origin at the top-left corner
of the frame at native resolution, x right, y down.

Implementation notes
--------------------
- Direct Linear Transform with Hartley normalisation (Hartley & Zisserman,
  *Multiple View Geometry*, 2nd ed., alg. 4.2), solved by SVD in NumPy. NumPy
  is used rather than ``cv2.findHomography`` so the estimator runs in the test
  environment, and so sampling is explicitly deterministic.
- RANSAC enumerates every 4-point subset in a fixed order when there are at
  most ``max_subsets`` of them (always the case for hand-annotated frames);
  otherwise it draws subsets from ``random.Random(seed)``, whose stream is
  stable across Python and NumPy versions.
- A minimal sample with three nearly collinear points, in either space, is
  skipped: it cannot determine a homography.
- Every returned matrix is scaled to unit Frobenius norm with a positive
  denominator at the fit-point centroid, then rounded to 12 significant
  digits. All metrics are computed from the rounded matrix, so a serialised
  artifact can be re-verified exactly.
- Errors are geometric distances in the pitch plane, in metres.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


Matrix = tuple[float, float, float, float, float, float, float, float, float]
_SIGNIFICANT_DIGITS = 12


def as_matrix(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (9,):
        array = array.reshape(3, 3)
    if array.shape != (3, 3):
        raise ValueError("a homography is a 3×3 matrix")
    return array


def round_matrix(matrix: np.ndarray) -> Matrix:
    return tuple(float(f"{value:.{_SIGNIFICANT_DIGITS - 1}e}") for value in np.asarray(matrix, dtype=float).ravel())  # type: ignore[return-value]


def _points(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must be an N×2 array")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite")
    return array


def normalisation_transform(points: np.ndarray) -> np.ndarray:
    """Similarity moving the centroid to 0 and the mean distance to √2."""
    centroid = points.mean(axis=0)
    distance = np.sqrt(((points - centroid) ** 2).sum(axis=1)).mean()
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("points are coincident")
    scale = math.sqrt(2) / distance
    return np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]], dtype=float)


def _homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((points.shape[0], 1))])


@dataclass(frozen=True, slots=True)
class DltSolution:
    matrix: np.ndarray
    singular_ratio: float
    """Second-smallest / largest singular value of the normalised system.

    Near zero means the null space is not one-dimensional: the points do not
    determine a unique homography (degenerate or ill-conditioned layout).
    """


def dlt(source: np.ndarray, target: np.ndarray) -> DltSolution:
    source, target = _points(source), _points(target)
    if source.shape != target.shape or source.shape[0] < 4:
        raise ValueError("DLT needs at least four paired points")
    source_t, target_t = normalisation_transform(source), normalisation_transform(target)
    src = (_homogeneous(source) @ source_t.T)
    dst = (_homogeneous(target) @ target_t.T)
    rows = []
    for (x, y, _), (u, v, _) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    system = np.asarray(rows, dtype=float)
    _, singular, vt = np.linalg.svd(system)
    padded = np.zeros(9)
    padded[: singular.shape[0]] = singular
    ratio = float(padded[7] / padded[0]) if padded[0] > 0 else 0.0
    normalised = vt[-1].reshape(3, 3)
    matrix = np.linalg.inv(target_t) @ normalised @ source_t
    return DltSolution(matrix=matrix, singular_ratio=ratio)


def canonical(matrix: np.ndarray, reference_pixel: np.ndarray) -> np.ndarray | None:
    """Unit Frobenius norm, positive denominator at ``reference_pixel``."""
    if not np.all(np.isfinite(matrix)):
        return None
    norm = np.linalg.norm(matrix)
    if not math.isfinite(norm) or norm <= 0:
        return None
    result = matrix / norm
    w = result[2] @ np.array([reference_pixel[0], reference_pixel[1], 1.0])
    if not math.isfinite(w) or w == 0:
        return None
    return result if w > 0 else -result


def apply(matrix: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map pixels to the pitch. Returns (points, denominators w)."""
    mapped = _homogeneous(_points(pixels)) @ np.asarray(matrix, dtype=float).T
    w = mapped[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        points = mapped[:, :2] / w[:, None]
    return points, w


def jacobian_determinant(matrix: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """det ∂(x_m, y_m)/∂(px, py) = det(H) / w³ (orientation and local area scale)."""
    _, w = apply(matrix, pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.linalg.det(matrix) / w ** 3


def derivative_along_image_down(matrix: np.ndarray, pixel: Sequence[float]) -> tuple[float, float]:
    """∂(x_m, y_m)/∂py: where moving down the image goes on the pitch."""
    h = np.asarray(matrix, dtype=float)
    vector = np.array([pixel[0], pixel[1], 1.0])
    w = float(h[2] @ vector)
    x = float(h[0] @ vector) / w
    y = float(h[1] @ vector) / w
    return (h[0, 1] - x * h[2, 1]) / w, (h[1, 1] - y * h[2, 1]) / w


def triangle_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Twice the triangle area over its longest edge squared (0 = collinear)."""
    edges = (np.sum((a - b) ** 2), np.sum((b - c) ** 2), np.sum((c - a) ** 2))
    longest = max(edges)
    if longest <= 0:
        return 0.0
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return float(abs(cross) / longest)


class TriangleTable:
    """Collinearity of every point triple in both spaces, computed once per frame."""

    def __init__(self, source: np.ndarray, target: np.ndarray, min_ratio: float) -> None:
        self._bad: set[tuple[int, int, int]] = set()
        for triple in itertools.combinations(range(source.shape[0]), 3):
            i, j, k = triple
            if (
                triangle_ratio(source[i], source[j], source[k]) < min_ratio
                or triangle_ratio(target[i], target[j], target[k]) < min_ratio
            ):
                self._bad.add(triple)

    def degenerate(self, subset: Sequence[int]) -> bool:
        return any(triple in self._bad for triple in itertools.combinations(sorted(subset), 3))


def sample_is_degenerate(source: np.ndarray, target: np.ndarray, min_ratio: float) -> bool:
    return TriangleTable(source, target, min_ratio).degenerate(range(source.shape[0]))


def has_general_position_subset(source: np.ndarray, target: np.ndarray, min_ratio: float) -> bool:
    """Whether any 4 points are in general position in both spaces.

    Exhaustive and early-exiting; a frame has at most one point per landmark
    (31), so the search is bounded.
    """
    table = TriangleTable(source, target, min_ratio)
    return any(not table.degenerate(subset) for subset in itertools.combinations(range(source.shape[0]), 4))


def errors_m(matrix: np.ndarray, pixels: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    mapped, w = apply(matrix, pixels)
    with np.errstate(invalid="ignore"):
        distance = np.sqrt(((mapped - pitch) ** 2).sum(axis=1))
    distance[~np.isfinite(distance) | (w <= 0)] = np.inf
    return distance


@dataclass(frozen=True, slots=True)
class RansacResult:
    matrix: np.ndarray
    inliers: np.ndarray
    errors_m: np.ndarray
    subsets_evaluated: int
    subsets_degenerate: int
    exhaustive: bool
    singular_ratio: float


def _subsets(n: int, max_subsets: int, seed: int) -> tuple[list[tuple[int, ...]], bool]:
    total = math.comb(n, 4)
    if total <= max_subsets:
        return list(itertools.combinations(range(n), 4)), True
    rng = random.Random(seed)
    return [tuple(sorted(rng.sample(range(n), 4))) for _ in range(max_subsets)], False


def _fit(pixels: np.ndarray, pitch: np.ndarray) -> tuple[np.ndarray, float] | None:
    try:
        solution = dlt(pixels, pitch)
    except (ValueError, np.linalg.LinAlgError):
        return None
    matrix = canonical(solution.matrix, pixels.mean(axis=0))
    if matrix is None:
        return None
    return as_matrix(round_matrix(matrix)), solution.singular_ratio


def ransac(
    pixels: np.ndarray,
    pitch: np.ndarray,
    *,
    threshold_m: float,
    max_subsets: int,
    seed: int,
    min_triangle_ratio: float,
    max_refinements: int,
) -> RansacResult | None:
    """Best consensus homography, or ``None`` when no valid minimal sample exists.

    Scoring: most inliers, then lowest truncated squared error (MSAC), then the
    earliest subset — so ties resolve identically on every run.
    """
    pixels, pitch = _points(pixels), _points(pitch)
    if pixels.shape != pitch.shape or pixels.shape[0] < 4:
        return None
    subsets, exhaustive = _subsets(pixels.shape[0], max_subsets, seed)
    table = TriangleTable(pixels, pitch, min_triangle_ratio)
    best: tuple[int, float] | None = None
    best_inliers: np.ndarray | None = None
    evaluated = degenerate = 0
    for subset in subsets:
        index = list(subset)
        if table.degenerate(subset):
            degenerate += 1
            continue
        fitted = _fit(pixels[index], pitch[index])
        if fitted is None:
            degenerate += 1
            continue
        matrix, _ = fitted
        _, w = apply(matrix, pixels[index])
        if np.any(w <= 0):
            degenerate += 1
            continue
        evaluated += 1
        error = errors_m(matrix, pixels, pitch)
        inliers = error <= threshold_m
        score = (int(inliers.sum()), -float(np.minimum(error, threshold_m) ** 2 @ np.ones_like(error)))
        if best is None or score > best:
            best, best_inliers = score, inliers
    if best_inliers is None or best_inliers.sum() < 4:
        return None
    inliers = best_inliers
    matrix = None
    singular_ratio = 0.0
    for _ in range(max(1, max_refinements)):
        fitted = _fit(pixels[inliers], pitch[inliers])
        if fitted is None:
            return None
        matrix, singular_ratio = fitted
        error = errors_m(matrix, pixels, pitch)
        updated = error <= threshold_m
        if np.array_equal(updated, inliers) or updated.sum() < 4:
            break
        inliers = updated
    assert matrix is not None
    error = errors_m(matrix, pixels, pitch)
    return RansacResult(
        matrix=matrix,
        inliers=error <= threshold_m,
        errors_m=error,
        subsets_evaluated=evaluated,
        subsets_degenerate=degenerate,
        exhaustive=exhaustive,
        singular_ratio=singular_ratio,
    )


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull, counter-clockwise in a y-up sense; deterministic."""
    unique = sorted({(float(x), float(y)) for x, y in _points(points)})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float).reshape(-1, 2)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def inside_convex_polygon(polygon: np.ndarray, point: Sequence[float]) -> bool:
    if polygon.shape[0] < 3:
        return False
    sign = 0.0
    for index in range(polygon.shape[0]):
        a, b = polygon[index], polygon[(index + 1) % polygon.shape[0]]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) <= 1e-12:
            continue
        if sign == 0.0:
            sign = cross
        elif (cross > 0) != (sign > 0):
            return False
    return True
