"""Configurable football pitch geometry in metres (AI3).

Coordinate system (``footballai.pitch-plan/v1``)
------------------------------------------------
The pitch is described on its *plan view*: the standard top-down drawing with
the two goal lines on the left and right and the two touchlines at the top and
bottom. LEFT/RIGHT/TOP/BOTTOM always name sides of that drawing, never sides of
a video frame.

- origin: the corner where the LEFT goal line meets the TOP touchline
- x_m: along the touchlines, from the LEFT goal line (0) to the RIGHT goal
  line (``length_m``)
- y_m: along the goal lines, from the TOP touchline (0) to the BOTTOM
  touchline (``width_m``)
- units: metres

The plan uses image-like handedness (x right, y down). A camera above the pitch
sees the pitch's upper surface, so its image→pitch homography must preserve
orientation. A negative Jacobian therefore means the annotation is mirrored
(for example LEFT and RIGHT swapped), never a valid camera.

Pitch dimensions are always explicit. There is no default length or width:
real pitches range from small youth pitches to 120 m × 90 m, and a wrong size
silently distorts every distance. Marking dimensions (penalty area, circle...)
are also explicit; :data:`IFAB_LAW_1_MARKINGS` names the Laws of the Game
values so a caller can select them deliberately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


COORDINATE_SYSTEM = "footballai.pitch-plan/v1"
COORDINATE_SYSTEM_DEFINITION: Mapping[str, str] = MappingProxyType({
    "name": COORDINATE_SYSTEM,
    "units": "metre",
    "view": "plan view seen from above; LEFT/RIGHT/TOP/BOTTOM name sides of the plan drawing, not of the image",
    "origin": "corner where the LEFT goal line meets the TOP touchline",
    "x_axis": "along the touchlines, from the LEFT goal line (x=0) to the RIGHT goal line (x=length_m)",
    "y_axis": "along the goal lines, from the TOP touchline (y=0) to the BOTTOM touchline (y=width_m)",
    "handedness": "image-like (x right, y down): a camera above the pitch preserves orientation",
})
_TOLERANCE_M = 1e-6


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number of metres")
    return result


class CameraSide(str, Enum):
    """Where a camera stands relative to the plan view, per shot.

    ``unknown`` disables the camera-side consistency check; it never guesses.
    """

    BOTTOM_TOUCHLINE = "bottom_touchline"
    TOP_TOUCHLINE = "top_touchline"
    LEFT_GOAL_LINE = "left_goal_line"
    RIGHT_GOAL_LINE = "right_goal_line"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PitchMarkings:
    """Line-marking dimensions in metres (Law 1 names)."""

    penalty_area_depth_m: float
    penalty_area_width_m: float
    goal_area_depth_m: float
    goal_area_width_m: float
    penalty_mark_distance_m: float
    centre_circle_radius_m: float
    penalty_arc_radius_m: float
    goal_width_m: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not self.goal_width_m < self.goal_area_width_m < self.penalty_area_width_m:
            raise ValueError("markings must nest: goal < goal area < penalty area (width)")
        if not self.goal_area_depth_m < self.penalty_area_depth_m:
            raise ValueError("goal area must be shallower than the penalty area")
        if not self.penalty_mark_distance_m < self.penalty_area_depth_m:
            raise ValueError("the penalty mark must lie inside the penalty area")
        offset = self.penalty_area_depth_m - self.penalty_mark_distance_m
        if not self.penalty_arc_radius_m > offset:
            raise ValueError("the penalty arc must reach beyond the penalty area")
        if not self.penalty_arc_half_chord_m < self.penalty_area_width_m / 2:
            raise ValueError("the penalty arc must meet the penalty area's front line")

    @property
    def penalty_arc_half_chord_m(self) -> float:
        offset = self.penalty_area_depth_m - self.penalty_mark_distance_m
        return math.sqrt(max(0.0, self.penalty_arc_radius_m ** 2 - offset ** 2))

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


IFAB_LAW_1_MARKINGS = PitchMarkings(
    penalty_area_depth_m=16.5,
    penalty_area_width_m=40.32,
    goal_area_depth_m=5.5,
    goal_area_width_m=18.32,
    penalty_mark_distance_m=11.0,
    centre_circle_radius_m=9.15,
    penalty_arc_radius_m=9.15,
    goal_width_m=7.32,
)


@dataclass(frozen=True, slots=True)
class PitchGeometry:
    """An explicitly configured pitch. ``length_m`` must exceed ``width_m``."""

    length_m: float
    width_m: float
    markings: PitchMarkings
    dimensions_source: str
    _landmarks: Mapping[str, tuple[float, float]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_m", _positive(self.length_m, "length_m"))
        object.__setattr__(self, "width_m", _positive(self.width_m, "width_m"))
        if not isinstance(self.markings, PitchMarkings):
            raise ValueError("markings must be PitchMarkings")
        if not isinstance(self.dimensions_source, str) or not self.dimensions_source.strip():
            raise ValueError("dimensions_source must say where the pitch dimensions came from")
        if self.length_m <= self.width_m:
            raise ValueError("length_m must exceed width_m (goal lines are the short sides)")
        markings = self.markings
        if markings.penalty_area_width_m >= self.width_m:
            raise ValueError("penalty area is wider than the pitch")
        if 2 * markings.centre_circle_radius_m >= self.width_m:
            raise ValueError("centre circle does not fit across the pitch")
        if self.length_m / 2 - markings.centre_circle_radius_m <= markings.penalty_area_depth_m:
            raise ValueError("centre circle would overlap a penalty area")
        object.__setattr__(self, "_landmarks", _landmarks(self))

    @property
    def within_law_1_range(self) -> bool:
        """Law 1 limits for 11-a-side: length 90–120 m, width 45–90 m."""
        return 90 <= self.length_m <= 120 and 45 <= self.width_m <= 90

    def contains(self, x_m: float, y_m: float, *, margin_m: float = 0.0) -> bool:
        return -margin_m <= x_m <= self.length_m + margin_m and -margin_m <= y_m <= self.width_m + margin_m

    def landmarks(self) -> Mapping[str, tuple[float, float]]:
        return self._landmarks

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "markings": self.markings.to_dict(),
            "dimensions_source": self.dimensions_source,
            "within_law_1_range": self.within_law_1_range,
            "coordinate_system": dict(COORDINATE_SYSTEM_DEFINITION),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PitchGeometry":
        if not isinstance(value, Mapping):
            raise ValueError("pitch must be an object")
        expected = {"length_m", "width_m", "markings", "dimensions_source", "within_law_1_range", "coordinate_system"}
        missing = {"length_m", "width_m", "markings", "dimensions_source"} - set(value)
        if missing or set(value) - expected:
            raise ValueError("pitch has missing or unknown fields")
        markings = value["markings"]
        if not isinstance(markings, Mapping) or set(markings) != set(PitchMarkings.__dataclass_fields__):
            raise ValueError("pitch markings have missing or unknown fields")
        geometry = cls(value["length_m"], value["width_m"], PitchMarkings(**markings), value["dimensions_source"])
        if "coordinate_system" in value and value["coordinate_system"] != dict(COORDINATE_SYSTEM_DEFINITION):
            raise ValueError(f"unsupported coordinate system; expected {COORDINATE_SYSTEM}")
        if "within_law_1_range" in value and value["within_law_1_range"] != geometry.within_law_1_range:
            raise ValueError("within_law_1_range disagrees with the pitch dimensions")
        return geometry


LANDMARK_DESCRIPTIONS: Mapping[str, str] = MappingProxyType({
    "corner_top_left": "LEFT goal line meets TOP touchline",
    "corner_top_right": "RIGHT goal line meets TOP touchline",
    "corner_bottom_left": "LEFT goal line meets BOTTOM touchline",
    "corner_bottom_right": "RIGHT goal line meets BOTTOM touchline",
    "halfway_top": "halfway line meets TOP touchline",
    "halfway_bottom": "halfway line meets BOTTOM touchline",
    "centre_spot": "centre mark",
    "centre_circle_top": "halfway line meets the centre circle, TOP side",
    "centre_circle_bottom": "halfway line meets the centre circle, BOTTOM side",
    **{
        f"{area}_{side}_{corner}": (
            f"{side.upper()} {area.replace('_', ' ')}: {corner.replace('_', ' ')}"
        )
        for side in ("left", "right")
        for area in ("penalty_area", "goal_area")
        for corner in ("top_goal_line", "top_corner", "bottom_corner", "bottom_goal_line")
    },
    **{f"penalty_mark_{side}": f"{side.upper()} penalty mark" for side in ("left", "right")},
    **{
        f"penalty_arc_{side}_{end}": f"{side.upper()} penalty arc meets the penalty area front line, {end.upper()} side"
        for side in ("left", "right")
        for end in ("top", "bottom")
    },
})


def _landmarks(geometry: PitchGeometry) -> Mapping[str, tuple[float, float]]:
    length, width, markings = geometry.length_m, geometry.width_m, geometry.markings
    middle_x, middle_y = length / 2, width / 2
    points: dict[str, tuple[float, float]] = {
        "corner_top_left": (0.0, 0.0),
        "corner_top_right": (length, 0.0),
        "corner_bottom_left": (0.0, width),
        "corner_bottom_right": (length, width),
        "halfway_top": (middle_x, 0.0),
        "halfway_bottom": (middle_x, width),
        "centre_spot": (middle_x, middle_y),
        "centre_circle_top": (middle_x, middle_y - markings.centre_circle_radius_m),
        "centre_circle_bottom": (middle_x, middle_y + markings.centre_circle_radius_m),
    }
    for side, goal_x, inward in (("left", 0.0, 1.0), ("right", length, -1.0)):
        for area, depth, area_width in (
            ("penalty_area", markings.penalty_area_depth_m, markings.penalty_area_width_m),
            ("goal_area", markings.goal_area_depth_m, markings.goal_area_width_m),
        ):
            front_x = goal_x + inward * depth
            top_y, bottom_y = middle_y - area_width / 2, middle_y + area_width / 2
            points[f"{area}_{side}_top_goal_line"] = (goal_x, top_y)
            points[f"{area}_{side}_top_corner"] = (front_x, top_y)
            points[f"{area}_{side}_bottom_corner"] = (front_x, bottom_y)
            points[f"{area}_{side}_bottom_goal_line"] = (goal_x, bottom_y)
        points[f"penalty_mark_{side}"] = (goal_x + inward * markings.penalty_mark_distance_m, middle_y)
        front_x = goal_x + inward * markings.penalty_area_depth_m
        chord = markings.penalty_arc_half_chord_m
        points[f"penalty_arc_{side}_top"] = (front_x, middle_y - chord)
        points[f"penalty_arc_{side}_bottom"] = (front_x, middle_y + chord)
    if set(points) != set(LANDMARK_DESCRIPTIONS):
        raise AssertionError("landmark table and descriptions disagree")
    return MappingProxyType(dict(sorted(points.items())))


def landmark_matches_geometry(geometry: PitchGeometry, landmark_id: str, x_m: float, y_m: float) -> bool:
    expected = geometry.landmarks().get(landmark_id)
    if expected is None:
        return False
    return abs(expected[0] - x_m) <= _TOLERANCE_M and abs(expected[1] - y_m) <= _TOLERANCE_M


def pitch_plan_svg(geometry: PitchGeometry, *, scale: float = 8.0) -> str:
    """A labelled plan of every landmark, for annotators. No external resources."""
    margin = 6.0
    width_px = (geometry.length_m + 2 * margin) * scale
    height_px = (geometry.width_m + 2 * margin) * scale

    def p(x: float, y: float) -> tuple[float, float]:
        return round((x + margin) * scale, 2), round((y + margin) * scale, 2)

    m = geometry.markings
    lines: list[str] = []
    x0, y0 = p(0, 0)
    lines.append(
        f'<rect x="{x0}" y="{y0}" width="{geometry.length_m * scale:.2f}" '
        f'height="{geometry.width_m * scale:.2f}" fill="none" stroke="#2b6a2b" stroke-width="2"/>'
    )
    hx_top, hy_top = p(geometry.length_m / 2, 0)
    hx_bottom, hy_bottom = p(geometry.length_m / 2, geometry.width_m)
    lines.append(f'<line x1="{hx_top}" y1="{hy_top}" x2="{hx_bottom}" y2="{hy_bottom}" stroke="#2b6a2b" stroke-width="2"/>')
    cx, cy = p(geometry.length_m / 2, geometry.width_m / 2)
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{m.centre_circle_radius_m * scale:.2f}" fill="none" stroke="#2b6a2b" stroke-width="2"/>')
    for goal_x, inward in ((0.0, 1.0), (geometry.length_m, -1.0)):
        for depth, area_width in ((m.penalty_area_depth_m, m.penalty_area_width_m), (m.goal_area_depth_m, m.goal_area_width_m)):
            left = min(goal_x, goal_x + inward * depth)
            rx, ry = p(left, geometry.width_m / 2 - area_width / 2)
            lines.append(
                f'<rect x="{rx}" y="{ry}" width="{depth * scale:.2f}" height="{area_width * scale:.2f}" '
                'fill="none" stroke="#2b6a2b" stroke-width="2"/>'
            )
    for name, (x, y) in geometry.landmarks().items():
        px, py = p(x, y)
        lines.append(f'<circle cx="{px}" cy="{py}" r="4" fill="#c0392b"><title>{name}</title></circle>')
        lines.append(f'<text x="{px + 5}" y="{py - 5}" font-size="9" font-family="sans-serif">{name}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px:.0f}" height="{height_px:.0f}" '
        f'viewBox="0 0 {width_px:.2f} {height_px:.2f}">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
        + "\n".join(lines)
        + f'\n<text x="{x0}" y="{height_px - 8:.2f}" font-size="11" font-family="sans-serif">'
        f"{COORDINATE_SYSTEM}: origin top-left corner, x to the RIGHT goal line, y to the BOTTOM touchline (metres)</text>\n"
        "</svg>\n"
    )
