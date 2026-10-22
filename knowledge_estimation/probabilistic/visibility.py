"""Probabilistic viewing opportunity from the sender's known route and obstacles.

This is a geometry opportunity model, not a continuous camera or perception
simulation. Unknown hidden occlusion must never be supplied to this estimator.
Its result is conditional on the route: multiply by route certainty and storage
probability separately when recording pass-by evidence. Repeated visible poses
do not become independent observations or increase the returned probability.
"""

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin
from typing import Iterable


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians; zero points along positive x

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw"):
            _finite(getattr(self, name), name)


@dataclass(frozen=True)
class Rectangle:
    """Known opaque axis-aligned obstacle; touching its boundary is blocked."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        for name in ("min_x", "min_y", "max_x", "max_y"):
            _finite(getattr(self, name), name)
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("rectangle bounds must be ordered")


@dataclass(frozen=True)
class VisibilityConfig:
    max_range_m: float
    horizontal_fov_rad: float
    p_visible_given_geometry: float = 0.8

    def __post_init__(self) -> None:
        for name in ("max_range_m", "horizontal_fov_rad", "p_visible_given_geometry"):
            _finite(getattr(self, name), name)
        if self.max_range_m < 0:
            raise ValueError("max_range_m must be non-negative")
        if not 0 < self.horizontal_fov_rad <= 2 * pi:
            raise ValueError("horizontal_fov_rad must be in (0, 2*pi]")
        if not 0 <= self.p_visible_given_geometry <= 1:
            raise ValueError("p_visible_given_geometry must be in [0, 1]")


@dataclass(frozen=True)
class VisibilityEstimate:
    p_visible: float
    visible_sample_count: int
    total_sample_count: int


def _blocked(pose: Pose2D, target: tuple[float, float], box: Rectangle) -> bool:
    """Inclusive segment/AABB slab intersection, including degenerate boxes."""
    entry, leave = 0.0, 1.0
    for origin, end, lower, upper in (
        (pose.x, target[0], box.min_x, box.max_x),
        (pose.y, target[1], box.min_y, box.max_y),
    ):
        delta = end - origin
        if delta == 0:
            if origin < lower or origin > upper:
                return False
        else:
            first, last = sorted(((lower - origin) / delta, (upper - origin) / delta))
            entry, leave = max(entry, first), min(leave, last)
            if entry > leave:
                return False
    return True


def estimate_visibility(
    route: Iterable[Pose2D],
    target: tuple[float, float],
    config: VisibilityConfig,
    known_occluders: Iterable[Rectangle] = (),
) -> VisibilityEstimate:
    """Return one conditional probability for a route-level viewing opportunity.

    Range and FOV boundaries are included. A target at the pose has no heading
    restriction. Only sampled poses count; visibility between samples is unknown.
    Every provided obstacle must be known to the estimating sender.
    """
    target_x, target_y = target
    _finite(target_x, "target x")
    _finite(target_y, "target y")
    obstacles = tuple(known_occluders)
    if not all(isinstance(box, Rectangle) for box in obstacles):
        raise ValueError("known_occluders must contain Rectangle instances")
    visible, total = 0, 0
    for pose in route:
        if not isinstance(pose, Pose2D):
            raise ValueError("route must contain Pose2D instances")
        total += 1
        dx, dy = target_x - pose.x, target_y - pose.y
        distance = hypot(dx, dy)
        if distance > config.max_range_m:
            continue
        angle = atan2(dy, dx) - pose.yaw
        deviation = abs(atan2(sin(angle), cos(angle)))
        if distance > 0 and deviation > config.horizontal_fov_rad / 2 + 1e-12:
            continue
        if not any(_blocked(pose, (target_x, target_y), box) for box in obstacles):
            visible += 1
    return VisibilityEstimate(
        config.p_visible_given_geometry if visible else 0.0, visible, total
    )
