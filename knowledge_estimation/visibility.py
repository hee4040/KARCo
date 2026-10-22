"""Compatibility imports for probabilistic viewing opportunity estimation."""

from .probabilistic.visibility import (
    Pose2D, Rectangle, VisibilityConfig, VisibilityEstimate, estimate_visibility,
)

__all__ = ["Pose2D", "Rectangle", "VisibilityConfig", "VisibilityEstimate", "estimate_visibility"]
