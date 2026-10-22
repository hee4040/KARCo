"""Receiver knowledge estimation, independent of ROS and communication policies."""

from .probabilistic import KnowledgeEstimator
from .types import (
    BeliefSnapshot, EstimatorConfig, Evidence, Information, ItemBelief,
    KnowledgeItem, KnowledgeKey, UpdateRecord,
)
from .probabilistic.visibility import (
    Pose2D, Rectangle, VisibilityConfig, VisibilityEstimate, estimate_visibility,
)

__all__ = [
    "KnowledgeEstimator", "BeliefSnapshot", "EstimatorConfig", "Evidence",
    "Information", "ItemBelief", "KnowledgeItem", "KnowledgeKey", "UpdateRecord",
    "Pose2D", "Rectangle", "VisibilityConfig", "VisibilityEstimate", "estimate_visibility",
]
