"""Compatibility imports. Data preparation now lives under learning/dataset/."""

from ..learning.dataset import FEATURE_NAMES, extract_features

__all__ = ["FEATURE_NAMES", "extract_features"]
