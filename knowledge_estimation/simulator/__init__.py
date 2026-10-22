"""Native 2D simulation and transport-independent knowledge event adapter.

Importing this package does not create a window or import ROS/Gazebo.
"""

from .bridge import EstimatorAdapter, Observation, ObservationKind
from .engine import Simulator

__all__ = ["Simulator", "EstimatorAdapter", "Observation", "ObservationKind"]
