"""Input/output schemas shared with belief, PPO, and robot adapters."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


MESSAGE_ORDER = ("E", "N", "C", "I")


class Action(str, Enum):
    SEND = "SEND"
    NO_COMM = "NO_COMM"
    REOBSERVE = "REOBSERVE"


@dataclass(frozen=True)
class Event:
    event_id: str
    spot_id: str = ""
    object_id: str = ""
    anomaly_type: str = "misplaced"
    target_confirmed: bool = True
    has_similar_objects: bool = False
    similar_object_count: int = 0
    object_distance_cm: float | None = None
    position_uncertainty_cm: float | None = None
    already_processing: bool = False
    scene_changed: bool = False

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.similar_object_count < 0:
            raise ValueError("similar_object_count must be >= 0")
        for name in ("object_distance_cm", "position_uncertainty_cm"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when supplied")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Event":
        return cls(**{key: val for key, val in value.items() if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BeliefState:
    b_N: float = 0.0
    b_I: float = 0.0
    b_C: float = 0.0

    def __post_init__(self) -> None:
        for name in ("b_N", "b_I", "b_C"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BeliefState":
        return cls(
            b_N=float(value.get("b_N", 0.0)),
            b_I=float(value.get("b_I", 0.0)),
            b_C=float(value.get("b_C", 0.0)),
        )


@dataclass(frozen=True)
class TaskRequirements:
    status: Action
    required: frozenset[str] = field(default_factory=frozenset)
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeEstimate:
    expected_requests: float = 0.0
    expected_reobservations: float = 0.0
    p_wrong: float = 0.0
    p_unresolved: float = 0.0
    expected_followup_bytes: float = 0.0


@dataclass(frozen=True)
class CandidateCost:
    items: tuple[str, ...]
    initial_bytes: int
    transmission_cost: float
    request_cost: float
    wrong_cost: float
    reobservation_cost: float
    unresolved_cost: float
    followup_transmission_cost: float
    total_cost: float
    outcome: OutcomeEstimate

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": list(self.items),
            "initial_bytes": self.initial_bytes,
            "transmission_cost": self.transmission_cost,
            "request_cost": self.request_cost,
            "wrong_cost": self.wrong_cost,
            "reobservation_cost": self.reobservation_cost,
            "unresolved_cost": self.unresolved_cost,
            "followup_transmission_cost": self.followup_transmission_cost,
            "total_cost": self.total_cost,
            "outcome": {
                "expected_requests": self.outcome.expected_requests,
                "expected_reobservations": self.outcome.expected_reobservations,
                "p_wrong": self.outcome.p_wrong,
                "p_unresolved": self.outcome.p_unresolved,
                "expected_followup_bytes": self.outcome.expected_followup_bytes,
            },
        }


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    selected_items: tuple[str, ...] = ()
    estimated_bytes: int = 0
    expected_cost: float = 0.0
    reason: str = ""
    candidate_costs: tuple[CandidateCost, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "selected_items": list(self.selected_items),
            "estimated_bytes": self.estimated_bytes,
            "expected_cost": self.expected_cost,
            "reason": self.reason,
            "candidate_costs": [cost.as_dict() for cost in self.candidate_costs],
        }

