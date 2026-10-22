"""Public contracts shared by Robot A and communication policies (Python >= 3.10)."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


def probability(value: float, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def counter(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


class Information(str, Enum):
    NORMAL = "N"
    CURRENT = "C"
    APPEARANCE = "I"


@dataclass(frozen=True)
class KnowledgeKey:
    """N is scoped to (site, object); C and I are scoped to object."""

    kind: Information
    object_id: str
    site_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", Information(self.kind))
        identifier(self.object_id, "object_id")
        if self.kind == Information.NORMAL:
            identifier(self.site_id, "site_id for N")
        elif self.site_id is not None:
            raise ValueError("C and I must not have site_id")


@dataclass(frozen=True)
class KnowledgeItem:
    key: KnowledgeKey
    version: int = 1

    def __post_init__(self) -> None:
        counter(self.version, "version")


@dataclass(frozen=True)
class EstimatorConfig:
    p_visible: float = 0.8
    p_store: float = 0.9
    q_learn: float = 0.9
    r_crop: float = 0.95
    independent_sources: bool = True

    def __post_init__(self) -> None:
        for name in ("p_visible", "p_store", "q_learn", "r_crop"):
            probability(getattr(self, name), name)
        if not isinstance(self.independent_sources, bool):
            raise ValueError("independent_sources must be bool")


@dataclass(frozen=True)
class Evidence:
    """p_valid = P(stored) * P(correct and usable | stored)."""

    evidence_id: str
    item: KnowledgeItem
    source: str
    p_stored: float
    p_usable_given_stored: float
    dependency_group: str
    provenance_ids: tuple[str, ...]
    timestamp: float

    @property
    def p_valid(self) -> float:
        return self.p_stored * self.p_usable_given_stored


@dataclass(frozen=True)
class ItemBelief:
    item: KnowledgeItem
    probability: float
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class BeliefSnapshot:
    """Immutable decision-time estimates; never contains B's ground truth."""

    receiver_id: str
    memory_epoch: int
    timestamp: float
    normal: ItemBelief
    current: ItemBelief
    appearance: ItemBelief

    def as_policy_input(self) -> dict[str, float]:
        return {
            "b_N": self.normal.probability,
            "b_C": self.current.probability,
            "b_I": self.appearance.probability,
        }


@dataclass(frozen=True)
class UpdateRecord:
    action: str
    timestamp: float
    reference: str
    items: tuple[KnowledgeItem, ...] = ()
