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
    version: int
    frame_id: str | None = None
    observed_at: float | None = None
    ttl_seconds: float | None = None
    content_id: str | None = None
    association_confidence: float | None = None
    value: tuple[float, ...] | str | None = None
    uncertainty: tuple[float, ...] | None = None
    registration_evidence_id: str | None = None

    def __post_init__(self) -> None:
        counter(self.version, "version")
        identifier(self.frame_id, "frame_id")
        if self.observed_at is None:
            raise ValueError("observed_at is required; do not invent a sensor timestamp")
        if self.key.kind == Information.CURRENT and self.ttl_seconds is None:
            raise ValueError("C requires an explicitly specified TTL from the task/measurement protocol")
        if self.value is None:
            raise ValueError("item requires a measured pose or a template/descriptor reference")
        if self.key.kind in {Information.NORMAL, Information.CURRENT} and self.uncertainty is None:
            raise ValueError("pose items require their recorded uncertainty; do not assume zero error")
        if self.key.kind == Information.NORMAL:
            identifier(self.registration_evidence_id, "approved normal-layout registration evidence")
        if self.association_confidence is not None:
            probability(self.association_confidence, "association_confidence")
        for name in ("value", "uncertainty"):
            values = getattr(self, name)
            if values is not None and not (name == "value" and isinstance(values, str)):
                if not isinstance(values, tuple) or not all(type(v) in (float, int) and isfinite(v) for v in values):
                    raise ValueError(f"{name} requires an immutable tuple of finite measured values")
        for name in ("frame_id", "content_id"):
            if getattr(self, name) is not None:
                identifier(getattr(self, name), name)
        for name in ("observed_at", "ttl_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.ttl_seconds is not None and (
            self.key.kind != Information.CURRENT or self.observed_at is None
        ):
            raise ValueError("TTL requires C and its sensor observation timestamp")

    def expired(self, now: float) -> bool:
        return self.ttl_seconds is not None and now >= self.observed_at + self.ttl_seconds


@dataclass(frozen=True)
class EstimatorConfig:
    # max is a union lower bound. An independent union is an explicit model
    # assumption, requiring the caller's documented justification.
    independent_sources: bool = False
    independence_evidence_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.independent_sources, bool):
            raise ValueError("independent_sources must be bool")
        if self.independent_sources:
            identifier(self.independence_evidence_id, "independence_evidence_id")


@dataclass(frozen=True)
class Evidence:
    """p_valid = P(stored) * P(correct and usable | stored)."""

    evidence_id: str
    item: KnowledgeItem
    source: str
    p_stored: float | None
    p_usable_given_stored: float | None
    dependency_group: str
    provenance_ids: tuple[str, ...]
    timestamp: float
    parameter_ids: tuple[str, ...] = ()
    receiver_received_at: float | None = None

    @property
    def p_valid(self) -> float | None:
        if self.p_stored == 0 or self.p_usable_given_stored == 0:
            return 0.
        if self.p_stored is None or self.p_usable_given_stored is None:
            return None
        return self.p_stored * self.p_usable_given_stored


@dataclass(frozen=True)
class ItemBelief:
    item: KnowledgeItem
    probability: float | None
    evidence: tuple[Evidence, ...]
    hard_invalid: bool = False
    semantics: str = "no_evidence"
    missing_parameters: tuple[str, ...] = ()
    stored_probability: float | None = None


@dataclass(frozen=True)
class BeliefSnapshot:
    """Immutable decision-time estimates; never contains B's ground truth."""

    receiver_id: str
    memory_epoch: int
    timestamp: float
    normal: ItemBelief
    current: ItemBelief
    appearance: ItemBelief
    estimator_fingerprint: str = ""

    def as_policy_input(self) -> dict[str, float | None]:
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
