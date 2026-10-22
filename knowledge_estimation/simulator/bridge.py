"""Serializable A-side observations, independent of graphics and middleware.

A future ROS subscriber can translate a received message to ``Observation`` and
call ``EstimatorAdapter.handle``. B's actual cache, physical truth, and simulator
object state have deliberately no field in this contract. Timestamps are A's
processing/simulation clock in seconds; coordinates used elsewhere are metres
in one map frame, with y pointing up and angles in radians.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any

from ..probabilistic import KnowledgeEstimator
from ..types import BeliefSnapshot, Information, KnowledgeItem, KnowledgeKey


class ObservationKind(str, Enum):
    SEND = "send"
    ACK = "ack"
    TIMEOUT = "timeout"
    STATUS = "status"
    PASS_BY = "pass_by"
    PASS_CONFIRMATION = "pass_confirmation"
    RESULT = "result"
    VERIFICATION = "verification"
    MISSING = "missing"
    VERSION = "version"
    REVOKE = "revoke"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("observation keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError("observation values must be finite JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Observation:
    kind: ObservationKind
    timestamp: float
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ObservationKind(self.kind))
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not isfinite(self.timestamp)
            or self.timestamp < 0
        ):
            raise ValueError("timestamp must be a finite non-negative number")
        if not isinstance(self.data, Mapping):
            raise ValueError("data must be an object")
        object.__setattr__(self, "data", _freeze(self.data))

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "timestamp": self.timestamp, "data": _thaw(self.data)}

    @classmethod
    def from_dict(cls, value: Mapping) -> "Observation":
        if not isinstance(value, Mapping) or set(value) != {"kind", "timestamp", "data"}:
            raise ValueError("observation requires exactly kind, timestamp, data")
        return cls(value["kind"], value["timestamp"], value["data"])


# Required/allowed fields are intentionally strict: truth labels cannot be
# appended accidentally when a simulator or a middleware callback is connected.
_FIELDS = {
    ObservationKind.SEND: ({"msg_id", "items"}, {"msg_id", "items", "attempt_id", "p_delivery", "p_ack"}),
    ObservationKind.ACK: ({"msg_id", "stored_items", "memory_epoch"}, {"msg_id", "stored_items", "memory_epoch", "needs_info"}),
    ObservationKind.TIMEOUT: ({"msg_id"}, {"msg_id", "attempt_id"}),
    ObservationKind.STATUS: ({"memory_epoch"}, {"memory_epoch", "boot_id"}),
    ObservationKind.PASS_BY: ({"pass_id", "items"}, {"pass_id", "items", "normal_before", "p_pass", "p_visible", "p_store", "dependency_group"}),
    ObservationKind.PASS_CONFIRMATION: ({"pass_id", "normal_after"}, {"pass_id", "normal_after"}),
    ObservationKind.RESULT: ({"task_id"}, {"task_id", "learned_items", "status", "current_item", "memory_epoch", "dependency_group"}),
    ObservationKind.VERIFICATION: ({"task_id", "success"}, {"task_id", "success"}),
    ObservationKind.MISSING: ({"item"}, {"item", "reason", "memory_epoch"}),
    ObservationKind.VERSION: ({"item"}, {"item"}),
    ObservationKind.REVOKE: ({"evidence_id"}, {"evidence_id"}),
}


def item_to_dict(item: KnowledgeItem) -> dict:
    return {
        "kind": item.key.kind.value,
        "object_id": item.key.object_id,
        "site_id": item.key.site_id,
        "version": item.version,
    }


def item_from_dict(value: Mapping) -> KnowledgeItem:
    if not isinstance(value, Mapping):
        raise ValueError("item must be an object")
    if not {"kind", "object_id", "version"} <= set(value) or set(value) - {"kind", "object_id", "site_id", "version"}:
        raise ValueError("item requires kind, object_id, version and optional site_id")
    return KnowledgeItem(
        KnowledgeKey(Information(value["kind"]), value["object_id"], value.get("site_id")),
        value["version"],
    )


class EstimatorAdapter:
    """Turn observed messages into core calls without a GUI or ROS dependency."""

    def __init__(self, estimator: KnowledgeEstimator | None = None) -> None:
        self.estimator = estimator or KnowledgeEstimator()

    def handle(self, event: Observation) -> None:
        if not isinstance(event, Observation):
            raise TypeError("handle expects an Observation")
        required, allowed = _FIELDS[event.kind]
        fields = set(event.data)
        if not required <= fields or fields - allowed:
            raise ValueError(f"invalid fields for {event.kind.value}: {sorted(fields)}")
        data = _thaw(event.data)
        for name in ("success", "normal_before", "normal_after"):
            if name in data and not isinstance(data[name], bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("p_delivery", "p_ack", "p_pass", "p_visible", "p_store"):
            if name in data and data[name] is not None and (
                isinstance(data[name], bool) or not isinstance(data[name], (int, float))
            ):
                raise ValueError(f"{name} must be a probability, not a boolean or string")
        for name in ("items", "stored_items", "needs_info", "learned_items"):
            if name in data:
                if not isinstance(data[name], list):
                    raise ValueError(f"{name} must be an array")
                data[name] = [item_from_dict(item) for item in data[name]]
        for name in ("item", "current_item"):
            if name in data and data[name] is not None:
                data[name] = item_from_dict(data[name])
        method = {
            ObservationKind.SEND: self.estimator.record_send,
            ObservationKind.ACK: self.estimator.observe_ack,
            ObservationKind.TIMEOUT: self.estimator.observe_ack_timeout,
            ObservationKind.STATUS: self.estimator.observe_status,
            ObservationKind.PASS_BY: self.estimator.record_pass_by,
            ObservationKind.PASS_CONFIRMATION: self.estimator.confirm_pass_by,
            ObservationKind.RESULT: self.estimator.record_result,
            ObservationKind.VERIFICATION: self.estimator.verify_task,
            ObservationKind.MISSING: self.estimator.observe_missing,
            ObservationKind.VERSION: self.estimator.register,
            ObservationKind.REVOKE: self.estimator.revoke_evidence,
        }[event.kind]
        method(**data, now=event.timestamp)

    def belief(self, object_id: str, site_id: str, *, now: float | None = None) -> BeliefSnapshot:
        return self.estimator.snapshot(object_id, site_id, now=now)
