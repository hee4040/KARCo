"""Evidence-based baseline for the four KARCo HTML scenarios.

Only observations available to A belong here. One instance models one receiver.
Methods are called serially in A's processing order; ``now`` is processing time
in seconds (simulation or monotonic clock), not an out-of-order sensor timestamp.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

from ..types import (
    BeliefSnapshot, EstimatorConfig, Evidence, Information, ItemBelief,
    KnowledgeItem, KnowledgeKey, UpdateRecord, counter, identifier, probability,
)
from .probabilities import acquisition_probability, independent_union, receipt_given_no_ack


@dataclass
class _Message:
    items: tuple[KnowledgeItem, ...]
    epoch: int
    revisions: dict[KnowledgeKey, int]
    qualities: dict[KnowledgeKey, float]
    attempts: dict[int, tuple[float, float]] = field(default_factory=dict)
    timed_out: set[int] = field(default_factory=set)
    p_received: float = 0.0
    ack_seen: bool = False
    ack_reports: set[tuple[int, frozenset[KnowledgeItem], frozenset[KnowledgeItem]]] = field(default_factory=set)


@dataclass
class _Pass:
    items: tuple[KnowledgeItem, ...]
    epoch: int
    revisions: dict[KnowledgeKey, int]
    p_acquire: float
    normal_before: bool
    timestamp: float
    dependency_group: str
    normal_after: bool | None = None


@dataclass
class _Task:
    items: tuple[KnowledgeItem, ...]
    epoch: int
    revisions: dict[KnowledgeKey, int]
    status: str
    dependency_group: str
    verified: bool = False


class KnowledgeEstimator:
    def __init__(
        self, receiver_id: str = "B", *, memory_epoch: int = 0,
        config: EstimatorConfig | None = None,
    ) -> None:
        identifier(receiver_id, "receiver_id")
        counter(memory_epoch, "memory_epoch")
        self.receiver_id = receiver_id
        self.memory_epoch = memory_epoch
        self.boot_id: str | None = None
        self.config = config or EstimatorConfig()
        self._versions: dict[KnowledgeKey, int] = {}
        self._revisions: dict[KnowledgeKey, int] = {}
        self._evidence: dict[KnowledgeKey, dict[str, Evidence]] = {}
        self._messages: dict[str, _Message] = {}
        self._passes: dict[str, _Pass] = {}
        self._tasks: dict[str, _Task] = {}
        self._revoked: set[str] = set()
        self._history: list[UpdateRecord] = []
        self._time = 0.0

    @property
    def history(self) -> tuple[UpdateRecord, ...]:
        return tuple(self._history)

    def _tick(self, now: float | None) -> None:
        if now is None:
            return
        if (isinstance(now, bool) or not isinstance(now, (int, float))
                or not isfinite(now) or now < self._time):
            raise ValueError("now must be finite, non-negative and non-decreasing")
        self._time = now

    def _log(self, action: str, reference: str, items=()) -> None:
        self._history.append(UpdateRecord(action, self._time, reference, tuple(items)))

    @staticmethod
    def _items(items: Iterable[KnowledgeItem]) -> tuple[KnowledgeItem, ...]:
        result = tuple(items)
        if len({item.key for item in result}) != len(result):
            raise ValueError("items must have distinct knowledge keys")
        return result

    def register(self, item: KnowledgeItem, *, now: float | None = None) -> bool:
        """A learned a version. New versions clear old knowledge; stale ones do not."""
        self._tick(now)
        old = self._versions.get(item.key)
        if old is not None and item.version < old:
            return False
        if old != item.version:
            self._versions[item.key] = item.version
            self._clear(item.key)
            self._log("version", str(item.version), (item,))
        return True

    def _register_all(self, items: tuple[KnowledgeItem, ...]) -> None:
        for item in items:
            self.register(item, now=self._time)

    def _clear(self, key: KnowledgeKey) -> None:
        self._evidence.pop(key, None)
        self._revisions[key] = self._revisions.get(key, 0) + 1

    def _revisions_for(self, items) -> dict[KnowledgeKey, int]:
        return {item.key: self._revisions.get(item.key, 0) for item in items}

    def _live(self, item: KnowledgeItem, epoch: int, revisions) -> bool:
        return (
            epoch == self.memory_epoch
            and self._versions.get(item.key) == item.version
            and self._revisions.get(item.key, 0) == revisions.get(item.key)
        )

    def _put(
        self, evidence_id: str, item: KnowledgeItem, source: str,
        stored: float, usable: float, group: str, provenance: tuple[str, ...],
    ) -> None:
        if evidence_id in self._revoked or any(p in self._revoked for p in provenance):
            return
        self._evidence.setdefault(item.key, {})[evidence_id] = Evidence(
            evidence_id, item, source, stored, usable, group, provenance, self._time,
        )

    @staticmethod
    def _content_group(item: KnowledgeItem) -> str:
        # The same version denotes the same content, even in a new message.
        # Re-sending one crop must not turn .95 into .9975.
        return f"content:{item.key!r}:{item.version}"

    def observe_status(
        self, memory_epoch: int, *, boot_id: str | None = None, now: float | None = None,
    ) -> bool:
        """Return False for a delayed old epoch. A boot alone preserves the cache."""
        counter(memory_epoch, "memory_epoch")
        if boot_id is not None:
            identifier(boot_id, "boot_id")
        self._tick(now)
        if memory_epoch < self.memory_epoch:
            self._log("ignored_old_epoch", str(memory_epoch))
            return False
        if memory_epoch > self.memory_epoch:
            for key in self._versions:
                self._clear(key)
            self.memory_epoch = memory_epoch
            self._log("memory_reset", str(memory_epoch))
        if boot_id is not None:
            self.boot_id = boot_id
        return True

    def record_send(
        self, msg_id: str, items: Iterable[KnowledgeItem] = (), *,
        attempt_id: int = 1, p_delivery: float = 0.9, p_ack: float = 0.9,
        content_quality: dict[KnowledgeKey, float] | None = None,
        now: float | None = None,
    ) -> None:
        """Record an actual transmission. Sending alone does not confirm storage.

        Retransmissions share msg_id/payload and use distinct attempt_id. Crop
        quality is r_crop; N/C default to 1 only for A's validated content.
        """
        identifier(msg_id, "msg_id")
        counter(attempt_id, "attempt_id")
        probability(p_delivery, "p_delivery")
        probability(p_ack, "p_ack")
        payload = self._items(items)
        qualities = {
            i.key: self.config.r_crop if i.key.kind == Information.APPEARANCE else 1.0
            for i in payload
        }
        if content_quality:
            if not set(content_quality) <= set(qualities):
                raise ValueError("content_quality contains an item outside the payload")
            qualities.update(content_quality)
        for value in qualities.values():
            probability(value, "content_quality")
        existing = self._messages.get(msg_id)
        if existing and (existing.items != payload or existing.qualities != qualities):
            raise ValueError("a msg_id must retain its original items and content quality")
        if existing and attempt_id in existing.attempts:
            if existing.attempts[attempt_id] != (p_delivery, p_ack):
                raise ValueError("an attempt_id must retain its delivery probabilities")
            return
        self._tick(now)
        self._register_all(payload)
        if existing is None:
            existing = _Message(payload, self.memory_epoch, self._revisions_for(payload), qualities)
            self._messages[msg_id] = existing
        existing.attempts[attempt_id] = (p_delivery, p_ack)
        self._log("send", f"{msg_id}/{attempt_id}", payload)

    def observe_ack_timeout(
        self, msg_id: str, *, attempt_id: int = 1, now: float | None = None,
    ) -> None:
        """Call only after the ACK deadline, not on each polling tick."""
        msg = self._messages[msg_id]
        delivery, ack = msg.attempts[attempt_id]
        if msg.ack_seen or attempt_id in msg.timed_out:
            return
        if delivery * ack == 1:
            raise ValueError("no ACK is impossible under p_delivery=p_ack=1")
        self._tick(now)
        posterior = receipt_given_no_ack(delivery, ack)
        # Retries add delivery opportunities, not independent copies of image quality.
        msg.p_received = independent_union((msg.p_received, posterior))
        msg.timed_out.add(attempt_id)
        for item in msg.items:
            if self._live(item, msg.epoch, msg.revisions):
                self._put(
                    f"message:{msg_id}", item, "ack_timeout", msg.p_received,
                    msg.qualities[item.key], self._content_group(item), (msg_id,),
                )
        self._log("ack_timeout", f"{msg_id}/{attempt_id}", msg.items)

    def observe_ack(
        self, msg_id: str, stored_items: Iterable[KnowledgeItem], *,
        memory_epoch: int, needs_info: Iterable[KnowledgeItem] = (),
        now: float | None = None,
    ) -> None:
        """ACK affirms only explicitly stored items and versions, never the label."""
        msg = self._messages[msg_id]
        stored = self._items(stored_items)
        missing = self._items(needs_info)
        if not set(stored) <= set(msg.items):
            raise ValueError("ACK stored_items must be a subset of this message's payload")
        if set(stored) & set(missing):
            raise ValueError("an ACK cannot both store and miss the same item")
        report = (memory_epoch, frozenset(stored), frozenset(missing))
        if report in msg.ack_reports:
            return
        if not self.observe_status(memory_epoch, now=now):
            return
        msg.ack_reports.add(report)
        if memory_epoch > msg.epoch:
            # This ACK explicitly describes the new cache, even if A sent the
            # message before discovering the reset. Fence subsequent replays.
            msg.epoch = memory_epoch
            msg.revisions = self._revisions_for(msg.items)
        msg.ack_seen = True
        for item in stored:
            if self._live(item, msg.epoch, msg.revisions):
                self._put(
                    f"message:{msg_id}", item, "ack", 1.0, msg.qualities[item.key],
                    self._content_group(item), (msg_id,),
                )
        self._log("ack", msg_id, stored)
        for item in missing:
            self.observe_missing(item, reason="needs_info", now=now)

    def observe_missing(
        self, item: KnowledgeItem, *, reason: str = "missing",
        memory_epoch: int | None = None, now: float | None = None,
    ) -> None:
        """Explicit unusable/missing feedback clears this item, not the whole cache."""
        identifier(reason, "reason")
        if memory_epoch is not None and not self.observe_status(memory_epoch, now=now):
            return
        self._tick(now)
        if self.register(item, now=now):
            self._clear(item.key)
            self._log(reason, "feedback", (item,))

    def record_pass_by(
        self, pass_id: str, items: Iterable[KnowledgeItem], *,
        normal_before: bool = False, p_pass: float = 1.0,
        p_visible: float | None = None, p_store: float | None = None,
        dependency_group: str | None = None, now: float | None = None,
    ) -> None:
        """Use predicted route/visibility known to A, never hidden simulator occlusion.

        N waits for confirm_pass_by(normal_after=True), with normal_before=True.
        I can be acquired without assuming the observed pose was normal.
        """
        identifier(pass_id, "pass_id")
        if not isinstance(normal_before, bool):
            raise ValueError("normal_before must be bool")
        visible = self.config.p_visible if p_visible is None else p_visible
        store = self.config.p_store if p_store is None else p_store
        for value in (p_pass, visible, store):
            probability(value, "pass-by probability")
        payload = self._items(items)
        if any(i.key.kind == Information.CURRENT for i in payload):
            raise ValueError("pass-by supports N/I; a current pose needs a versioned report")
        if pass_id in self._passes:
            raise ValueError("pass_id already exists; call confirm_pass_by to resolve N")
        group = dependency_group or f"pass:{pass_id}"
        identifier(group, "dependency_group")
        self._tick(now)
        self._register_all(payload)
        visit = _Pass(
            payload, self.memory_epoch, self._revisions_for(payload),
            acquisition_probability(p_pass, visible, store), normal_before, self._time, group,
        )
        self._passes[pass_id] = visit
        self._apply_pass(pass_id, visit)
        self._log("pass_by", pass_id, payload)

    def _apply_pass(self, pass_id: str, visit: _Pass) -> None:
        for item in visit.items:
            if not self._live(item, visit.epoch, visit.revisions):
                continue
            if item.key.kind == Information.NORMAL and not (
                visit.normal_before and visit.normal_after is True
            ):
                continue
            self._put(
                f"pass:{pass_id}", item, "pass_by", visit.p_acquire, 1.0,
                visit.dependency_group, (pass_id,),
            )

    def confirm_pass_by(
        self, pass_id: str, *, normal_after: bool, now: float | None = None,
    ) -> None:
        if not isinstance(normal_after, bool):
            raise ValueError("normal_after must be bool")
        visit = self._passes[pass_id]
        if visit.normal_after is not None and visit.normal_after != normal_after:
            raise ValueError("conflicting confirmation; revoke the evidence instead")
        self._tick(now)
        visit.normal_after = normal_after
        self._apply_pass(pass_id, visit)
        self._log("pass_confirmed" if normal_after else "pass_abnormal", pass_id, visit.items)

    def record_result(
        self, task_id: str, learned_items: Iterable[KnowledgeItem] = (), *,
        status: str = "ok", current_item: KnowledgeItem | None = None,
        dependency_group: str | None = None, now: float | None = None,
        memory_epoch: int | None = None,
    ) -> None:
        """task_id identifies an event AND attempt, e.g. e4/attempt2.

        RESULT ok adds no evidence until verify_task. A not_found report only
        invalidates C. Learned items must be explicitly reported appearances.
        """
        identifier(task_id, "task_id")
        if status not in {"ok", "not_found", "failed"}:
            raise ValueError("unknown result status")
        payload = self._items(learned_items)
        if any(i.key.kind != Information.APPEARANCE for i in payload):
            raise ValueError("learned_items must contain I; N/C require explicit reports")
        if status == "not_found" and (
            current_item is None or current_item.key.kind != Information.CURRENT
        ):
            raise ValueError("not_found requires the C item/version searched by B")
        if memory_epoch is not None and not self.observe_status(memory_epoch, now=now):
            return
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.items != payload or task.status != status:
                raise ValueError("a task attempt cannot change its result")
            return
        group = dependency_group or f"task:{task_id}"
        identifier(group, "dependency_group")
        self._tick(now)
        self._register_all(payload)
        self._tasks[task_id] = _Task(
            payload, self.memory_epoch, self._revisions_for(payload), status, group,
        )
        if status == "not_found":
            self.observe_missing(current_item, reason="not_found", now=now)
        self._log("result_pending", task_id, payload)

    def verify_task(
        self, task_id: str, *, success: bool, now: float | None = None,
    ) -> None:
        """A's verification, not B's own success flag. Does not close events."""
        if not isinstance(success, bool):
            raise ValueError("success must be bool")
        task = self._tasks[task_id]
        if not success:
            self.revoke_evidence(task_id, now=now)
            return
        if task.verified or task.status != "ok":
            return
        self._tick(now)
        for item in task.items:
            if self._live(item, task.epoch, task.revisions):
                self._put(
                    f"task:{task_id}", item, "verified_task", self.config.q_learn, 1.0,
                    task.dependency_group, (task_id,),
                )
        task.verified = True
        self._log("task_verified", task_id, task.items)

    def revoke_evidence(self, evidence_id: str, *, now: float | None = None) -> None:
        """Retract a source or its provenance, including future delayed replays."""
        identifier(evidence_id, "evidence_id")
        self._tick(now)
        self._revoked.add(evidence_id)
        removed = []
        for sources in self._evidence.values():
            for key, evidence in list(sources.items()):
                if key == evidence_id or evidence_id in evidence.provenance_ids:
                    removed.append(evidence.item)
                    del sources[key]
        self._log("evidence_revoked", evidence_id, removed)

    def estimate(self, item: KnowledgeItem) -> ItemBelief:
        sources = tuple(self._evidence.get(item.key, {}).values())
        if self._versions.get(item.key) != item.version:
            sources = ()
        groups: dict[str, float] = {}
        for source in sources:
            group = source.dependency_group
            groups[group] = max(groups.get(group, 0.0), source.p_valid)
        if self.config.independent_sources:
            result = independent_union(groups.values())
        else:
            result = max(groups.values(), default=0.0)
        return ItemBelief(item, result, sources)

    def snapshot(
        self, object_id: str, site_id: str, *, now: float | None = None,
    ) -> BeliefSnapshot:
        """Take BEFORE message selection/transmission; policy order is N,C,I."""
        self._tick(now)
        keys = (
            KnowledgeKey(Information.NORMAL, object_id, site_id),
            KnowledgeKey(Information.CURRENT, object_id),
            KnowledgeKey(Information.APPEARANCE, object_id),
        )
        values = [self.estimate(KnowledgeItem(k, self._versions.get(k, 1))) for k in keys]
        return BeliefSnapshot(self.receiver_id, self.memory_epoch, self._time, *values)
