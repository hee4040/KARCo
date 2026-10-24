"""Measured-evidence receiver estimator for the supplied Notion specification.

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
from .probabilities import independent_union, receipt_given_no_ack
import hashlib
from pathlib import Path
from ..parameters import MissingMeasurement, ParameterStore


_PACKAGE = Path(__file__).resolve().parents[1]
_CORE_CODE_FINGERPRINT = hashlib.sha256(b"".join(path.read_bytes().replace(b"\r\n", b"\n") for path in (
    Path(__file__).resolve(), _PACKAGE / "types.py", _PACKAGE / "parameters.py",
    Path(__file__).with_name("probabilities.py"), Path(__file__).with_name("perspective.py")))).hexdigest()


@dataclass
class _Message:
    items: tuple[KnowledgeItem, ...]
    epoch: int
    revisions: dict[KnowledgeKey, int]
    qualities: dict[KnowledgeKey, float | None]
    attempts: dict[int, tuple[float | None, float | None]] = field(default_factory=dict)
    sent_at: dict[int, float] = field(default_factory=dict)
    channel_context: dict | None = None
    parameter_ids: tuple[str, ...] = ()
    timed_out: set[int] = field(default_factory=set)
    receipts: dict[int, float | None] = field(default_factory=dict)
    p_received: float | None = None
    ack_seen: bool = False
    ack_reports: set[tuple[int, frozenset[KnowledgeItem], frozenset[KnowledgeItem]]] = field(default_factory=set)


@dataclass
class _Pass:
    items: tuple[KnowledgeItem, ...]
    epoch: int
    revisions: dict[KnowledgeKey, int]
    normal_before: bool
    timestamp: float
    dependency_group: str
    normal_after: bool | None = None
    context: dict = field(default_factory=dict)
    progress: float = 0.
    complete: bool = False
    layouts: dict = field(default_factory=dict)


@dataclass
class _Task:
    items: tuple[KnowledgeItem, ...]
    epoch: int
    revisions: dict[KnowledgeKey, int]
    status: str
    dependency_group: str
    verified: bool = False
    qualities: dict = field(default_factory=dict)
    parameter_ids: tuple[str, ...] = ()


class KnowledgeEstimator:
    def __init__(
        self, receiver_id: str = "B", *, memory_epoch: int = 0,
        config: EstimatorConfig | None = None,
        parameters: ParameterStore | None = None,
        acquisition=None,
    ) -> None:
        """acquisition: pass-by model, context -> (p_stored, p_usable, parameter_ids), e.g.
        perspective.PerspectiveAcquisition. None = the measured p_acq frequency table of that context."""
        identifier(receiver_id, "receiver_id")
        counter(memory_epoch, "memory_epoch")
        self.receiver_id = receiver_id
        self.memory_epoch = memory_epoch
        self.boot_id: str | None = None
        self.config = config or EstimatorConfig()
        self.parameters = parameters or ParameterStore()
        self.acquisition = acquisition
        self._versions: dict[KnowledgeKey, int] = {}
        self._items_by_key: dict[KnowledgeKey, KnowledgeItem] = {}
        self._revisions: dict[KnowledgeKey, int] = {}
        self._evidence: dict[KnowledgeKey, dict[str, Evidence]] = {}
        self._storage: dict[KnowledgeKey, dict[str, Evidence]] = {}
        self._invalidated: set[KnowledgeKey] = set()
        self._bad_content: set[tuple[KnowledgeKey, int, str | None]] = set()
        self._invalid_frames: set[str] = set()
        self._messages: dict[str, _Message] = {}
        self._passes: dict[str, _Pass] = {}
        self._tasks: dict[str, _Task] = {}
        self._revoked: set[str] = set()
        self._history: list[UpdateRecord] = []
        self._time = 0.0

    @property
    def history(self) -> tuple[UpdateRecord, ...]:
        return tuple(self._history)

    @property
    def fingerprint(self):
        definition = "notion-receiver-v1:" + _CORE_CODE_FINGERPRINT + self.parameters.fingerprint + repr(self.config)
        if self.acquisition is not None:
            definition += ":acquisition:" + self.acquisition.fingerprint
        return hashlib.sha256(definition.encode()).hexdigest()

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
        if item.observed_at is not None and item.observed_at > self._time:
            raise ValueError("sensor observation time cannot be in A's future")
        old = self._versions.get(item.key)
        if old is not None and item.version < old:
            return False
        if old != item.version:
            self._versions[item.key] = item.version
            self._clear(item.key)
            self._log("version", str(item.version), (item,))
        previous = self._items_by_key.get(item.key)
        if (previous is not None and previous.version == item.version and previous.frame_id != item.frame_id
                and item.key.kind in {Information.NORMAL, Information.CURRENT}
                and item.observed_at >= previous.observed_at):
            self._clear(item.key, clear_storage=False)
            self._invalidated.add(item.key)
            self._log("frame_changed", item.frame_id, (item,))
        if previous is None or item.version > previous.version or (
            item.version == previous.version
            and (item.observed_at or 0) >= (previous.observed_at or 0)
        ):
            self._items_by_key[item.key] = item
        if item.key.kind != Information.APPEARANCE and item.frame_id in self._invalid_frames:
            self._invalidated.add(item.key)
        return True

    def _register_all(self, items: tuple[KnowledgeItem, ...]) -> None:
        for item in items:
            self.register(item, now=self._time)

    def _clear(self, key: KnowledgeKey, *, clear_storage=True) -> None:
        self._evidence.pop(key, None)
        if clear_storage:
            self._storage.pop(key, None)
        self._invalidated.discard(key)
        self._revisions[key] = self._revisions.get(key, 0) + 1

    def _revisions_for(self, items) -> dict[KnowledgeKey, int]:
        return {item.key: self._revisions.get(item.key, 0) for item in items}

    def _live(self, item: KnowledgeItem, epoch: int, revisions) -> bool:
        return (
            epoch == self.memory_epoch
            and self._versions.get(item.key) == item.version
            and self._revisions.get(item.key, 0) == revisions.get(item.key)
            and (item.key.kind == Information.APPEARANCE or self._items_by_key[item.key].frame_id == item.frame_id)
        )

    def _put(
        self, evidence_id: str, item: KnowledgeItem, source: str,
        stored: float | None, usable: float | None, group: str, provenance: tuple[str, ...],
        parameter_ids=(), received_at=None,
    ) -> None:
        if item.registration_evidence_id:
            provenance = tuple(provenance) + (item.registration_evidence_id,)
        if evidence_id in self._revoked or any(p in self._revoked for p in provenance):
            return
        if ((item.key, item.version, item.content_id) in self._bad_content
                or (item.key.kind != Information.APPEARANCE and item.frame_id in self._invalid_frames)):
            usable = 0.
            self._invalidated.add(item.key)
        elif not item.expired(self._time):
            self._invalidated.discard(item.key)
        self._evidence.setdefault(item.key, {})[evidence_id] = Evidence(
            evidence_id, item, source, stored, usable, group, provenance, self._time,
            tuple(parameter_ids), received_at,
        )
        self._storage.setdefault(item.key, {})[evidence_id] = self._evidence[item.key][evidence_id]

    def _context(self, condition_id, *, kind="", site_id="", route_id=""):
        return {"receiver_id": self.receiver_id, "kind": kind, "site_id": site_id,
                "route_id": route_id, "condition_id": condition_id}

    def _measured(self, name, context):
        if context is None:
            return None, ()
        try:
            entry = self.parameters.entry(name, context)
            return entry["value"], (entry["parameter_id"],)
        except MissingMeasurement:
            return None, ()

    @staticmethod
    def _content_group(item: KnowledgeItem) -> str:
        # The same version denotes the same content, even in a new message.
        # Re-sending a content asset must not invent a new quality trial.
        return f"content:{item.key!r}:{item.version}:{item.content_id or 'default'}"

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
        attempt_id: int = 1, channel_condition: str | None = None,
        content_condition: str | None = None,
        now: float | None = None,
    ) -> None:
        """Record an actual transmission. Sending alone does not confirm storage.

        Retransmissions share msg_id/payload and use distinct attempt_id. Crop
        quality is measured r_crop/r_current. Registered N is protocol-valid.
        Missing measurements remain None; ACKs can still record storage R.
        """
        identifier(msg_id, "msg_id")
        for name, value in (("channel_condition", channel_condition), ("content_condition", content_condition)):
            if value is not None:
                identifier(value, name)
        counter(attempt_id, "attempt_id")
        channel = self._context(channel_condition) if channel_condition else None
        p_delivery, delivery_ids = self._measured("p_recv", channel)
        p_ack, ack_ids = self._measured("p_ack", channel)
        payload = self._items(items)
        qualities, quality_ids = {}, []
        for item in payload:
            if item.key.kind == Information.NORMAL:
                qualities[item.key] = 1.
            else:
                context = self._context(content_condition, kind=item.key.kind.value,
                                        site_id=item.key.site_id or "") if content_condition else None
                name = "r_crop" if item.key.kind == Information.APPEARANCE else "r_current"
                qualities[item.key], ids = self._measured(name, context)
                quality_ids.extend(ids)
        existing = self._messages.get(msg_id)
        if existing and existing.channel_context != channel:
            raise ValueError("a retry must retain its measured channel context")
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
            existing.channel_context = channel
            existing.parameter_ids = tuple(delivery_ids + ack_ids) + tuple(quality_ids)
            self._messages[msg_id] = existing
        existing.attempts[attempt_id] = (p_delivery, p_ack)
        existing.sent_at[attempt_id] = self._time
        self._log("send", f"{msg_id}/{attempt_id}", payload)
        if not existing.ack_seen:
            progress = None
            if p_delivery is not None and channel is not None:
                try:
                    progress = p_delivery * self.parameters.cdf("F_recv", channel, 0.)
                    existing.parameter_ids = tuple(dict.fromkeys(existing.parameter_ids + (
                        self.parameters.entry("F_recv", channel)["parameter_id"],)))
                except MissingMeasurement:
                    pass
            existing.receipts[attempt_id] = progress
            self._apply_receipts(msg_id, existing, "send")

    def observe_delivery_progress(
        self, msg_id: str, *, attempt_id: int = 1,
        now: float | None = None,
    ) -> None:
        """A-side delay model, never the hidden physical delivery outcome.

        The value is an unconditional storage probability by this time. Once
        the ACK deadline expires, timeout replaces this attempt's prior.
        """
        msg = self._messages[msg_id]
        delivery, _ = msg.attempts[attempt_id]
        self._tick(now)
        if msg.ack_seen or attempt_id in msg.timed_out:
            return
        if delivery is None or msg.channel_context is None:
            msg.receipts[attempt_id] = None
            self._apply_receipts(msg_id, msg, "unmeasured_delivery_progress")
            raise MissingMeasurement("p_recv and F_recv are required for delivery progress")
        self._tick(now)
        try:
            p_received_by_now = delivery * self.parameters.cdf(
                "F_recv", msg.channel_context, self._time-msg.sent_at[attempt_id])
            msg.parameter_ids = tuple(dict.fromkeys(msg.parameter_ids + (
                self.parameters.entry("F_recv", msg.channel_context)["parameter_id"],)))
        except MissingMeasurement:
            msg.receipts[attempt_id] = None
            self._apply_receipts(msg_id, msg, "unmeasured_delivery_progress")
            raise
        msg.receipts[attempt_id] = p_received_by_now
        self._apply_receipts(msg_id, msg, "delivery_progress")
        self._log("delivery_progress", f"{msg_id}/{attempt_id}", msg.items)

    def _apply_receipts(self, msg_id: str, msg: _Message, source: str) -> None:
        known = [p for p in msg.receipts.values() if p is not None]
        if None in msg.receipts.values() and 1. not in known:
            msg.p_received = None
        else:
            msg.p_received = independent_union(known) if self.config.independent_sources else max(known, default=0.)
        for item in msg.items:
            if self._live(item, msg.epoch, msg.revisions):
                self._put(f"message:{msg_id}", item, source, msg.p_received,
                          msg.qualities[item.key], self._content_group(item), (msg_id,), msg.parameter_ids)

    def observe_ack_timeout(
        self, msg_id: str, *, attempt_id: int = 1, now: float | None = None,
    ) -> None:
        """Call only after the ACK deadline, not on each polling tick."""
        msg = self._messages[msg_id]
        delivery, ack = msg.attempts[attempt_id]
        if msg.ack_seen or attempt_id in msg.timed_out:
            return
        self._tick(now)
        if delivery is None or ack is None or msg.channel_context is None:
            msg.receipts[attempt_id] = None
            self._apply_receipts(msg_id, msg, "unmeasured_timeout")
            msg.timed_out.add(attempt_id)
            raise MissingMeasurement("no-ACK update requires measured p_recv and conditional p_ack")
        deadline = self.parameters.entry("p_recv", msg.channel_context).get("deadline_seconds")
        if deadline is None:
            msg.receipts[attempt_id] = None
            self._apply_receipts(msg_id, msg, "unmeasured_timeout")
            msg.timed_out.add(attempt_id)
            raise MissingMeasurement("channel measurement must declare its ACK deadline")
        self._tick(now)
        if self._time < msg.sent_at[attempt_id] + deadline:
            raise ValueError("ACK timeout precedes the measured deadline")
        if delivery * ack == 1:
            raise ValueError("no ACK is impossible under p_delivery=p_ack=1")
        self._tick(now)
        posterior = receipt_given_no_ack(delivery, ack)
        # Retries add delivery opportunities, not independent copies of image quality.
        msg.receipts[attempt_id] = posterior
        msg.timed_out.add(attempt_id)
        self._apply_receipts(msg_id, msg, "ack_timeout")
        self._log("ack_timeout", f"{msg_id}/{attempt_id}", msg.items)

    def observe_ack(
        self, msg_id: str, stored_items: Iterable[KnowledgeItem], *,
        memory_epoch: int, needs_info: Iterable[KnowledgeItem] = (),
        received_at: float | None = None,
        now: float | None = None,
    ) -> None:
        """ACK affirms only explicitly stored items and versions, never the label."""
        msg = self._messages[msg_id]
        if received_at is not None and (type(received_at) not in (float, int)
                                       or not isfinite(received_at) or received_at < 0
                                       or received_at > (self._time if now is None else now)):
            raise ValueError("received_at must be a synchronized receiver storage time <= ACK processing time")
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
                    msg.parameter_ids, received_at,
                )
        self._log("ack", msg_id, stored)
        for item in missing:
            self.observe_missing(item, reason="needs_info", now=now)

    def observe_missing(
        self, item: KnowledgeItem, *, reason: str = "missing",
        memory_epoch: int | None = None, now: float | None = None,
    ) -> None:
        """Only an explicit cache-invalidity report clears possession.

        Ambiguous identity, not_found, occlusion and action failures are
        diagnostics: their causes cannot be inferred from that report alone.
        """
        identifier(reason, "reason")
        if memory_epoch is not None and not self.observe_status(memory_epoch, now=now):
            return
        self._tick(now)
        if reason not in {"missing", "needs_info", "stale_version", "expired", "invalid_content"}:
            self._log("diagnostic:" + reason, "feedback", (item,))
            return
        if self.register(item, now=now):
            if reason == "invalid_content":
                self._bad_content.add((item.key, item.version, item.content_id))
            self._clear(item.key, clear_storage=reason in {"missing", "needs_info", "stale_version"})
            self._invalidated.add(item.key)
            self._log(reason, "feedback", (item,))

    def record_pass_by(
        self, pass_id: str, items: Iterable[KnowledgeItem], *,
        route_id: str, condition_id: str, site_id: str,
        normal_before: bool, complete: bool = False, elapsed: float | None = None,
        dependency_group: str | None = None, now: float | None = None,
    ) -> None:
        """Use predicted route/visibility known to A, never hidden simulator occlusion.

        N waits for confirm_pass_by(normal_after=True), with normal_before=True.
        Section 8's normal-interval gate applies before pass storage is used.
        """
        identifier(pass_id, "pass_id")
        if not isinstance(normal_before, bool):
            raise ValueError("normal_before must be bool")
        for name, value in (("route_id", route_id), ("site_id", site_id), ("condition_id", condition_id)):
            identifier(value, name)
        if not isinstance(complete, bool) or (not complete and elapsed is None):
            raise ValueError("declare a completed pass or provide elapsed time for measured F_pass")
        if elapsed is not None and (type(elapsed) not in (float, int) or not isfinite(elapsed) or elapsed < 0):
            raise ValueError("elapsed must be finite and non-negative")
        payload = self._items(items)
        if any(i.key.kind == Information.CURRENT for i in payload):
            raise ValueError("pass-by supports N/I; a current pose needs a versioned report")
        if pass_id in self._passes:
            raise ValueError("pass_id already exists; call confirm_pass_by to resolve N")
        group = dependency_group or f"pass:{pass_id}"
        identifier(group, "dependency_group")
        self._tick(now)
        self._register_all(payload)
        layouts = {}
        for content in payload:
            layout_key = KnowledgeKey(Information.NORMAL, content.key.object_id, site_id)
            if layout_key not in self._versions:
                raise ValueError("register A's normal-layout metadata before recording a pass")
            layouts[layout_key] = (self._versions[layout_key], self._revisions[layout_key])
        visit = _Pass(
            payload, self.memory_epoch, self._revisions_for(payload),
            normal_before, self._time, group,
        )
        visit.context = self._context(condition_id, site_id=site_id, route_id=route_id)
        visit.layouts = layouts
        visit.progress = 1. if complete else elapsed
        # Store whether progress is a completion fact or an elapsed measurement.
        visit.complete = complete
        self._passes[pass_id] = visit
        self._apply_pass(pass_id, visit)
        self._log("pass_by", pass_id, payload)

    def observe_frame_invalid(self, frame_id: str, *, reason: str, now: float | None = None) -> None:
        """Explicit transform/frame diagnosis; a mere tag_lost report is not this."""
        identifier(frame_id, "frame_id")
        identifier(reason, "frame diagnosis")
        self._tick(now)
        self._invalid_frames.add(frame_id)
        for key, item in self._items_by_key.items():
            if key.kind != Information.APPEARANCE and item.frame_id == frame_id:
                self._clear(key, clear_storage=False)
                self._invalidated.add(key)
                self._log("frame_invalid:" + reason, frame_id, (item,))

    def _apply_pass(self, pass_id: str, visit: _Pass) -> None:
        for item in visit.items:
            if not self._live(item, visit.epoch, visit.revisions):
                continue
            layout_key = KnowledgeKey(Information.NORMAL, item.key.object_id, visit.context["site_id"])
            if visit.layouts[layout_key] != (self._versions.get(layout_key), self._revisions.get(layout_key)):
                continue
            # Notion section 8: pass-by storage is gated by a confirmed normal
            # interval. No provisional evidence is usable before confirmation.
            if not (
                visit.normal_before and visit.normal_after is True
            ):
                continue
            context = {**visit.context, "kind": item.key.kind.value}
            if self.acquisition is None:
                acquired, ids = self._measured("p_acq", context)
                usable = 1.
            else:
                acquired, usable, ids = self.acquisition(context)
            progress = 1.
            if not visit.complete:
                try:
                    progress = self.parameters.cdf("F_pass", context, visit.progress)
                    ids += (self.parameters.entry("F_pass", context)["parameter_id"],)
                except MissingMeasurement:
                    progress = None
            self._put(
                f"pass:{pass_id}", item, "pass_by",
                acquired * progress if acquired is not None and progress is not None else None, usable,
                visit.dependency_group, (pass_id,), ids,
            )

    def advance_pass(self, pass_id: str, *, elapsed: float, now: float | None = None) -> None:
        visit = self._passes[pass_id]
        if visit.complete:
            raise ValueError("completed passes cannot advance")
        if type(elapsed) not in (int, float) or not isfinite(elapsed) or elapsed < visit.progress:
            raise ValueError("elapsed pass time must be finite and non-decreasing")
        self._tick(now)
        visit.progress = elapsed
        self._apply_pass(pass_id, visit)
        self._log("pass_progress", pass_id, visit.items)

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
        learning_condition: str | None = None,
    ) -> None:
        """task_id identifies an event AND attempt, e.g. e4/attempt2.

        RESULT ok adds no evidence until verify_task. not_found is diagnostic.
        N and I storage must be explicitly reported; C is never inferred here.
        """
        identifier(task_id, "task_id")
        if learning_condition is not None:
            identifier(learning_condition, "learning_condition")
        if status not in {"ok", "ambiguous", "not_found", "already_normal", "grasp_fail", "nav_fail", "tag_lost"}:
            raise ValueError("unknown result status")
        payload = self._items(learned_items)
        if any(i.key.kind == Information.CURRENT for i in payload):
            raise ValueError("learned_items supports explicitly reported N/I only")
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
        for content in payload:
            if content.key.kind == Information.NORMAL:
                approved = self._items_by_key.get(content.key)
                if (approved is None or approved.version != content.version or approved.frame_id != content.frame_id
                        or approved.value != content.value
                        or approved.registration_evidence_id != content.registration_evidence_id):
                    raise ValueError("B's learned N must match A's approved layout; RESULT cannot register a new layout")
            else:
                self.register(content, now=now)
        task = _Task(
            payload, self.memory_epoch, self._revisions_for(payload), status, group,
        )
        parameter_ids = []
        for item in payload:
            if item.key.kind == Information.NORMAL:
                task.qualities[item.key] = 1.
            else:
                context = self._context(learning_condition, kind="I") if learning_condition else None
                task.qualities[item.key], ids = self._measured("q_learn", context)
                parameter_ids.extend(ids)
        task.parameter_ids = tuple(parameter_ids)
        self._tasks[task_id] = task
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
                    f"task:{task_id}", item, "verified_task",
                    1.0, task.qualities[item.key],
                    task.dependency_group, (task_id,), task.parameter_ids,
                )
        task.verified = True
        self._log("task_verified", task_id, task.items)

    def revoke_evidence(self, evidence_id: str, *, now: float | None = None) -> None:
        """Retract a source or its provenance, including future delayed replays."""
        identifier(evidence_id, "evidence_id")
        self._tick(now)
        self._revoked.add(evidence_id)
        removed = []
        for sources in [*self._evidence.values(), *self._storage.values()]:
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
        unknown = []
        for source in sources:
            if source.item.expired(self._time):
                continue
            group = source.dependency_group
            if source.p_valid is None:
                unknown.append(source.evidence_id)
            else:
                groups[group] = max(groups.get(group, 0.0), source.p_valid)
        if self.config.independent_sources:
            result = independent_union(groups.values())
        else:
            result = max(groups.values(), default=0.0)
        if unknown and result != 1.:
            result = None
        # A's new observation does not renew an older source held by B.
        expired = bool(sources) and all(s.item.expired(self._time) for s in sources)
        invalid = expired or item.key in self._invalidated
        semantics = "independent_group_lower_bound" if self.config.independent_sources else "union_lower_bound"
        if unknown and result is None:
            semantics = "unmeasured"
        elif not sources:
            semantics = "no_evidence"
        elif expired:
            semantics = "expired"
        stored = [s.p_stored for s in self._storage.get(item.key, {}).values() if s.item.version == item.version]
        known_stored = max((p for p in stored if p is not None), default=0.)
        stored_probability = None if None in stored and known_stored != 1. else known_stored
        return ItemBelief(item, result, sources, invalid, semantics, tuple(unknown), stored_probability)

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
        if any(k not in self._items_by_key for k in keys):
            raise ValueError("register explicit N/C/I metadata before requesting a policy snapshot")
        values = [self.estimate(self._items_by_key[k]) for k in keys]
        return BeliefSnapshot(self.receiver_id, self.memory_epoch, self._time, *values, self.fingerprint)

    def task_completion_probability(self, *, condition_id: str, elapsed: float) -> float:
        """Measured F_task only; this never marks a task verified or complete."""
        return self.parameters.cdf("F_task", self._context(condition_id), elapsed)
