"""Adapter: the knowledge_estimation core (KnowledgeEstimator) as an estimator inside the team 2D world (sim/).

World(...).knows["core"] = CoreKnowledge(world, parameters). It exposes the same per-spot arrays the world and
the policies read (obj = b_I, normal = b_N, cur = b_C of the open anomaly), but every value comes from
KnowledgeEstimator.estimate() fed with A-side events only:

  world event                                   core call
  task / retransmit / TASK v2 / answer / POSE   record_send(msg_id, items, attempt_id)
  ACK deadline passed without ACK               observe_ack_timeout (measured p_recv, p_ack, deadline)
  ACK (carries B's memory epoch)                observe_ack(stored items, memory_epoch)
  needs_info / info request                     observe_missing(reason="needs_info")
  not_found                                     observe_missing(reason="not_found")   (diagnostic only)
  B's trip report (A knows the route)           record_pass_by(normal_before=spot had no open event)
  A sees the spot normal / abnormal again       confirm_pass_by(normal_after=...)     (Notion section 8)
                                                P(pass stored) = perspective model: A re-draws B's planned route
                                                with its own map and D455 model (probabilistic/perspective.py)
  A's re-check verifies the repair              record_result + verify_task (measured q_learn)
  A finds B handled the wrong object            observe_missing(I) + revoke that task's evidence
  A learns a new memory epoch                   observe_status(epoch)
  A re-measures the target pose                 register a new C version (older C evidence is void)

Three levels (Notion section 5), computed with knowledge_estimation.probabilistic.projection:
  b_raw   KnowledgeEstimator.estimate(); None when a needed parameter is unmeasured
  b_cal   b_raw mapped by a fixed PAV score calibration fitted on decision-time records of separate seeds
          (sim2d_calibrate.py --scores); None outside its measured support; identity when no model is given
  b_safe  b_cal under the policy's caps (a policy choice; none by default, so b_safe = b_cal)
The policy reads `level` (b_safe by default). A None there is passed as 0.0 - the declared conservative rule
"unknown means B does not hold it", so A sends - and counted in `unsupported`; the core itself never fills it.
Every decision leaves a decision-time record (section 11) in `decisions`: all three levels, the reason the
calibration was or was not applied, the evidence sources, the estimator fingerprint and the calibration id.
Differences from features.BKnowledge("proposed") are deliberate core semantics: C is never inferred from
passing (it needs a versioned report), and not_found does not clear C because its cause is unknown.
Built without plans, the same estimator reads the older per-(item, site, route) p_acq frequency table instead;
the world keeps that variant as "core_table", for comparison only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..parameters import MissingMeasurement, ParameterStore
from ..probabilistic import KnowledgeEstimator
from ..probabilistic.perspective import PerspectiveAcquisition
from ..probabilistic.projection import calibration_result, validate_model
from ..types import Information, KnowledgeItem, KnowledgeKey

PARAMETERS = Path(__file__).with_name("sim2d_parameters_v2.json")
SCORE_MODEL = Path(__file__).with_name("sim2d_score_calibration_v2.json")              # perspective (default)
SCORE_MODEL_TABLE = Path(__file__).with_name("sim2d_score_calibration_table_v2.json")  # p_acq table (comparison)


def load_parameters(path=PARAMETERS):
    """Measured simulation parameters (sim2d_calibrate.py), or None when they have not been measured yet."""
    path = Path(path)
    return ParameterStore.load(path, allow_simulation=True) if path.exists() else None


POLICY = Path(__file__).with_name("sim2d_policy_v2.json")


def load_caps(path=POLICY):
    """b_safe caps chosen on validation seeds (sim2d_rule_select.py), or {} (b_safe = b_cal)."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))["caps"] if path.exists() else {}


def load_score_model(path=SCORE_MODEL):
    """Fixed b_cal model (sim2d_calibrate.py --scores), or None (then b_cal = b_raw, marked uncalibrated)."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

# items in the team world (rl/features.py): CUR=1, NORM=2, IMG=3 (ID never carries N/C/I knowledge here)
CUR, NORM, IMG = 1, 2, 3
C_TTL_SECONDS = 600.0  # declared task protocol TTL for a current pose (the world has no time-based expiry)
PASS_CONDITION, CONTENT_CONDITION, LEARNING_CONDITION = "pass", "content", "verified"


def channel_condition(loss):
    """A knows its own link class from its measured loss (the observation already carries it)."""
    return "link_good" if loss < 0.05 else "link_lossy"


class _Probe:
    """Array view: kn.obj[k] etc. evaluate the core estimate when read."""

    def __init__(self, owner, kind):
        self.owner, self.kind = owner, kind

    def __getitem__(self, k):
        return self.owner.value(self.kind, int(k))

    def __len__(self):
        return self.owner.n

    def __array__(self, dtype=None, copy=None):
        return np.array([self[k] for k in range(self.owner.n)], dtype=dtype or float)


class CoreKnowledge:
    mode = "core"

    def __init__(self, world, parameters: ParameterStore | None, spots_xy, score_model=None, caps=None,
                 level="safe", plans=None, camera=None):
        """parameters=None: nothing measured yet, so every b_raw is None. score_model must belong to exactly
        this estimator and parameter set (fingerprint), as projection.project requires.
        plans (route_id -> B's planned poses) + camera: the perspective pass-by model; plans=None: p_acq table."""
        if level not in ("raw", "cal", "safe"):
            raise ValueError("level must be raw, cal or safe")
        self.world, self.n = world, len(spots_xy)
        parameters = parameters or ParameterStore()
        acquisition = None if plans is None else PerspectiveAcquisition(
            plans, {f"S{k}": xy for k, xy in enumerate(spots_xy)}, camera, parameters)  # A's map: no occluder
        self.core = KnowledgeEstimator("B", parameters=parameters, acquisition=acquisition)
        self.score_model, self.caps, self.level = score_model, dict(caps or {}), level
        if score_model is not None:
            validate_model(score_model)
            if score_model["estimator_fingerprint"] != self.core.fingerprint or "B" not in score_model["receiver_ids"]:
                raise ValueError("score calibration belongs to a different estimator/parameter set; refit it")
        self.decisions = []  # decision-time records (section 11)
        self.channel = channel_condition(world.loss)
        self.deadline = self._deadline(parameters)
        self.normal_item, self.look_item, self.cur_item = {}, {}, {}
        for k, xy in enumerate(spots_xy):
            self.normal_item[k] = KnowledgeItem(KnowledgeKey(Information.NORMAL, f"Q{k}", f"S{k}"), 1, "map", 0.0,
                                                value=(float(xy[0]), float(xy[1])), uncertainty=(0.0, 0.0),
                                                registration_evidence_id="approved-layout-v1")
            self.look_item[k] = KnowledgeItem(KnowledgeKey(Information.APPEARANCE, f"Q{k}"), 1, "map", 0.0,
                                              content_id=f"crop-Q{k}", value=f"template-Q{k}")
            self.core.register(self.normal_item[k], now=0.0)
            self.core.register(self.look_item[k], now=0.0)
        self.c_version = {k: 0 for k in range(self.n)}
        self.messages = {}      # id(payload) -> (msg_id, items, payload); holding the payload keeps its id unique
        self.waiting = {}       # (msg_id, attempt) -> deadline time
        self.passes = {}        # spot -> [pass_id awaiting confirmation]
        self.tasks = {}         # spot -> task_id of the last verified repair
        self.count = 0
        self.unsupported = self.contradictions = 0
        self.obj, self.normal, self.cur = _Probe(self, "I"), _Probe(self, "N"), _Probe(self, "C")

    def _deadline(self, parameters):
        """The measured ACK deadline of this episode's channel; None if that channel is unmeasured."""
        try:
            deadline = parameters.entry("p_recv", dict(receiver_id="B", kind="", site_id="", route_id="",
                                                       condition_id=self.channel)).get("deadline_seconds")
        except MissingMeasurement:
            return None
        return None if deadline is None else float(deadline)

    # ----- time -----
    def _now(self):
        """Process ACK deadlines that have passed, then return A's clock."""
        now = float(self.world.t)
        if self.deadline is not None:
            for (msg_id, attempt), due in sorted(self.waiting.items(), key=lambda kv: kv[1]):
                if due <= now:
                    del self.waiting[(msg_id, attempt)]
                    try:
                        self.core.observe_ack_timeout(msg_id, attempt_id=attempt, now=max(due, self.core._time))
                    except MissingMeasurement:  # unmeasured channel: the core keeps None for this attempt
                        pass
                    except ValueError:  # measured p_recv = p_ack = 1 contradicts a missing ACK; not hidden
                        self.contradictions += 1
        return now

    # ----- reads -----
    def levels(self, kind, k):
        """b_raw / b_cal / b_safe of one item now (projection.project per item) with the A-side context:
        the evidence sources and the age of A's latest evidence (seconds), None when there is none."""
        now = self._now()
        self.core._tick(now)
        item = {"N": self.normal_item, "I": self.look_item, "C": self.cur_item}[kind].get(k)
        if item is None:  # no open event at k: there is no C item to hold
            return dict(raw=0.0, cal=0.0, safe=0.0, reason="no_item", sources=(), age=None)
        belief = self.core.estimate(item)
        raw = None if belief.probability is None else float(belief.probability)
        sources = tuple(sorted({e.source for e in belief.evidence}))
        age = now - max(e.timestamp for e in belief.evidence) if belief.evidence else None
        if self.score_model is None:
            cal, reason = (0.0, "hard_invalid_zero") if belief.hard_invalid else (raw, "uncalibrated_identity")
        else:
            try:
                cal, reason = calibration_result(self.score_model, kind, raw, hard_invalid=belief.hard_invalid)
            except MissingMeasurement:  # no extrapolation outside the scores the map was fitted on
                cal, reason = None, "outside_calibration_support"
        safe = None if cal is None else min(cal, self.caps.get(f"b_{kind}", 1.0))
        return dict(raw=raw, cal=cal, safe=safe, reason=reason, sources=sources, age=age)

    def value(self, kind, k):
        """What the policy reads: the configured level; None -> 0.0 by the declared conservative rule."""
        v = self.levels(kind, k)[self.level]
        if v is None:
            self.unsupported += 1
            return 0.0
        return v

    def decision(self, k):
        """Decision-time record, taken before the message is chosen and sent (section 11)."""
        rec = dict(t=float(self.world.t), spot=k, fingerprint=self.core.fingerprint,
                   calibration_id=self.score_model.get("model_id") if self.score_model else None)
        for kind in "NIC":
            rec[kind] = self.levels(kind, k)
        self.decisions.append(rec)
        return rec

    # ----- events added by the integration hooks (World._core) -----
    def pose_measured(self, k, cur_ver, observed_at):
        """A measured the target pose: a new C version (a different version voids older C evidence)."""
        self.c_version[k] += 1
        self.cur_item[k] = KnowledgeItem(KnowledgeKey(Information.CURRENT, f"Q{k}"), self.c_version[k], "map",
                                         float(observed_at), ttl_seconds=C_TTL_SECONDS,
                                         value=(float(cur_ver), float(observed_at)), uncertainty=(0.0,))
        self.core.register(self.cur_item[k], now=self._now())

    def _items(self, k, items):
        out = []
        if NORM in items:
            out.append(self.normal_item[k])
        if CUR in items and k in self.cur_item:
            out.append(self.cur_item[k])
        if IMG in items:
            out.append(self.look_item[k])
        return out

    def send(self, k, payload, items, attempt):
        now = self._now()
        key = id(payload)
        if key not in self.messages:
            self.count += 1
            self.messages[key] = (f"m{self.count}", self._items(k, items), payload)
        msg_id, payload_items, _ = self.messages[key]
        self.core.record_send(msg_id, payload_items, attempt_id=attempt, channel_condition=self.channel,
                              content_condition=CONTENT_CONDITION, now=now)
        if self.deadline is not None:
            self.waiting[(msg_id, attempt)] = now + self.deadline

    def ack(self, payload, epoch):
        now = self._now()
        if id(payload) not in self.messages:
            return
        msg_id, payload_items, _ = self.messages[id(payload)]
        for key in [w for w in self.waiting if w[0] == msg_id]:
            del self.waiting[key]
        self.core.observe_ack(msg_id, payload_items, memory_epoch=int(epoch), now=now)

    def trip(self, k, open_spots, route_spots):
        """B's trip report: A knows B drove to k, so it records a pass for every spot on its own route map."""
        now = self._now()
        for j in route_spots:
            if j == k:
                continue
            self.count += 1
            pass_id = f"p{self.count}"
            self.core.record_pass_by(pass_id, [self.normal_item[j], self.look_item[j]], route_id=f"to{k}",
                                     condition_id=PASS_CONDITION, site_id=f"S{j}", normal_before=j not in open_spots,
                                     complete=True, now=now)
            self.passes.setdefault(j, []).append(pass_id)

    # ----- the BKnowledge interface the world already calls (World._knows) -----
    def on_ack(self, spot, items):
        """Handled by ack() with the message identity and epoch."""

    def on_pass(self, qs, open_spots):
        """Handled by trip() with the destination."""

    def on_new_anomaly(self, spot):
        """A new event at the spot; its C item is registered when A measures the pose (pose_measured)."""

    def on_spot_seen(self, spot, normal):
        now = self._now()
        for pass_id in self.passes.pop(spot, []):
            self.core.confirm_pass_by(pass_id, normal_after=bool(normal), now=now)

    def on_missing(self, spot, items):
        now = self._now()
        if CUR in items and spot in self.cur_item:  # not_found: cause unknown, diagnostic only
            self.core.observe_missing(self.cur_item[spot], reason="not_found", now=now)
        for item in self._items(spot, set(items) - {CUR}):
            self.core.observe_missing(item, reason="needs_info", now=now)

    def on_success(self, spot):
        """A's own re-check verified the repair: B reported handling it, A now trusts what B learned there."""
        now = self._now()
        self.count += 1
        task_id = f"t{self.count}"
        self.core.record_result(task_id, [self.normal_item[spot], self.look_item[spot]], status="ok",
                                learning_condition=LEARNING_CONDITION, now=now)
        self.core.verify_task(task_id, success=True, now=now)
        self.tasks[spot] = task_id

    def on_wrong(self, spot):
        """A's re-check: B handled the other object, so the look B stored there is void."""
        now = self._now()
        if spot in self.tasks:
            self.core.revoke_evidence(self.tasks.pop(spot), now=now)
        self.core.observe_missing(self.look_item[spot], reason="missing", now=now)

    def reset(self):
        """A learned B's new memory epoch (STATUS or an ACK)."""
        self.core.observe_status(int(self.world.a_epoch), now=self._now())
