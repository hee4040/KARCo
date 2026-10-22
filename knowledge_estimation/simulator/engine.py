"""Deterministic-clock 2D robot, sensor and packet simulation (metres, y up).

This module has no graphics or middleware dependency. Scene truth and B's cache
are separate from A's estimator; all estimator updates cross EstimatorAdapter.
The N/NC message choices are an explicit demonstration policy, not an optimum.
"""

from collections import deque
from copy import deepcopy
from math import atan2, cos, hypot, isfinite, pi
import random

from ..probabilistic import KnowledgeEstimator
from ..probabilistic.visibility import Pose2D, VisibilityConfig, estimate_visibility
from ..types import EstimatorConfig, Information, KnowledgeItem, KnowledgeKey
from .bridge import EstimatorAdapter, Observation, item_to_dict


SCENARIOS = ("first_event", "pass_clear", "pass_occluded", "ack_loss")
_DEFAULTS = {
    "p_visible": .8, "p_store": .9, "q_learn": .9, "r_crop": .95,
    "p_delivery": .9, "p_ack": .9, "robot_speed": 1.0,
    "a_speed": 1.15, "network_delay": .35, "work_duration": 1.2,
}
_SCENE_DEFAULTS = {
    "q_x": 6., "q_y": 3.5, "pass_x": 4.5, "pass_y": 3.5,
    "b_start_x": 1., "b_start_y": 1., "b_camera_range": 2.2,
    "b_camera_fov": .9*pi,
}


def _scene_configuration(values):
    result = dict(_SCENE_DEFAULTS)
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(result):
            raise ValueError("scene must contain only supported geometry settings")
        result.update(values)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"invalid scene setting: {key}")
        if not 0 < value <= (2*pi if key == "b_camera_fov" else 12 if key.endswith("_x") else 8):
            raise ValueError(f"scene setting out of bounds: {key}")
    # Keep the collection family within this two-station demonstration world.
    if not (5.5 <= result["q_x"] <= 6.5 and 3 <= result["q_y"] <= 4
            and 4 <= result["pass_x"] <= 4.8 and 3 <= result["pass_y"] <= 4):
        raise ValueError("Q and pass waypoint must stay within the supported station")
    return result


def _configuration(values):
    result = dict(_DEFAULTS)
    if values is None:
        return result
    if not isinstance(values, dict) or set(values) - set(result):
        raise ValueError("config must contain only supported simulation settings")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number")
        try:
            valid = isfinite(value) and (0 <= value <= 1 if key.startswith("p_") or key in {"q_learn", "r_crop"} else 0 < value <= 20)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"invalid {key}")
        result[key] = float(value)
    return result


def _wire(value):
    if isinstance(value, KnowledgeItem):
        return item_to_dict(value)
    if isinstance(value, dict):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    return value


class Simulator:
    """Advance physical movement and network delivery using fixed 50 ms ticks."""

    def __init__(self, scenario="pass_clear", seed=7, config=None, scene=None):
        self.config = _configuration(config)
        self.scene = _scene_configuration(scene)
        self.scenario = scenario
        self.seed = seed
        self.reset()

    def reset(self, scenario=None, seed=None):
        selected = self.scenario if scenario is None else scenario
        selected_seed = self.seed if seed is None else seed
        if selected not in SCENARIOS:
            raise ValueError(f"unknown scenario: {selected}")
        if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
            raise ValueError("seed must be an integer")
        self.scenario, self.seed = selected, selected_seed
        self.time, self._remainder = 0.0, 0.0
        self._network_rng = random.Random(self.seed)
        self._storage_rng = random.Random(f"{self.seed}/storage")
        self.estimator = KnowledgeEstimator(config=EstimatorConfig(**{
            key: self.config[key] for key in ("p_visible", "p_store", "q_learn", "r_crop")
        }))
        self.adapter = EstimatorAdapter(self.estimator)
        self.observations = deque(maxlen=1000)
        self._observation_history = []  # Full A-only journal; the GUI deque is bounded.
        self.decision_records = []
        self.logs = deque(maxlen=90)
        self.stats = {key: 0 for key in ("bytes_sent", "packets_sent", "packets_lost", "retries", "requests")}
        self._robots = {
            "A": {"x": 10.8, "y": 6.6, "heading": pi, "state": "patrolling", "path": [], "camera_range": 2.3, "camera_fov": 2*pi},
            "B": {"x": self.scene["b_start_x"], "y": self.scene["b_start_y"], "heading": 0.0, "state": "idle", "path": [], "camera_range": self.scene["b_camera_range"], "camera_fov": self.scene["b_camera_fov"]},
        }
        self._pass_waypoint = (self.scene["pass_x"], self.scene["pass_y"])
        self._patrol = [(7.6, 3.5), (10.8, 1.4), (10.8, 6.6)]
        self._patrol_index = 0
        self._objects = {
            "P": {"id": "P", "site_id": "S1", "x": 10.75, "y": 6.15, "normal_x": 10.0, "normal_y": 6.5, "version": 1, "appearance_version": 1},
            "Q": {"id": "Q", "site_id": "S2", "x": self.scene["q_x"], "y": self.scene["q_y"], "normal_x": self.scene["q_x"], "normal_y": self.scene["q_y"], "version": 1, "appearance_version": 1},
        }
        self._known_positions = {key: (obj["x"], obj["y"]) for key, obj in self._objects.items()}
        self._known_normal = {key: self._normal(key) for key in self._objects}
        self._pass_geometry = {}
        self._known_versions = {key: 1 for key in self._objects}
        self._known_appearance_versions = {key: 1 for key in self._objects}
        self._cache = {key: {} for key in self._objects}
        self._memory_epoch = 0
        self._packets = []
        self._outbox = {}
        self._controls = {}
        self._seen_messages = set()
        self._seen_controls = set()
        self._sequence = 0
        self._event_sequence = 0
        self._event = {"id": "", "target": "P", "state": "idle"}
        self._receiver_task = None
        self._pending_verification = None
        self._reobserve_target = None
        self._task_send_at = None
        self._pass_seen = False
        self._pass_confirmed = False
        self._pass_time = None
        self._first_verified = False
        self._second_started = False
        self._verified_at = None
        self._hidden_occlusion = selected == "pass_occluded"
        self._forced_ack_loss = selected == "ack_loss" and self.config["p_ack"] < 1
        for target in self._objects:
            for kind in "NCI":
                self._observe("version", item=self._item(target, kind))
        self._log("시작: A는 등록된 정상 배치와 직전 순찰의 Q 정상 상태를 알고 있습니다.")
        return self.snapshot()

    def _log(self, message):
        self.logs.append({"time": round(self.time, 2), "message": message})

    def _observe(self, kind, **data):
        event = Observation(kind, self.time, _wire(data))
        self.adapter.handle(event)
        self.observations.append(event.to_dict())
        self._observation_history.append(event.to_dict())

    def _item(self, target, kind, version=None):
        obj = self._objects[target]
        if version is None:
            version = self._known_versions[target] if kind == "C" else self._known_appearance_versions[target] if kind == "I" else 1
        return KnowledgeItem(KnowledgeKey(Information(kind), target, obj["site_id"] if kind == "N" else None), version)

    def _normal(self, target):
        obj = self._objects[target]
        return hypot(obj["x"] - obj["normal_x"], obj["y"] - obj["normal_y"]) < .12

    def _truth(self, target):
        obj = self._objects[target]
        truth = {}
        for kind in "NCI":
            cached = self._cache[target].get(kind)
            version = obj["version"] if kind == "C" else obj["appearance_version"] if kind == "I" else 1
            truth[kind] = int(bool(cached and cached["version"] == version and cached["usable"]))
        return truth

    def _decision(self, target, reason):
        belief = self.estimator.snapshot(target, self._objects[target]["site_id"], now=self.time)
        self.decision_records.append({
            "episode_id": f"{self.scenario}/{self.seed}", "scenario": self.scenario,
            "time": self.time, "target": target, "event_id": self._event["id"],
            "reason": reason, "memory_epoch": belief.memory_epoch,
            "versions": {b.item.key.kind.value: b.item.version for b in (belief.normal, belief.current, belief.appearance)},
            "features": belief.as_policy_input(), "labels": self._truth(target),
            "a_view": self.a_view(target), "label_audit": self.label_audit(target),
        })

    def a_view(self, target):
        """A-observable sample only. Never read B's cache or physical versions here.

        Camera specifications and the planned pass waypoint are known mission
        settings. A knows the last observed target pose, not hidden occluders or
        the success of B's observation/storage attempt.
        """
        obj = self._objects[target]
        belief = self.estimator.snapshot(target, obj["site_id"], now=self.time)
        tx, ty = self._known_positions[target]
        return {
            "time": self.time, "target": target,
            "history_end": len(self._observation_history),
            "known_epoch": belief.memory_epoch,
            "known_versions": {v.item.key.kind.value: v.item.version for v in (belief.normal, belief.current, belief.appearance)},
            "known_position": [tx, ty],
            "planned_pass": deepcopy(self._pass_geometry.get(target, self._planned_pass_geometry(target))),
            "baseline": belief.as_policy_input(),
        }

    def _planned_pass_geometry(self, target):
        px, py = self._pass_waypoint
        tx, ty = self._known_positions[target]
        heading = atan2(py - 1., px - 3.5)
        return {"distance": hypot(tx-px, ty-py),
                "bearing_cos": cos(atan2(ty-py, tx-px)-heading),
                "camera_range": self.scene["b_camera_range"],
                "camera_fov": self.scene["b_camera_fov"]}

    def label_audit(self, target):
        """Privileged training/evaluation evidence; never a model input."""
        obj = self._objects[target]
        return {"actual_epoch": self._memory_epoch, "items": {
            kind: {"stored": kind in self._cache[target],
                   "stored_version": self._cache[target].get(kind, {}).get("version"),
                   "actual_version": obj["version"] if kind == "C" else obj["appearance_version"] if kind == "I" else 1,
                   "usable": self._cache[target].get(kind, {}).get("usable", False)}
            for kind in "NCI"
        }}

    def record_sample(self, target, reason="periodic"):
        """Freeze input and labels together at the current processing boundary."""
        self._decision(target, reason)
        return deepcopy(self.decision_records[-1])

    def export_observations(self):
        return deepcopy(self._observation_history)

    def _enqueue(self, sender, kind, data, label, size, *, force_loss=False):
        self._sequence += 1
        probability = self.config["p_ack"] if kind == "ACK" else self.config["p_delivery"]
        lost = force_loss or self._network_rng.random() >= probability
        duration = self.config["network_delay"] + self._network_rng.uniform(0, .12)
        receiver = "B" if sender == "A" else "A"
        self._packets.append({
            "id": f"packet-{self._sequence}", "from": sender, "to": receiver,
            "kind": kind, "data": data, "label": label, "bytes": size,
            "lost": lost, "sent": self.time, "arrival": self.time + duration,
            "start": (self._robots[sender]["x"], self._robots[sender]["y"]),
            "end": (self._robots[receiver]["x"], self._robots[receiver]["y"]), "processed": False,
        })
        self.stats["bytes_sent"] += size
        self.stats["packets_sent"] += 1
        if lost:
            self.stats["packets_lost"] += 1

    def _transmit(self, target, kinds, *, task=False, message_id=None):
        items = tuple(self._item(target, kind) for kind in kinds)
        if message_id is None:
            message_id = f"{self._event['id']}/{'task' if task else 'info'}/{self._sequence + 1}"
            self._outbox[message_id] = {
                "target": target, "items": items, "task": task, "attempt": 0,
                "pose": self._known_positions[target], "acked": False,
                "retry_at": None, "event_id": self._event["id"],
            }
        transmission = self._outbox[message_id]
        if transmission["acked"]:
            return
        self._decision(target, "TASK" if task else "INFO_RESPONSE")
        transmission["attempt"] += 1
        transmission["retry_at"] = None
        transmission["deadline"] = self.time + 2*(self.config["network_delay"] + .12) + .45
        self._observe("send", msg_id=message_id, items=transmission["items"], attempt_id=transmission["attempt"], p_delivery=self.config["p_delivery"], p_ack=self.config["p_ack"])
        selected = "".join(item.key.kind.value for item in transmission["items"])
        size = (120 if task else 40) + sum(30720 if item.key.kind == Information.APPEARANCE else 40 for item in transmission["items"])
        label = f"{'TASK' if task else 'INFO_RESPONSE'} {target}({selected or 'E'}) #{transmission['attempt']}"
        self._enqueue("A", "DATA", {"msg_id": message_id, **transmission}, label, size)
        self._log(f"A → B: {label}")

    def _control(self, kind, data, label, size):
        key = f"control-{self._sequence + 1}"
        self._controls[key] = {"kind": kind, "data": {**data, "control_id": key}, "label": label, "size": size, "next": self.time + 1.6, "attempt": 1}
        self._enqueue("B", kind, self._controls[key]["data"], label, size)

    def _request(self, kinds):
        task = self._receiver_task
        if task is None or task.get("waiting_for") == tuple(kinds):
            return
        task["waiting_for"] = tuple(kinds)
        task["phase"] = "waiting"
        self._robots["B"]["state"] = "waiting_info"
        self._robots["B"]["path"] = []
        self.stats["requests"] += 1
        target = task["target"]
        self._control("REQUEST", {"target": target, "kinds": tuple(kinds), "epoch": self._memory_epoch}, f"INFO_REQUEST {target}({''.join(kinds)})", 60)
        self._log(f"B: {target} {''.join(kinds)} 없음 → 정보 요청")

    def _deliver(self, packet):
        kind, data = packet["kind"], packet["data"]
        if kind == "DATA":
            target, msg_id = data["target"], data["msg_id"]
            duplicate = msg_id in self._seen_messages
            if not duplicate:
                self._seen_messages.add(msg_id)
                for item in data["items"]:
                    letter = item.key.kind.value
                    previous = self._cache[target].get(letter)
                    if previous and previous["version"] > item.version:
                        # A delayed old packet must not roll B's cache back.
                        continue
                    usable = self._storage_rng.random() < self.config["r_crop"] if letter == "I" else True
                    if previous and previous["version"] == item.version and previous["usable"]:
                        usable = True
                    self._cache[target][letter] = {"version": item.version, "usable": usable, "pose": data["pose"]}
            stored = tuple(item for item in data["items"] if self._cache[target].get(item.key.kind.value, {}).get("version") == item.version)
            missing = (self._item(target, "N"),) if data["task"] and "N" not in self._cache[target] else ()
            drop = self._forced_ack_loss
            self._forced_ack_loss = False
            self._enqueue("B", "ACK", {"msg_id": msg_id, "stored": stored, "epoch": self._memory_epoch, "missing": missing}, f"ACK {target} · epoch {self._memory_epoch}", 60 if missing else 40, force_loss=drop)
            if duplicate:
                self._log("B: 같은 msg_id 중복 수신 → 작업 중복 없이 ACK 재전송")
            elif data["task"]:
                self._receiver_task = {"target": target, "task_id": f"{data['event_id']}/attempt1", "attempt": 1, "phase": "new", "waiting_for": None, "work_until": None}
                self._resume_receiver()
            elif self._receiver_task and self._receiver_task["target"] == target and self._receiver_task["phase"] in {"waiting", "waiting_pose"}:
                if self._receiver_task["phase"] == "waiting_pose":
                    task = self._receiver_task
                    task["attempt"] += 1
                    task["task_id"] = task["task_id"].rsplit("/attempt", 1)[0] + f"/attempt{task['attempt']}"
                self._receiver_task["waiting_for"] = None
                self._resume_receiver()
            return
        if kind == "ACK":
            if data["epoch"] < self.estimator.memory_epoch:
                self._log("A: 이전 memory_epoch의 ACK 무시")
                return
            outgoing = self._outbox[data["msg_id"]]
            outgoing["acked"] = True
            self._observe("ack", msg_id=data["msg_id"], stored_items=data["stored"], memory_epoch=data["epoch"], needs_info=data["missing"])
            if outgoing["task"] and self._event["id"] == outgoing["event_id"]:
                self._event["state"] = "executing" if not data["missing"] else "needs_info"
            self._log(f"A: ACK 수신 · epoch {data['epoch']} · 저장 항목 확인")
            return
        key = data.get("control_id")
        self._controls.pop(key, None)
        if key in self._seen_controls:
            return
        self._seen_controls.add(key)
        if kind == "STATUS":
            self._observe("status", memory_epoch=data["epoch"], boot_id=f"boot-{data['epoch']}")
            if self._pending_verification and self._pending_verification["epoch"] < self.estimator.memory_epoch:
                self._pending_verification = None
            self._log(f"A: STATUS 수신 → memory_epoch {data['epoch']}")
        elif kind == "REQUEST":
            for letter in data["kinds"]:
                self._observe("missing", item=self._item(data["target"], letter), reason="missing", memory_epoch=data["epoch"])
            self._event["state"] = "needs_info"
            self._transmit(data["target"], data["kinds"])
        elif kind == "RESULT":
            if data["epoch"] < self.estimator.memory_epoch:
                self._log("A: 이전 memory_epoch의 RESULT 무시")
                return
            self._observe("result", task_id=data["task_id"], learned_items=data.get("learned", ()), status=data["status"], current_item=data.get("current_item"), memory_epoch=data["epoch"])
            self._event["state"] = "result_reported" if data["status"] == "ok" else "needs_info"
            if data["status"] == "ok":
                self._pending_verification = data
                self._log("A: RESULT ok 수신 · 실제 카메라 재검증까지 학습 반영 보류")
            else:
                self._log("A: not_found 수신 → C 무효화 · 다음 순찰에서 재관측")
                self._reobserve_target = data["target"]
                obj = self._objects[data["target"]]
                self._robots["A"]["path"] = [(obj["normal_x"] + .8, obj["normal_y"] + .5)]
                self._robots["A"]["state"] = "inspecting"

    def _resume_receiver(self):
        task = self._receiver_task
        if task is None:
            return
        if "N" not in self._cache[task["target"]]:
            self._request(("N",))
            return
        robot = self._robots["B"]
        goal = (8.9, 5.7) if task["target"] == "P" else (self.scene["q_x"]+.1, self.scene["q_y"]-1.)
        if hypot(robot["x"]-goal[0], robot["y"]-goal[1]) < .15:
            self._at_site()
            return
        route = [(3.5, 1.0), self._pass_waypoint, (4.5, 4.8), goal] if task["target"] == "P" else [(4.5, 2.4), goal]
        robot["path"] = route
        robot["state"] = "moving"
        task["phase"] = "travel"

    def _at_site(self):
        task = self._receiver_task
        target = task["target"]
        if target == "Q":
            current = self._cache[target].get("C")
            if current is None:
                self._request(("C",))
                return
            obj = self._objects[target]
            if hypot(obj["x"]-current["pose"][0], obj["y"]-current["pose"][1]) > .65:
                task["phase"] = "waiting_pose"
                self._robots["B"]["state"] = "waiting_info"
                self._control("RESULT", {"target": target, "task_id": task["task_id"], "status": "not_found", "current_item": self._item(target, "C", current["version"]), "epoch": self._memory_epoch}, "RESULT not_found", 40)
                return
            if not self._cache[target].get("I", {}).get("usable", False):
                self._request(("I",))
                return
        task["phase"] = "working"
        task["work_until"] = self.time + self.config["work_duration"]
        self._robots["B"]["state"] = "working"
        self._log(f"B: {target} 식별 완료 · 정상 배치로 복구 중")

    def _finish_work(self):
        task = self._receiver_task
        target, obj = task["target"], self._objects[task["target"]]
        obj["x"], obj["y"] = obj["normal_x"], obj["normal_y"]
        obj["version"] += 1
        # Handling is observable; the hidden retention/validity outcome is not.
        # A always receives the attempted I learning, while actual valid memory
        # is sampled independently. Failure does not erase a good prior crop.
        learned = (self._item(target, "I", obj["appearance_version"]),)
        if self._storage_rng.random() < self.config["q_learn"]:
            self._cache[target]["I"] = {"version": obj["appearance_version"], "usable": True, "pose": (obj["x"], obj["y"])}
        elif "I" not in self._cache[target]:
            self._cache[target]["I"] = {"version": obj["appearance_version"], "usable": False, "pose": (obj["x"], obj["y"])}
        self._control("RESULT", {"target": target, "task_id": task["task_id"], "status": "ok", "learned": learned, "epoch": self._memory_epoch}, f"RESULT {target} ok", 40)
        task["phase"] = "done"
        self._robots["B"]["state"] = "returning"
        home = (self.scene["b_start_x"], self.scene["b_start_y"])
        self._robots["B"]["path"] = [(4.5, 4.8), self._pass_waypoint, (3.5, 1.0), home] if target == "P" else [(4.5, 2.4), home]
        self._log(f"B: {target} 복구 완료 · RESULT 전송 후 복귀")

    @staticmethod
    def _advance(robot, distance):
        reached = []
        while robot["path"] and distance > 1e-12:
            x, y = robot["path"][0]
            dx, dy = x-robot["x"], y-robot["y"]
            remaining = hypot(dx, dy)
            if remaining > 1e-9:
                robot["heading"] = atan2(dy, dx)
            if remaining <= distance:
                robot["x"], robot["y"] = x, y
                distance -= remaining
                reached.append((tuple(robot["path"].pop(0)), robot["heading"]))
            else:
                robot["x"] += dx / remaining * distance
                robot["y"] += dy / remaining * distance
                break
        return reached

    def _can_see_q_on_pass(self, heading):
        robot, obj = self._robots["B"], self._objects["Q"]
        dx, dy = obj["x"]-robot["x"], obj["y"]-robot["y"]
        angle = atan2(dy, dx)-heading
        blocked = self._hidden_occlusion and self._ray_hits_obstacle(robot["x"], robot["y"], obj["x"], obj["y"])
        return hypot(dx, dy) <= robot["camera_range"] and cos(angle) >= cos(robot["camera_fov"]/2) and not blocked

    @staticmethod
    def _ray_hits_obstacle(x1, y1, x2, y2):
        low, high = 0., 1.
        for start, delta, minimum, maximum in ((x1, x2-x1, 5., 5.25), (y1, y2-y1, 2.8, 4.4)):
            if abs(delta) < 1e-10:
                if not minimum <= start <= maximum:
                    return False
            else:
                entering, leaving = sorted(((minimum-start)/delta, (maximum-start)/delta))
                low, high = max(low, entering), min(high, leaving)
                if low > high:
                    return False
        return True

    def _pass_opportunity(self, heading):
        if self._pass_seen or self.scenario not in {"pass_clear", "pass_occluded"}:
            return
        self._pass_seen, self._pass_time = True, self.time
        # Only the shared plan, known camera specs and A's last observed target
        # location may constrain A's estimate. The hidden obstacle is excluded.
        px, py = self._pass_waypoint
        planned_heading = atan2(py-1., px-3.5)
        self._pass_geometry["Q"] = self._planned_pass_geometry("Q")
        predicted = estimate_visibility(
            [Pose2D(px, py, planned_heading)], self._known_positions["Q"],
            VisibilityConfig(self.scene["b_camera_range"], self.scene["b_camera_fov"], self.config["p_visible"]),
        )
        self._observe("pass_by", pass_id="route-S2", items=(self._item("Q", "N"), self._item("Q", "I")),
                      normal_before=self._known_normal["Q"], p_pass=1.,
                      p_visible=predicted.p_visible, p_store=self.config["p_store"])
        if self._can_see_q_on_pass(heading) and self._storage_rng.random() < self.config["p_store"]:
            obj = self._objects["Q"]
            pose = (obj["x"], obj["y"])
            self._cache["Q"]["I"] = {"version": obj["appearance_version"], "usable": True, "pose": pose}
            self._cache["Q"]["N"] = {"version": 1, "usable": self._normal("Q"), "pose": pose}
            self._log("B 실제 센서: Q가 시야에 들어와 스냅샷 저장")
        else:
            self._log("B 실제 센서: Q 스냅샷 미획득 · A에게 원인 전달 없음")
        self._log("A: 경로 통과 기회 반영 · I 추정 갱신, N은 통과 후 정상 확인 대기")

    def _new_event(self, target):
        self._event_sequence += 1
        self._event = {"id": f"e{self._event_sequence}", "target": target, "state": "confirmed"}
        self._task_send_at = self.time + .3
        self._log(f"A: 카메라로 {target} 이상 확인 → {self._event['id']}")

    def _perceive_a(self):
        robot = self._robots["A"]
        for target, obj in self._objects.items():
            if hypot(robot["x"]-obj["x"], robot["y"]-obj["y"]) > robot["camera_range"]:
                continue
            changed = obj["version"] != self._known_versions[target]
            self._known_positions[target] = (obj["x"], obj["y"])
            self._known_normal[target] = self._normal(target)
            if obj["appearance_version"] != self._known_appearance_versions[target]:
                self._known_appearance_versions[target] = obj["appearance_version"]
                self._observe("version", item=self._item(target, "I"))
            reobserved = self._reobserve_target == target
            if changed:
                self._known_versions[target] = obj["version"]
                self._observe("version", item=self._item(target, "C"))
                self._log(f"A: {target} 재이동 실제 관측 → C v{obj['version']}")
            if changed or reobserved:
                self._reobserve_target = None if reobserved else self._reobserve_target
                if self._event["target"] == target and self._receiver_task and self._receiver_task["phase"] in {"travel", "waiting", "waiting_pose"}:
                    self._transmit(target, ("C",))
            if target == "Q" and self._pass_seen and not self._pass_confirmed and self.time > self._pass_time + .1:
                self._observe("pass_confirmation", pass_id="route-S2", normal_after=self._normal("Q"))
                self._pass_confirmed = True
                self._log("A: 통과 뒤 Q 정상 확인 → N 근거 허용" if self._normal("Q") else "A: 통과 뒤 Q 이상 확인 → N 통과 근거 제외")
            pending = self._pending_verification
            if pending and pending["target"] == target and self._normal(target):
                self._observe("verification", task_id=pending["task_id"], success=True)
                self._pending_verification = None
                self._event["state"] = "verified_closed"
                self._log(f"A: {target} 실제 재검증 성공 → 학습 증거 반영")
                if target == "P" and not self._first_verified:
                    self._first_verified, self._verified_at = True, self.time
                if target == "Q" or self.scenario in {"first_event", "ack_loss"}:
                    self._event["state"] = "complete"
            if not self._normal(target) and self._event["state"] in {"idle", "complete"}:
                self._new_event(target)

    def _tick(self):
        self.time = round(self.time + .05, 10)
        a, b = self._robots["A"], self._robots["B"]
        if not a["path"]:
            a["path"] = [self._patrol[self._patrol_index]]
            self._patrol_index = (self._patrol_index + 1) % len(self._patrol)
        self._advance(a, self.config["a_speed"] * .05)
        reached = self._advance(b, self.config["robot_speed"] * .05)
        for waypoint, heading in reached:
            if waypoint == self._pass_waypoint:
                self._pass_opportunity(heading)
        task = self._receiver_task
        if task and task["phase"] == "travel" and not b["path"]:
            self._at_site()
        elif task and task["phase"] == "working" and self.time >= task["work_until"]:
            self._finish_work()
        elif not b["path"] and b["state"] == "returning":
            b["state"] = "idle"
        for packet in list(self._packets):
            if not packet["processed"] and self.time >= packet["arrival"]:
                packet["processed"] = True
                if not packet["lost"]:
                    self._deliver(packet)
                else:
                    self._log(f"패킷 유실: {packet['label']}")
        self._packets = [p for p in self._packets if self.time < p["arrival"] + .4]
        self._perceive_a()
        if self._task_send_at is not None and self.time >= self._task_send_at:
            self._task_send_at = None
            target = self._event["target"]
            self._transmit(target, ("N",) if target == "P" else ("N", "C"), task=True)
            self._event["state"] = "sent"
        for msg_id, outgoing in list(self._outbox.items()):
            if outgoing["acked"]:
                continue
            if outgoing["retry_at"] is not None:
                if self.time >= outgoing["retry_at"] and outgoing["attempt"] < 8:
                    self.stats["retries"] += 1
                    self._transmit(outgoing["target"], (), task=outgoing["task"], message_id=msg_id)
                continue
            if self.time >= outgoing["deadline"]:
                self._observe("timeout", msg_id=msg_id, attempt_id=outgoing["attempt"])
                outgoing["retry_at"] = self.time + .9
                self._log(f"A: ACK timeout · 수신 여부 불확실 → {outgoing['target']} 확률 갱신")
        for control in list(self._controls.values()):
            if self.time >= control["next"] and control["attempt"] < 8:
                control["attempt"] += 1
                control["next"] = self.time + 1.6
                self.stats["retries"] += 1
                self._enqueue("B", control["kind"], control["data"], control["label"], control["size"])
        if self.scenario in {"pass_clear", "pass_occluded"} and self._first_verified and self._pass_confirmed and not self._second_started and self.time >= self._verified_at + 1:
            self._second_started = True
            obj = self._objects["Q"]
            obj["x"], obj["y"] = self.scene["q_x"]+.75, self.scene["q_y"]
            obj["version"] += 1
            self._event["state"] = "idle"
            self._log("환경: 작업자가 Q를 유사품 쪽으로 밀었습니다. A가 관측할 때까지 추정기는 모릅니다.")

    def step(self, dt):
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not isfinite(dt) or dt < 0:
            raise ValueError("dt must be a finite non-negative number")
        self._remainder += dt
        while self._remainder >= .05 - 1e-10:
            self._remainder -= .05
            self._tick()
        self._remainder = max(0., self._remainder)
        return self.snapshot()

    def move_target(self):
        target = self._event["target"]
        return self.move_object(target)

    def move_object(self, target):
        """Physical intervention; A learns it only on a later observation."""
        obj = self._objects[target]
        direction = 1 if obj["version"] % 2 == 0 else -1
        obj["x"], obj["y"] = obj["normal_x"] + .85*direction, obj["normal_y"] + .35
        obj["version"] += 1
        self._log(f"환경: {target} 수동 재이동 · 실제 C v{obj['version']} (A는 관측 전까지 모름)")
        return self.snapshot()

    def change_appearance(self, target):
        """Abstract content replacement, without simulating real RGB recognition."""
        self._objects[target]["appearance_version"] += 1
        self._log(f"환경: {target} 외형 변경 (A는 관측 전까지 모름)")
        return self.snapshot()

    def reset_receiver_memory(self, notify=True):
        if not isinstance(notify, bool):
            raise ValueError("notify must be bool")
        self._memory_epoch += 1
        self._cache = {target: {} for target in self._objects}
        self._seen_messages.clear()
        self._log(f"B: 영속 캐시 초기화 · 실제 memory_epoch {self._memory_epoch}")
        if notify:
            self._control("STATUS", {"epoch": self._memory_epoch}, f"STATUS epoch {self._memory_epoch}", 40)
        if self._receiver_task and self._receiver_task["phase"] not in {"done"}:
            self._receiver_task["waiting_for"] = None
            self._request(("N",))
        return self.snapshot()

    def set_hidden_occlusion(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be bool")
        self._hidden_occlusion = enabled
        self._log("환경: 숨은 가림 설치" if enabled else "환경: 숨은 가림 제거")
        return self.snapshot()

    def send_info(self, kinds, *, target=None):
        try:
            values = tuple(dict.fromkeys(kinds))
        except TypeError as exc:
            raise ValueError("kinds must be a non-empty iterable of N, C, I") from exc
        if not values or any(not isinstance(value, str) or len(value) != 1 or value not in "NCI" for value in values):
            raise ValueError("kinds must be a non-empty iterable of N, C, I")
        selected = self._event["target"] if target is None else target
        if selected not in self._objects:
            raise ValueError("unknown target")
        self._transmit(selected, values)
        return self.snapshot()

    def export_records(self):
        return deepcopy(self.decision_records)

    def snapshot(self):
        beliefs = {}
        for target, obj in self._objects.items():
            beliefs[target] = self.estimator.snapshot(target, obj["site_id"], now=self.time).as_policy_input()
        packets = []
        for packet in self._packets:
            progress = min(1., max(0., (self.time-packet["sent"])/(packet["arrival"]-packet["sent"])))
            packets.append({
                "id": packet["id"], "from": packet["from"], "to": packet["to"],
                "x": packet["start"][0] + progress*(packet["end"][0]-packet["start"][0]),
                "y": packet["start"][1] + progress*(packet["end"][1]-packet["start"][1]),
                "progress": progress, "label": packet["label"], "bytes": packet["bytes"], "lost": packet["lost"],
            })
        objects = [{**obj, "state": "normal" if self._normal(target) else "displaced"} for target, obj in self._objects.items()]
        if self.scenario in {"pass_clear", "pass_occluded"}:
            objects.append({"id": "Q′", "site_id": "S2", "x": self.scene["q_x"]+.45, "y": self.scene["q_y"], "normal_x": self.scene["q_x"]+.45, "normal_y": self.scene["q_y"], "version": 1, "state": "normal", "lookalike": True})
        return {
            "time": self.time, "scenario": self.scenario, "seed": self.seed,
            "world": {"width": 12, "height": 8}, "robots": deepcopy(self._robots),
            "objects": objects, "network": packets, "beliefs": beliefs,
            "receiver": {"memory_epoch": self._memory_epoch, "known_epoch": self.estimator.memory_epoch, "knowledge": {target: self._truth(target) for target in self._objects}},
            "event": dict(self._event), "stats": dict(self.stats), "logs": list(self.logs),
            "occlusion": self._hidden_occlusion,
            "occluders": [{"min_x": 5.0, "min_y": 2.8, "max_x": 5.25, "max_y": 4.4, "known_to_a": False}] if self._hidden_occlusion else [],
        }
