"""2D world: A patrols an ellipse line, B drives a trunk+branch line, a worker fixes objects on B's text.

Units: metres and seconds. Every tag frame is aligned with the world axes: +x = right, +y = forward (away from B's
side). No physics: robots move along their lines at constant speed, objects are moved by code.
Perception error models (position error, class accuracy, image matching) are the same placeholders as rl/env.py;
geometry here only supplies their inputs (distance, occlusion) and decides what B sees while driving.
"""
import math
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
from features import CUR, ID, IMG, ITEMS, NEED_ID, NEED_NORMAL, NORM, OK, BKnowledge, History, items_of  # noqa: E402

LAYOUT = dict(
    size=(2.0, 1.6), car=(0.20, 0.15), tag=0.07, dt=0.2, horizon=300.0,
    ellipse=((1.0, 0.8), 0.80, 0.60),            # A's line: centre, semi-axes (x, y); A drives counter-clockwise
    a_speed=0.15, b_speed=0.15,
    fov=math.radians(87), depth=(0.40, 1.20),    # D455, both robots
    obs_dist=(0.45, 1.20), obs_angle=math.radians(35), obs_stop=1.5,
    spots=[(0.60, 0.95), (0.90, 0.65), (1.20, 0.95), (1.50, 0.65)],
    spot_r=0.12, obj_r=0.015, obj_front=0.08, pair_dx=0.02,   # pair: 4 cm apart (scenario doc)
    home=(0.15, 0.07), trunk_y=0.07, stop_before=0.45,
    occluder=((1.05, 0.35), (1.05, 0.55)),       # 20 cm, blocks B's view of S1 from the S2 branch
    anomaly_every=40.0,
    # ponytail: worker and recognition numbers are guesses; replace with measurements
    worker_delay=(3.0, 6.0), p_wrong=0.2, wrong_off=(0.03, 0.08), remind_after=10.0, pos_tol=0.03,
    p_overconf=0.2, overconf=3.0,                # A trusts a sigma 3x too small this often (scenario 4(b))
    q_learn=0.9,                                 # P(B learns the right appearance when it sees/handles an object)
    p_frame=0.15,                                # P(B recognises a spot in one frame while it is in view)
    req_timeout=3.0,
)
COLORS = ["red", "green", "blue", "yellow"]
COLORS_KO = {"red": "빨간", "green": "초록", "blue": "파란", "yellow": "노란"}
ATYPES = ["misplaced", "toppled", "wrong part"]
TOPPLED, WRONG = 1, 2
N_ACTIONS, FULL = 17, 16
WRONG_OP = 5.0  # penalty when B confidently points the worker at the wrong object
BASE_BYTES, ITEM_BYTES, REPLY_BYTES = 60, {ID: 40, CUR: 60, NORM: 60, IMG: 30_000}, 50
CONDITIONS = {
    "base":     dict(occluder=False, bw=(200, 1000), loss=(0, .02)),
    "occluded": dict(occluder=True,  bw=(200, 1000), loss=(0, .02)),   # A's visibility table does not know it
    "limited":  dict(occluder=False, bw=(5, 50),     loss=(.1, .3)),
}


# ---------- geometry ----------
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def seg_hit(p, q, a, b):
    """Do segments pq and ab cross?"""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    return cross(a, b, p) * cross(a, b, q) < 0 and cross(p, q, a) * cross(p, q, b) < 0


def seg_near(p, q, c, r):
    """Does segment pq pass within r of point c?"""
    p, q, c = np.asarray(p), np.asarray(q), np.asarray(c)
    d = q - p
    t = np.clip(np.dot(c - p, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return np.hypot(*(p + t * d - c)) < r


def in_view(pos, heading, target):
    v = np.subtract(target, pos)
    d = math.hypot(*v)
    lo, hi = LAYOUT["depth"]
    return lo <= d <= hi and abs(wrap(math.atan2(v[1], v[0]) - heading)) <= LAYOUT["fov"] / 2


def sees_spot(pos, heading, j, occluder):
    s = LAYOUT["spots"][j]
    return in_view(pos, heading, s) and not (occluder and seg_hit(pos, s, *LAYOUT["occluder"]))


def ellipse_pose(phi):
    """A's position, heading and camera direction (left of heading = inside the ellipse)."""
    (cx, cy), ax, ay = LAYOUT["ellipse"]
    pos = np.array([cx + ax * math.cos(phi), cy + ay * math.sin(phi)])
    heading = math.atan2(ay * math.cos(phi), -ax * math.sin(phi))
    return pos, heading, heading + math.pi / 2


def object_slots(k):
    """Every place an object of spot k can sit when normal (single centre, or the pair's two sides)."""
    tx, ty = LAYOUT["spots"][k]
    y, dx = ty + LAYOUT["obj_front"], LAYOUT["pair_dx"]
    return [(tx, y), (tx - dx, y), (tx + dx, y)]


@lru_cache(None)
def obs_points():
    """phi where A stops for each spot: the closest point on the ellipse from which every object slot is
    >= 0.45 m away (D455 minimum), inside the camera's view, and not hidden behind another spot's objects."""
    others = lambda k: [s for j in range(len(LAYOUT["spots"])) if j != k for s in object_slots(j)]
    out = []
    for k, tag in enumerate(LAYOUT["spots"]):
        best = None
        for phi in np.linspace(0, 2 * math.pi, 720, endpoint=False):
            pos, _, cam = ellipse_pose(phi)
            slots = object_slots(k)
            if min(math.hypot(*np.subtract(s, pos)) for s in slots) < LAYOUT["obs_dist"][0]:
                continue
            if not all(in_view(pos, cam, s) and abs(wrap(math.atan2(s[1] - pos[1], s[0] - pos[0]) - cam))
                       <= LAYOUT["obs_angle"] for s in slots):
                continue
            if any(seg_near(pos, s, o, LAYOUT["obj_r"] + 0.01) for s in slots for o in others(k)):
                continue
            far = max(math.hypot(*np.subtract(s, pos)) for s in slots)
            if best is None or far < best[0]:
                best = (far, phi)
        assert best, f"no observation point for spot {k}; move it or resize the ellipse"
        out.append(best[1])
    return tuple(out)


def _leg(a, b, heading):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, round(math.hypot(*(b - a)) / (LAYOUT["b_speed"] * LAYOUT["dt"])))
    return [(a + (b - a) * i / n, heading) for i in range(1, n + 1)]


@lru_cache(None)
def routes(occluder):
    """Per spot k: B's frames out (home -> trunk -> up the branch) and back (reverse down, trunk to home).
    Each frame = (pos, heading, spots in view). Cached: the layout is static."""
    sp, home, ty = LAYOUT["spots"], LAYOUT["home"], LAYOUT["trunk_y"]
    res = []
    for k, (sx, sy) in enumerate(sp):
        stop = (sx, sy - LAYOUT["stop_before"])
        out = _leg(home, (sx, ty), 0.0) + _leg((sx, ty), stop, math.pi / 2)
        back = _leg(stop, (sx, ty), math.pi / 2) + _leg((sx, ty), home, math.pi)  # backs out of the branch
        tag = lambda legs: [(p, h, frozenset(j for j in range(len(sp)) if j != k and sees_spot(p, h, j, occluder)))
                            for p, h in legs]
        res.append((tag(out), tag(back)))
    return tuple(res)


@lru_cache(None)
def visibility_table():
    """A's table: P(B recognises spot j on a round trip to k). Built from the line layout, so it knows no occluder."""
    q = LAYOUT["p_frame"]
    table = []
    for out, back in routes(False):
        n = {}
        for _, _, vis in out + back:
            for j in vis:
                n[j] = n.get(j, 0) + 1
        table.append({j: 1 - (1 - q) ** c for j, c in n.items()})
    return tuple(table)


# ---------- world ----------
class Obj:
    def __init__(self, normal, cls, shape):
        self.normal, self.pos = np.array(normal, float), np.array(normal, float)
        self.cls, self.color, self.shape, self.toppled = cls, cls, shape, False

    def deviates(self):
        return self.toppled or self.color != self.cls or math.hypot(*(self.pos - self.normal)) > LAYOUT["pos_tol"]


class Spot:
    def __init__(self, k, rng, pair):
        centre, left, right = object_slots(k)
        self.pair, self.identical = pair, bool(pair and rng.random() < 0.5)
        if pair:  # same colour; identical copies, or same colour but a different shape
            shapes = ("o", "o") if self.identical else ("o", "s")
            self.objs = [Obj(left, COLORS[k], shapes[0]), Obj(right, COLORS[k], shapes[1])]
        else:
            self.objs = [Obj(centre, COLORS[k], "o")]

    @property
    def n_similar(self):
        return len(self.objs)

    @property
    def spacing_cm(self):
        return 2 * LAYOUT["pair_dx"] * 100 if self.pair else 20.0


class Anomaly:
    def __init__(self, k, obj, atype, t):
        self.spot, self.obj, self.atype, self.t_event = k, obj, atype, t
        self.detected, self.hist, self.job, self.b_saw = False, None, None, False


def describe(act):
    return "nothing" if act == 0 else "base" + "".join(" + " + ITEMS[i] for i in sorted(items_of(act)))


class World:
    def __init__(self, rng, condition="base", alpha=0.1, beta=1.0, gamma=1.0, trace=False):
        L, c = LAYOUT, CONDITIONS[condition]
        self.rng, self.cond, self.trace = rng, condition, trace
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.occluder = c["occluder"]
        self.bw, self.loss = rng.uniform(*c["bw"]), rng.uniform(*c["loss"])
        pairs = set(rng.choice(len(L["spots"]), 2, replace=False))  # 2 of the 4 spots hold two objects
        self.spots = [Spot(k, rng, k in pairs) for k in range(len(L["spots"]))]
        self.know = BKnowledge(len(self.spots))                       # A's estimate of B
        self.b_obj = np.zeros(len(self.spots), bool)                  # B's truth, hidden from A
        self.b_normal = np.zeros(len(self.spots), bool)
        self.b_wrong = np.zeros(len(self.spots), bool)                # B learned the wrong appearance (truth)
        self.t, self.r, self.log = 0.0, 0.0, []
        self.anom, self.inbox, self.queue, self.pending = {}, [], [], []
        self.a_phi, self.a_stop_until = 0.0, 0.0
        self.b_state, self.b_job, self.b_frames, self.b_i = "idle", None, None, 0
        self.b_pose = (np.array(L["home"]), 0.0)
        self.next_event = rng.exponential(L["anomaly_every"])
        self.next_tick5 = 5.0
        self.stats = dict(condition=condition, anomalies=0, fixed=0, times=[], bytes=0, msgs=0, images=0,
                          redundant=0, requests=0, reminders=0, wrong_ops=0, know_err=[])

    # ----- bookkeeping -----
    def _log(self, who, msg):
        if self.trace:
            self.log.append(f"{self.t:6.1f}s  {who:<2} {msg}")

    def _spend(self, n):
        self.stats["bytes"] += n
        self.r -= self.alpha * n / 1000

    def _send(self, n, image):
        self._spend(n)
        self.stats["msgs"] += 1
        self.stats["images"] += image

    def _delivered(self):
        return self.rng.random() >= self.loss

    def _reply(self):
        """B -> A message (ACK / info request / result): costs bytes, may be lost."""
        self._spend(REPLY_BYTES)
        return self._delivered()

    def _delay(self, n):
        return n / 1000 / self.bw  # seconds

    def take_reward(self):
        r, self.r = self.r, 0.0
        return r

    @property
    def done(self):
        return self.t >= LAYOUT["horizon"]

    # ----- A's decision (called by the env / viewer) -----
    def decide(self, act):
        a = self.pending.pop(0)
        k = a.spot
        self.stats["know_err"].append(abs(self.know.normal[k] - self.b_normal[k]))
        self._log("A", f"spot {k}: {ATYPES[a.atype]} | B knows obj {self.know.obj[k]:.2f} "
                       f"normal {self.know.normal[k]:.2f} current {a.hist.est_current:.2f} | send: {describe(act)}")
        if not act:
            return
        redundant = a.job is not None and not a.job["failed"]
        self.stats["redundant"] += redundant
        self.r -= self.gamma * redundant
        items = items_of(act)
        n = BASE_BYTES + sum(ITEM_BYTES[i] for i in items)
        if self.trace:  # what the A -> B message would carry (positions in the tag frame, cm)
            o, tag = a.obj, np.array(LAYOUT["spots"][k])
            cm = lambda p: f"({(p[0] - tag[0]) * 100:+.0f}, {(p[1] - tag[1]) * 100:+.0f})"
            body = {ID: f"id={o.cls} {'circle' if o.shape == 'o' else 'square'}", CUR: f"cur={cm(a.cur_rep)}±{a.pos_err:.0f}",
                    NORM: f"norm={cm(o.normal)}", IMG: "img=crop 30KB"}
            self._log("", f"  msg {{spot={k}, {ATYPES[a.atype]}" + "".join(", " + body[i] for i in sorted(items))
                      + f"}} {n} B, {self._delay(n) * 1000:.0f} ms")
        self._send(n, IMG in items)
        a.hist.on_send(items)
        if self._delivered():
            self.inbox.append((self.t + self._delay(n), "cmd", a, items))
        else:
            self._log("", f"  message lost ({n} B)")

    def run_until_decision(self):
        """Tick until A needs a decision; returns True when the patrol is over."""
        while not self.pending:
            if self.done:
                self.r -= 10 * sum(a.detected for a in self.anom.values())  # unresolved at the end
                self._log("--", f"patrol over, unresolved at spots {sorted(k for k, a in self.anom.items() if a.detected)}")
                return True
            self.tick()
        return False

    # ----- one time step -----
    def tick(self):
        L, dt = LAYOUT, LAYOUT["dt"]
        self.t += dt
        self.r -= self.beta * sum(a.detected for a in self.anom.values()) * dt / 5  # waiting cost, per 5 s
        if self.t >= self.next_tick5:
            self.next_tick5 += 5.0
            for a in self.anom.values():
                if a.detected:
                    a.hist.tick()
        if self.t >= self.next_event:
            self.next_event = self.t + self.rng.exponential(L["anomaly_every"])
            self._new_anomaly()
        self._tick_a()
        self._tick_inbox()
        self._tick_b()

    def _new_anomaly(self):
        free = [k for k in range(len(self.spots)) if k not in self.anom]
        if not free:
            return
        k = int(self.rng.choice(free))
        obj = self.spots[k].objs[self.rng.integers(len(self.spots[k].objs))]
        atype = int(self.rng.integers(3))
        if atype == TOPPLED:
            obj.toppled = True
        elif atype == WRONG:
            obj.color = COLORS[(COLORS.index(obj.cls) + 1 + self.rng.integers(len(COLORS) - 1)) % len(COLORS)]
        else:
            obj.pos = obj.normal + self._offset(0.04, 0.10)
        self.anom[k] = Anomaly(k, obj, atype, self.t)
        self._log("!", f"spot {k}: {ATYPES[atype]} happens ({self.spots[k].n_similar} object(s)"
                       f"{', identical' if self.spots[k].identical else ''})")

    def _offset(self, lo, hi):
        ang, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(lo, hi)
        return np.array([math.cos(ang), math.sin(ang)]) * d

    # ----- Robot A -----
    def _tick_a(self):
        if self.t < self.a_stop_until:
            return
        (_, ax, ay), phi = LAYOUT["ellipse"], self.a_phi
        speed = math.hypot(ax * math.sin(phi), ay * math.cos(phi))
        new = phi + LAYOUT["a_speed"] * LAYOUT["dt"] / speed
        hit = [k for k, p in enumerate(obs_points())
               if math.floor((phi - p) / (2 * math.pi)) != math.floor((new - p) / (2 * math.pi))]
        self.a_phi = new
        if hit:
            self.a_stop_until = self.t + LAYOUT["obs_stop"]
            for k in hit:
                self._observe(k)

    def visible_fraction(self, cam, obj):
        """Share of 5 rays across the object that reach the camera unblocked."""
        if not in_view(cam[0], cam[2], obj.pos):
            return 0.0
        v = obj.pos - cam[0]
        n = np.array([-v[1], v[0]]) / math.hypot(*v)
        others = [o for s in self.spots for o in s.objs if o is not obj]
        free = 0
        for s in (-0.8, -0.4, 0, 0.4, 0.8):
            tip = obj.pos + n * LAYOUT["obj_r"] * s
            blocked = any(seg_near(cam[0], tip, o.pos, LAYOUT["obj_r"]) for o in others) or (
                self.occluder and seg_hit(cam[0], tip, *LAYOUT["occluder"]))
            free += not blocked
        return free / 5

    def _observe(self, k):
        a = self.anom.get(k)
        if a is None or not a.obj.deviates():
            return
        cam = ellipse_pose(self.a_phi)
        vis = self.visible_fraction(cam, a.obj)
        if vis < 0.3:
            self._log("A", f"spot {k}: anomaly hidden (visible {vis:.0%}), missed")
            return
        if not a.detected:  # perception fixed per anomaly, as in rl/env.py
            r, s = self.rng, self.spots[k]
            a.detected, a.hist, a.occ = True, History(self.t), 1 - vis
            dist = math.hypot(*(a.obj.pos - cam[0]))
            a.pos_err = 1 + 1.5 * dist + 4 * a.occ + abs(r.normal(0, 0.5))  # sigma A reports (cm)
            true_err = a.pos_err * (LAYOUT["overconf"] if r.random() < LAYOUT["p_overconf"] else 1)
            a.cur_rep = a.obj.pos + r.normal(0, true_err / 100, 2)  # the "cur" A would send
            acc = np.clip(0.98 - 0.4 * a.occ - 0.35 * (s.pair and not s.identical), 0.05, 1)
            a.class_ok, a.conf = r.random() < acc, float(np.clip(acc + r.normal(0, 0.08), 0, 1))
            a.img_match = r.random() < 0.92 - 0.4 * a.occ
            self.stats["anomalies"] += 1
        self.pending.append(a)

    # ----- channel -----
    def _tick_inbox(self):
        for msg in [m for m in self.inbox if m[0] <= self.t]:
            self.inbox.remove(msg)
            _, kind, a, items = msg
            if self.anom.get(a.spot) is not a:
                continue  # already fixed
            self._learn(a, items)
            if kind == "resp":
                if self.b_job and self.b_job[0] is a:
                    self.b_job[1]["items"] |= items
                continue
            if self._reply():  # ACK
                a.hist.on_ack()
                self.know.on_ack(a.spot, items)
            if a.job is None or a.job["failed"]:
                a.job = dict(items=set(items), failed=False, started=False, asked=False)
                self.queue.append(a)
            elif not a.job["started"]:
                a.job["items"] |= items  # more info arrived before B left

    def _learn(self, a, items):
        if ID in items:  # A's description; wrong if A misclassified
            self.b_obj[a.spot], self.b_wrong[a.spot] = True, not a.class_ok
        if NORM in items:
            self.b_normal[a.spot] = True

    # ----- Robot B -----
    def _drive(self):
        """Move one frame; returns True at the end of the leg. Recognises spots in view (truth, hidden from A)."""
        pos, heading, vis = self.b_frames[self.b_i]
        self.b_pose = (pos, heading)
        for j in vis:
            if self.rng.random() < LAYOUT["p_frame"]:
                if not self.b_obj[j]:
                    self.b_obj[j], self.b_wrong[j] = True, self.rng.random() > LAYOUT["q_learn"]
                if j in self.anom:
                    self.anom[j].b_saw = True
                else:
                    self.b_normal[j] = True
                self.b_seen.add(j)
        self.b_i += 1
        return self.b_i >= len(self.b_frames)

    def _pick(self, a, items):
        """Which object B would point the worker at, or None when B is unsure (then it asks A).
        Can be confidently wrong: A's sigma too small, A's wrong description, or B's wrongly learned appearance."""
        s = self.spots[a.spot]
        if a.atype in (TOPPLED, WRONG) or s.n_similar == 1 or a.b_saw:  # odd one out (pairs share a colour)
            return a.obj
        if IMG in items and a.img_match:
            return a.obj
        if CUR in items:  # B only considers objects within 2 sigma of the reported position
            near = [o for o in s.objs if math.hypot(*(o.pos - a.cur_rep)) <= 2 * a.pos_err / 100]
            if len(near) == 1:
                return near[0]
        if not s.identical:  # shapes differ: a description or B's own memory of the look tells them apart
            other = next(o for o in s.objs if o is not a.obj)
            if ID in items:
                return a.obj if a.class_ok else other
            if self.b_obj[a.spot]:
                return other if self.b_wrong[a.spot] else a.obj
        return None

    def _missing(self, a, job):
        miss = set()
        if self._pick(a, job["items"]) is None:
            miss.add(IMG)
        if NORM not in job["items"] and not self.b_normal[a.spot]:
            miss.add(NORM)
        return miss

    def _instruction(self, a, target):
        k, o = a.spot, a.obj
        side = ("왼쪽 " if target.normal[0] < LAYOUT["spots"][k][0] else "오른쪽 ") if self.spots[k].pair else ""
        name = f"{side}{COLORS_KO[o.cls]} 물건"
        if a.atype == TOPPLED:
            return f"스팟 {k}에서 넘어진 {name}을 세워서 제자리에 놓아주세요"
        if a.atype == WRONG:
            return f"스팟 {k}의 {COLORS_KO[o.color]} 물건을 {name}으로 바꿔주세요"
        dx, dy = (o.normal - o.pos) * 100
        parts = [f"{'오른쪽' if dx > 0 else '왼쪽'}으로 {abs(dx):.0f}cm" if abs(dx) >= 1 else "",
                 f"{'앞쪽' if dy > 0 else '뒤쪽'}으로 {abs(dy):.0f}cm" if abs(dy) >= 1 else ""]
        return f"스팟 {k}에서 {name}을 " + ", ".join(p for p in parts if p) + " 옮겨주세요"

    def _worker_fix(self, a, job):
        if job.get("undo") is not None:  # put back the object moved by mistake
            u, job["undo"] = job["undo"], None
            u.pos = u.normal + self.rng.normal(0, 0.005, 2)
        if job["target"] is not a.obj:  # worker follows the instruction on the wrong object
            job["target"].pos = job["target"].pos + (a.obj.normal - a.obj.pos)
            self._log("W", f"spot {a.spot}: worker moved the object B pointed at (the wrong one)")
            return
        o = a.obj
        o.toppled, o.color = False, o.cls
        if self.rng.random() < LAYOUT["p_wrong"]:
            o.pos = o.normal + self._offset(*LAYOUT["wrong_off"])
            self._log("W", f"spot {a.spot}: worker moved it, but to the wrong place")
        else:
            o.pos = o.normal + self.rng.normal(0, 0.005, 2)
            self._log("W", f"spot {a.spot}: worker fixed it")

    def _tick_b(self):
        L, st = LAYOUT, self.b_state
        if st == "idle":
            if not self.queue:
                return
            a = self.queue.pop(0)
            job = a.job
            job.update(started=True, t_start=self.t)
            self.b_job, self.b_seen = (a, job), set()
            self.b_frames, self.b_i, self.b_state = routes(self.occluder)[a.spot][0], 0, "out"
            self._log("B", f"drives standby -> spot {a.spot} with {sorted(ITEMS[i] for i in job['items'])}")
            return
        a, job = self.b_job
        if st == "out":
            if self._drive():
                self._log("B", f"arrives at spot {a.spot}, passed and saw spots {sorted(self.b_seen)} (hidden from A)")
                self._check(a, job)
        elif st == "wait":
            if IMG in job["items"] or NORM in job["items"] or self.t >= job["wait_until"]:
                if not self._missing(a, job) or self.t >= job["wait_until"]:
                    self._check(a, job)
        elif st == "watch":
            if not any(o.deviates() for o in self.spots[a.spot].objs):
                self.anom.pop(a.spot)
                if not self.b_obj[a.spot]:
                    self.b_obj[a.spot], self.b_wrong[a.spot] = True, self.rng.random() > L["q_learn"]
                self.b_normal[a.spot] = True
                self.stats["fixed"] += 1
                self.stats["times"].append(self.t - a.hist.t0)
                self.r += 10
                job["reason"] = OK
                self._log("B", f"spot {a.spot}: fix confirmed, driving back")
                self._go_back(a)
            elif job["target"] is not a.obj and job["fixed_at"] < math.inf:
                self._wrong_op(a, job)
            elif self.t >= job["worker_at"]:
                job["worker_at"], job["fixed_at"] = math.inf, self.t
                self._worker_fix(a, job)
            elif self.t - job["fixed_at"] >= L["remind_after"]:
                self.stats["reminders"] += 1
                self._log("B", "-> 노트북: 물건 위치를 수정해주세요")
                job["fixed_at"], job["worker_at"] = math.inf, self.t + self.rng.uniform(*L["worker_delay"])
        elif st == "back":
            if self._drive():
                self.b_state, self.b_job = "idle", None
                self._log("B", f"back at standby, saw spots {sorted(self.b_seen)} on the trip (hidden from A)")
                if self._reply():  # result carries the trip, so A can update what B may have seen
                    a.hist.on_result(job["reason"])
                    if job["reason"] == OK:
                        self.know.on_success(a.spot)
                    seen = {k: x.hist for k, x in self.anom.items() if x.detected and x.hist.t0 <= job["t_start"]}
                    self.know.on_pass(visibility_table()[a.spot], seen)

    def _check(self, a, job):
        """At the spot: ask A once for anything missing, then instruct the worker or give up."""
        miss = self._missing(a, job)
        if miss and not job["asked"]:
            job["asked"], job["wait_until"], self.b_state = True, self.t + LAYOUT["req_timeout"], "wait"
            self.stats["requests"] += 1
            self._log("B", f"spot {a.spot}: missing {[ITEMS[i] for i in sorted(miss)]} -> asks A")
            if self._reply():
                n = BASE_BYTES + sum(ITEM_BYTES[i] for i in miss)
                self._send(n, IMG in miss)
                a.hist.items |= miss
                if self._delivered():
                    self.inbox.append((self.t + self._delay(n), "resp", a, miss))
            return
        if miss:
            job["reason"], job["failed"] = (NEED_ID if IMG in miss else NEED_NORMAL), True
            self.r -= 3  # wasted trip; must stay < unresolved penalty or sending nothing wins
            self._log("B", f"spot {a.spot}: cannot instruct ({['ok', 'need id', 'need normal'][job['reason']]}), driving back")
            self._go_back(a)
            return
        job["target"] = self._pick(a, job["items"])
        self._log("B", "-> 노트북: " + self._instruction(a, job["target"]))
        job.update(worker_at=self.t + self.rng.uniform(*LAYOUT["worker_delay"]), fixed_at=math.inf)
        self.b_state = "watch"

    def _wrong_op(self, a, job):
        """B's re-check finds the wrong object moved: void B's look of this spot, get A's crop, redo (TASK v2)."""
        self.stats["wrong_ops"] += 1
        self.r -= WRONG_OP
        self.b_obj[a.spot] = self.b_wrong[a.spot] = False
        delay = 0.0
        if IMG not in job["items"]:
            self.stats["requests"] += 1
            if self._reply():
                n = BASE_BYTES + ITEM_BYTES[IMG]
                self._send(n, True)
                a.hist.items |= {IMG}
                delay = self._delay(n)  # ponytail: a lost crop is ignored; the worker compares with the shelf
            job["items"].add(IMG)
        job.update(undo=job["target"], target=a.obj, fixed_at=math.inf,
                   worker_at=self.t + delay + self.rng.uniform(*LAYOUT["worker_delay"]))
        self._log("B", f"spot {a.spot}: re-check -> wrong object was moved (TASK v2, with A's crop)")
        self._log("B", "-> 노트북: 방금 옮긴 물건을 제자리로 되돌리고, " + self._instruction(a, a.obj) + " (사진 첨부)")

    def _go_back(self, a):
        self.b_frames, self.b_i, self.b_state = routes(self.occluder)[a.spot][1], 0, "back"


if __name__ == "__main__":
    # self-check: geometry and the worker loop
    L = LAYOUT
    for k, phi in enumerate(obs_points()):
        pos, _, _ = ellipse_pose(phi)
        assert math.hypot(*np.subtract(L["spots"][k], pos)) >= 0.4, k
    seen = lambda occl, k, j: sum(j in v for o_b in routes(occl)[k] for _, _, v in o_b)  # frames, out + back
    assert seen(False, 2, 1) > 0 and seen(False, 2, 3) > 0, "S2 trip should show S1 and S3"
    assert seen(True, 2, 1) < seen(False, 2, 1) / 2, "occluder should hide most of S1 on the S2 trip"
    assert visibility_table()[2][1] > 0.9  # A's table does not know about the occluder

    def scenario(k, shift, act, pair=False, pos_err=2, cur_rep=None):
        """One misplaced object at spot k (left one if pair), A sends act; run until B is home again."""
        w = World(np.random.default_rng(0), "base", trace=True)
        w.next_event = math.inf
        if pair:
            w.spots[k] = Spot(k, w.rng, True)
            w.spots[k].identical = True
        o = w.spots[k].objs[0]
        o.pos = o.normal + np.array(shift)
        a = w.anom[k] = Anomaly(k, o, 0, 0.0)
        a.detected, a.hist, a.pos_err, a.class_ok, a.img_match, a.conf, a.occ = True, History(0), pos_err, True, True, .9, 0
        a.cur_rep = o.pos if cur_rep is None else cur_rep
        w.pending.append(a)
        w.decide(act)
        for _ in range(600):
            w.tick()
            if any("수정해주세요" in s for s in w.log):
                L["p_wrong"] = 0.0
            if w.b_state == "idle" and w.stats["fixed"]:
                break
        return w

    # worker gets it wrong once: B keeps watching, reminds after 10 s, then confirms and drives home
    p_wrong, L["p_wrong"] = L["p_wrong"], 1.0
    w = scenario(1, [0.06, 0.0], FULL)
    assert any("왼쪽으로 6cm" in s for s in w.log), w.log
    assert w.stats["reminders"] >= 1 and w.stats["fixed"] == 1 and w.b_state == "idle", w.stats

    # scenario 4(b): identical pair, A trusts sigma 1 cm but its "cur" lands on the other object -> B points at the
    # wrong one, the re-check catches it, A's crop is fetched (TASK v2), then the fix is confirmed
    L["p_wrong"] = 0.0
    w2 = scenario(2, [0.0, 0.06], 1 + 2 + 4, pair=True, pos_err=1, cur_rep=np.array(object_slots(2)[2]))
    assert w2.stats["wrong_ops"] == 1 and w2.stats["images"] == 1 and w2.stats["fixed"] == 1, (w2.stats, w2.log)
    assert any("오른쪽 파란 물건" in s for s in w2.log) and w2.b_state == "idle", w2.log
    L["p_wrong"] = p_wrong
    print("ok | observation phi:", [round(p, 2) for p in obs_points()],
          "| S2 trip sees S1 frames:", seen(False, 2, 1), "-> with occluder:", seen(True, 2, 1))
    print("\n".join(w.log + ["-- wrong object --"] + w2.log))
