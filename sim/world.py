"""2D world: A patrols an ellipse line, B drives a trunk+branch line, a worker fixes objects on B's text.

Units: metres and seconds. Every tag frame is aligned with the world axes: +x = right, +y = forward (away from B's
side). No physics: robots move along their lines at constant speed, objects are moved by code.
Perception error models (position error, class accuracy, image matching) are placeholders to replace with
measurements; geometry here only supplies their inputs (distance, occlusion) and decides what B sees while driving.

Scenario doc (적응형 통신 시나리오.html) coverage:
1(a) event gate, E+N, A re-verifies before trusting B's ok    1(b) ACK lost -> same-message retransmit
2(a) reuse      2(b) B memory reset (epoch, STATUS lost, needs_info at ACK)    2(c) look-alike, appearance memory
2(d) wrongly learned appearance -> unreported wrong object -> A re-check -> knowledge void -> TASK v2
3A pass-by normal evidence held until A sees the spot normal again (dropped if abnormal)    3B pass-by appearance
4(a) look-alike needs crop   4(b) A's sigma too small   4(c) A unsure of the target -> re-observe from the 2nd view
4(d) object moved again: A sees it -> POSE_UPDATE; B cannot find it -> not_found -> A re-observes -> POSE_UPDATE
"""
import math
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
from features import (CUR, ID, IMG, ITEMS, MODES, NEED_ID, NEED_NORMAL, NORM, OK, BKnowledge,  # noqa: E402
                      History, items_of)
# A's estimate of what B knows = knowledge_estimation (KnowledgeEstimator), parameters measured in this world
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge_estimation.integration.sim2d import (CoreKnowledge, load_caps, load_parameters,  # noqa: E402
                                                    load_score_model)

CORE_PARAMETERS = load_parameters()  # knowledge_estimation/integration/sim2d_parameters_v1.json, or None
SCORE_MODEL = load_score_model()     # its b_cal score calibration, or None (b_cal = b_raw)
CORE_CAPS = load_caps()              # b_safe caps chosen on validation seeds, or {} (b_safe = b_cal)
BASELINES = ("cache", "cache_result", "optimistic")  # Notion section 10 own baselines / ablation, comparison only

LAYOUT = dict(
    size=(2.0, 1.6), car=(0.20, 0.15), tag=0.07, dt=0.2, horizon=300.0,
    ellipse=((1.0, 0.8), 0.80, 0.60),            # A's line: centre, semi-axes (x, y); A drives counter-clockwise
    a_speed=0.15, b_speed=0.15,
    fov=math.radians(87), depth=(0.40, 1.20),    # D455, both robots
    obs_dist=(0.45, 1.20), obs_angle=math.radians(35), obs_stop=1.5, obs_gap=0.6,
    spots=[(0.60, 0.95), (0.90, 0.65), (1.20, 0.95), (1.50, 0.65)],
    spot_r=0.12, obj_r=0.015, obj_front=0.08, pair_dx=0.02,   # pair: 4 cm apart, differ only by a sticker
    home=(0.15, 0.07), trunk_y=0.07, stop_before=0.45,
    occluder=((1.05, 0.35), (1.05, 0.55)),       # 20 cm, blocks B's view of S1 from the S2 branch
    anomaly_every=40.0,
    # ponytail: worker, recognition and protocol numbers are guesses; replace with measurements
    worker_delay=(3.0, 6.0), p_wrong=0.2, wrong_off=(0.03, 0.08), remind_after=10.0, pos_tol=0.03,
    p_overconf=0.2, overconf=3.0,                # A trusts a sigma 3x too small this often (4(b))
    q_learn=0.9,                                 # P(B learns the right appearance when it sees/handles an object)
    p_frame=0.15,                                # P(B recognises a spot in one frame while it is in view)
    p_sticker=(0.6, 0.95),                       # P(A reads the sticker -> target confirmed) from view 1 / view 2 (4(c))
    req_timeout=3.0, ack_timeout=2.0, max_retx=2,  # 1(b)
    reset_every=600.0, status_loss=0.5,          # 2(b): B memory reset, its STATUS lost this often (failure injection)
    p_move_again=0.15, move_off=(0.03, 0.08), pose_wait=40.0,  # 4(d)
    deadline=60.0,                               # metric: detected -> verified within this
)
COLORS = ["red", "green", "blue", "yellow"]
COLORS_KO = {"red": "빨간", "green": "초록", "blue": "파란", "yellow": "노란"}
ATYPES = ["misplaced", "toppled", "wrong part"]
MISPLACED, TOPPLED, WRONG = 0, 1, 2
N_ACTIONS, FULL = 17, 16
WRONG_OP = 12.0  # penalty when B confidently points the worker at the wrong object
REQUEST, REOBSERVE = 1.0, 2.0  # per request round trip / per re-observation A must make (scenario doc cost weights)
BASE_BYTES, ITEM_BYTES, REPLY_BYTES = 60, {ID: 40, CUR: 60, NORM: 60, IMG: 30_000}, 50
V2_EXTRA = ITEM_BYTES[NORM] + ITEM_BYTES[CUR]  # TASK v2 also carries the other object's normal + current pose
NOT_FOUND = "not_found"
CONDITIONS = {
    "base":     dict(occluder=False, bw=(200, 1000), loss=(0, .02)),
    "occluded": dict(occluder=True,  bw=(200, 1000), loss=(0, .02)),   # A's visibility table does not know it
    "limited":  dict(occluder=False, bw=(5, 50),     loss=(.1, .3)),
    # scenario doc 2(c), 3B, 4: every spot holds a sticker-only look-alike pair, anomalies are displacements
    "lookalike":          dict(occluder=False, bw=(200, 1000), loss=(0, .02), pairs=4, atypes=(0,)),
    "lookalike_occluded": dict(occluder=True,  bw=(200, 1000), loss=(0, .02), pairs=4, atypes=(0,)),
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


@lru_cache(None)
def obs_points2():
    """A second stop per spot (4(c): re-observe from another view): the best point at least obs_gap rad away from
    the first one, under the same rules. None when the layout has no such point."""
    first = obs_points()
    out = []
    for k in range(len(LAYOUT["spots"])):
        best = None
        for phi in np.linspace(0, 2 * math.pi, 720, endpoint=False):
            if abs(wrap(phi - first[k])) < LAYOUT["obs_gap"]:
                continue
            pos, _, cam = ellipse_pose(phi)
            slots = object_slots(k)
            if min(math.hypot(*np.subtract(s, pos)) for s in slots) < LAYOUT["obs_dist"][0]:
                continue
            if not all(in_view(pos, cam, s) and abs(wrap(math.atan2(s[1] - pos[1], s[0] - pos[0]) - cam))
                       <= LAYOUT["obs_angle"] for s in slots):
                continue
            far = max(math.hypot(*np.subtract(s, pos)) for s in slots)
            if best is None or far < best[0]:
                best = (far, phi)
        out.append(None if best is None else best[1])
    return tuple(out)


# ---------- world ----------
class Obj:
    def __init__(self, normal, cls, sticker=""):
        self.normal, self.pos = np.array(normal, float), np.array(normal, float)
        self.cls, self.color, self.shape, self.toppled, self.sticker = cls, cls, "o", False, sticker

    def deviates(self):
        return self.toppled or self.color != self.cls or math.hypot(*(self.pos - self.normal)) > LAYOUT["pos_tol"]


class Spot:
    def __init__(self, k, pair):
        centre, left, right = object_slots(k)
        self.pair = pair
        self.identical = False  # look-alikes differ only by a top sticker: text cannot tell them apart (2(c), 3B, 4)
        self.objs = [Obj(left, COLORS[k], "P"), Obj(right, COLORS[k], "P'")] if pair else [Obj(centre, COLORS[k])]

    @property
    def n_similar(self):
        return len(self.objs)

    @property
    def spacing_cm(self):
        return 2 * LAYOUT["pair_dx"] * 100 if self.pair else 20.0


class Anomaly:
    def __init__(self, k, obj, atype, t, rng=None):
        self.spot, self.obj, self.atype, self.t_event = k, obj, atype, t
        self.rng = rng if rng is not None else np.random.default_rng(0)  # perception + worker draws of this anomaly
        self.detected = self.confirmed = self.b_saw = self.reported = self.wrong = self.need_pose = False
        self.hist = self.job = self.sent = self.move_at = None
        self.moved = self.cur_ver = 0  # true position version / version A last measured (4(d))


def describe(act):
    return "nothing" if act == 0 else "base" + "".join(" + " + ITEMS[i] for i in sorted(items_of(act)))


class World:
    def __init__(self, rng, condition="base", alpha=0.1, beta=1.0, gamma=1.0, trace=False):
        L, c = LAYOUT, CONDITIONS[condition]
        self.rng, self.cond, self.trace = rng, condition, trace
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.occluder = c["occluder"]
        self.bw, self.loss = rng.uniform(*c["bw"]), rng.uniform(*c["loss"])
        pairs = set(rng.choice(len(L["spots"]), c.get("pairs", 2), replace=False))  # spots holding two objects
        self.atypes = c.get("atypes", (0, 1, 2))
        self.spots = [Spot(k, k in pairs) for k in range(len(L["spots"]))]
        n = len(self.spots)
        # A's estimate of B = knowledge_estimation (fed by _knows(...) and the _core(...) hooks below). The Notion
        # section 10 baselines run alongside only so that the comparison methods can read them.
        self.knows = {"core": CoreKnowledge(self, CORE_PARAMETERS, L["spots"], SCORE_MODEL, CORE_CAPS)}
        self.knows.update({m: BKnowledge(n, m) for m in BASELINES})
        self.know = self.knows["core"]
        self.b_obj, self.b_normal, self.b_wrong = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)  # truth
        self.b_epoch = self.a_epoch = 0                   # B's memory epoch / the one A last heard of (2(b))
        self.t, self.r, self.log = 0.0, 0.0, []
        self.anom, self.inbox, self.queue, self.pending = {}, [], [], []
        self.a_phi, self.a_stop_until = 0.0, 0.0
        self.b_state, self.b_job, self.b_frames, self.b_i = "idle", None, None, 0
        self.b_pose = (np.array(L["home"]), 0.0)
        self.b_trip = set()  # spots B recognised on the current trip (truth)
        # events get their own stream, and each anomaly its own generator: the same seed gives the same anomalies,
        # perception and worker behaviour whatever the policy sends (fair method comparison)
        self.ev = np.random.default_rng(int(rng.integers(2 ** 63)))
        self.next_event = self.ev.exponential(L["anomaly_every"])
        self.next_reset = self.ev.exponential(L["reset_every"])
        self.next_tick5 = 5.0
        self.stats = dict(condition=condition, anomalies=0, fixed=0, times=[], bytes=0, msgs=0, images=0,
                          redundant=0, requests=0, pulls=0, reminders=0, wrong_ops=0, retx=0, resets=0,
                          not_found=0, pose_updates=0, unconfirmed=0, v2=0, deadline_miss=0, verify_delay=[],
                          kappa=[], know_err=[], dec=[], trips=[], trip_open=[],
                          bytes_by=dict(task=0, retx=0, info=0, pose=0, v2=0, ack=0, result=0, status=0))

    # ----- bookkeeping -----
    def _log(self, who, msg):
        if self.trace:
            self.log.append(f"{self.t:6.1f}s  {who:<2} {msg}")

    def _spend(self, n, kind):
        self.stats["bytes"] += n
        self.stats["bytes_by"][kind] += n
        self.r -= self.alpha * n / 1000

    def _send(self, n, image, kind):
        """A -> B message."""
        self._spend(n, kind)
        self.stats["msgs"] += 1
        self.stats["images"] += image

    def _delivered(self):
        return self.rng.random() >= self.loss

    def _reply(self, kind="ack"):
        """B -> A message (ACK / info request / result / STATUS): costs bytes, may be lost."""
        self._spend(REPLY_BYTES, kind)
        return self._delivered()

    def _delay(self, n):
        return n / 1000 / self.bw  # seconds

    def _knows(self, method, *args):
        for kn in self.knows.values():
            getattr(kn, method)(*args)

    def _core(self, method, *args):
        """Events with message identity / time for the knowledge_estimation core (A-side information only)."""
        kn = self.knows.get("core")
        if kn is not None:
            getattr(kn, method)(*args)

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
        self._core("decision", k)  # decision-time record of b_raw / b_cal / b_safe, before anything is sent
        self._record(a, act)
        self.stats["know_err"].append(abs(self.know.normal[k] - self.b_normal[k]))
        self._log("A", f"spot {k}: {ATYPES[a.atype]} | B knows obj {self.know.obj[k]:.2f} "
                       f"normal {self.know.normal[k]:.2f} current {self.know.cur[k]:.2f} | send: {describe(act)}")
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
            body = {ID: f"id={o.cls}{' ' + o.sticker if o.sticker else ''}",
                    CUR: f"cur={cm(a.cur_rep)}±{a.pos_err:.0f} v{a.cur_ver}", NORM: f"norm={cm(o.normal)}",
                    IMG: "img=crop 30KB"}
            self._log("", f"  msg {{spot={k}, {ATYPES[a.atype]}" + "".join(", " + body[i] for i in sorted(items))
                      + f"}} {n} B, {self._delay(n) * 1000:.0f} ms")
        self._send(n, IMG in items, "task")
        a.hist.on_send(items)
        payload = dict(items=set(items), cur_rep=a.cur_rep.copy(), cur_ver=a.cur_ver)
        a.sent = dict(payload=payload, n=n, tries=0, deadline=self.t + LAYOUT["ack_timeout"] + self._delay(n))
        self._core("send", k, payload, items, 1)
        self._post(a, "cmd", payload, n)

    def _post(self, a, kind, payload, n):
        if self._delivered():
            self.inbox.append((self.t + self._delay(n), kind, a, payload))
        else:
            self._log("", f"  {kind} lost ({n} B)")

    def _record(self, a, act):
        """Decision-time record for the scenario doc's metrics: A's estimates (every estimator), B's truth, need."""
        k = a.spot
        items = items_of(act) if act else set()
        job = a.job
        k_c = a.b_saw or bool(job and not job["failed"] and job["items"] & {CUR, IMG} and job["cur_ver"] == a.moved)
        no_img = self._pick(a, {ID, CUR, NORM}, a.cur_rep)
        self.stats["dec"].append(dict(
            act=act, sent=sorted(items), est={m: (kn.normal[k], kn.obj[k], kn.cur[k]) for m, kn in self.knows.items()},
            K=(bool(self.b_normal[k]), bool(self.b_obj[k] and not self.b_wrong[k]), bool(k_c)),
            need_n=not self.b_normal[k], need_i=no_img is not a.obj,
            without_img="ok" if no_img is a.obj else "ask" if no_img is None else "not_found" if no_img == NOT_FOUND
            else "wrong"))

    def run_until_decision(self):
        """Tick until A needs a decision; returns True when the patrol is over."""
        while not self.pending:
            if self.done:
                open_ = [a for a in self.anom.values() if a.detected]
                self.r -= 10 * len(open_)  # unresolved at the end
                self.stats["deadline_miss"] += sum(self.t - a.hist.t0 > LAYOUT["deadline"] for a in open_)
                self._log("--", f"patrol over, unresolved at spots {sorted(a.spot for a in open_)}")
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
            self.next_event = self.t + self.ev.exponential(L["anomaly_every"])
            self._new_anomaly()
        if self.t >= self.next_reset:
            self._memory_reset()
        self._tick_retx()
        self._tick_a()
        self._tick_inbox()
        self._tick_moves()
        self._tick_b()

    def _new_anomaly(self):
        e = self.ev  # always the same number of draws, so the stream stays aligned across policies
        u, which, atype, shift = e.random(), e.random(), int(e.integers(3)), int(e.integers(1, len(COLORS)))
        atype = self.atypes[atype % len(self.atypes)]
        offset, seed = self._offset(0.04, 0.10, e), int(e.integers(2 ** 63))
        free = [k for k in range(len(self.spots)) if k not in self.anom]
        if not free:
            return
        k = free[int(u * len(free))]
        obj = self.spots[k].objs[int(which * len(self.spots[k].objs))]
        if atype == TOPPLED:
            obj.toppled = True
        elif atype == WRONG:
            obj.color = COLORS[(COLORS.index(obj.cls) + shift) % len(COLORS)]
        else:
            obj.pos = obj.normal + offset
        self.anom[k] = Anomaly(k, obj, atype, self.t, np.random.default_rng(seed))
        self._log("!", f"spot {k}: {ATYPES[atype]} happens ({self.spots[k].n_similar} object(s))")

    def _offset(self, lo, hi, rng):
        ang, d = rng.uniform(0, 2 * math.pi), rng.uniform(lo, hi)
        return np.array([math.cos(ang), math.sin(ang)]) * d

    def _memory_reset(self):
        """2(b): B's persistent cache is wiped; its STATUS(epoch) to A is lost this often (failure injection)."""
        lost = self.ev.random() < LAYOUT["status_loss"]
        self.next_reset = self.t + self.ev.exponential(LAYOUT["reset_every"])
        self.b_obj[:] = self.b_normal[:] = self.b_wrong[:] = False
        self.b_epoch += 1
        self.stats["resets"] += 1
        self._spend(REPLY_BYTES, "status")
        if not lost and self._delivered():
            self._epoch_seen()
        self._log("B", f"memory reset -> epoch {self.b_epoch}" + (", STATUS lost" if self.a_epoch != self.b_epoch else ""))

    def _epoch_seen(self):
        if self.a_epoch != self.b_epoch:
            self.a_epoch = self.b_epoch
            self._knows("reset")
            self._log("A", f"B's memory epoch {self.b_epoch}: all estimates void")

    def _tick_retx(self):
        """1(b): no ACK by the deadline -> resend the same message (B drops duplicates)."""
        for a in self.anom.values():
            s = a.sent
            if s is None or a.hist.ack or a.hist.result or self.t < s["deadline"] or s["tries"] >= LAYOUT["max_retx"]:
                continue
            s["tries"] += 1
            s["deadline"] = self.t + LAYOUT["ack_timeout"] + self._delay(s["n"])
            self._core("send", a.spot, s["payload"], s["payload"]["items"], s["tries"] + 1)
            self.stats["retx"] += 1
            self._send(s["n"], IMG in s["payload"]["items"], "retx")
            self._log("A", f"spot {a.spot}: no ACK -> resend same message ({s['n']} B)")
            self._post(a, "cmd", s["payload"], s["n"])

    # ----- Robot A -----
    def _tick_a(self):
        if self.t < self.a_stop_until:
            return
        (_, ax, ay), phi = LAYOUT["ellipse"], self.a_phi
        speed = math.hypot(ax * math.sin(phi), ay * math.cos(phi))
        new = phi + LAYOUT["a_speed"] * LAYOUT["dt"] / speed
        hit = [(k, view) for view, pts in enumerate((obs_points(), obs_points2())) for k, p in enumerate(pts)
               if p is not None and math.floor((phi - p) / (2 * math.pi)) != math.floor((new - p) / (2 * math.pi))]
        self.a_phi = new
        if hit:
            self.a_stop_until = self.t + LAYOUT["obs_stop"]
            for k, view in hit:
                self._observe(k, view)

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

    def _measure(self, a):
        # sigma = RMS distance error (cm), so each axis gets sigma / sqrt(2); B searches within 2 sigma
        a.cur_rep = a.obj.pos + a.rng.normal(0, a.true_err / 100 / math.sqrt(2), 2)  # the "cur" A would send
        a.cur_ver = a.moved
        self._core("pose_measured", a.spot, a.cur_ver, self.t)

    def _observe(self, k, view):
        a = self.anom.get(k)
        if not any(o.deviates() for o in self.spots[k].objs):
            self._knows("on_spot_seen", k, True)  # 3A: held pass-by evidence was a normal snapshot
            if a is not None:
                self._verified(a)
            return
        if a is None:
            return
        vis = self.visible_fraction(ellipse_pose(self.a_phi), a.obj)
        if vis < 0.3:
            self._log("A", f"spot {k}: anomaly hidden (visible {vis:.0%}), missed")
            return
        self._knows("on_spot_seen", k, False)  # 3A: the spot is abnormal, drop held pass-by evidence
        if a.reported:  # B said ok, A's re-check disagrees
            self._reopen(a)
            return
        if not a.detected:  # perception fixed per anomaly
            r, s = a.rng, self.spots[k]
            a.detected, a.hist, a.occ = True, History(self.t), 1 - vis
            dist = math.hypot(*(a.obj.pos - ellipse_pose(self.a_phi)[0]))
            a.pos_err = 1 + 1.5 * dist + 4 * a.occ + abs(r.normal(0, 0.5))  # sigma A reports (cm)
            a.true_err = a.pos_err * (LAYOUT["overconf"] if r.random() < LAYOUT["p_overconf"] else 1)
            self._measure(a)
            acc = np.clip(0.98 - 0.4 * a.occ - 0.35 * s.pair, 0.05, 1)
            a.class_ok, a.conf = r.random() < acc, float(np.clip(acc + r.normal(0, 0.08), 0, 1))
            a.img_match = r.random() < 0.92 - 0.4 * a.occ
            a.confirmed = not (s.pair and a.atype == MISPLACED)  # a moved look-alike needs the sticker read
            self.stats["anomalies"] += 1
            self.stats["kappa"].append(a.true_err / a.pos_err)
            self._knows("on_new_anomaly", k)
        if not a.confirmed:  # 4(c): do not send a label A is unsure of; look again from the other view
            if a.rng.random() < LAYOUT["p_sticker"][view]:
                a.confirmed = True
                self._log("A", f"spot {k}: sticker read from view {view + 1} -> target confirmed")
            else:
                self.stats["unconfirmed"] += 1
                self._log("A", f"spot {k}: sticker not visible from view {view + 1} -> re-observe, nothing sent")
                return
        if a.moved != a.cur_ver or a.need_pose:  # 4(d): it moved again since A measured it
            self._measure(a)
            if a.job is not None and not a.job["failed"] and not a.job.get("manip"):
                n = BASE_BYTES + ITEM_BYTES[CUR]
                self._send(n, False, "pose")
                self.stats["pose_updates"] += 1
                self._log("A", f"spot {k}: moved again -> POSE_UPDATE v{a.cur_ver}")
                pose = dict(cur_rep=a.cur_rep.copy(), cur_ver=a.cur_ver)
                self._core("send", k, pose, {CUR}, 1)
                self._post(a, "pose", pose, n)
            a.need_pose = False
        self.pending.append(a)

    def _verified(self, a):
        """A's re-check: every object at the spot is normal -> close the event, now trust what B learned."""
        k = a.spot
        self.anom.pop(k)
        if not a.detected:
            return  # fixed before A ever saw it (worker undid a mistake): nothing to count
        self.stats["fixed"] += 1
        self.stats["times"].append(self.t - a.hist.t0)
        self.stats["deadline_miss"] += self.t - a.hist.t0 > LAYOUT["deadline"]
        if a.reported:
            self.stats["verify_delay"].append(self.t - a.t_report)
        self.r += 10
        self._knows("on_success", k)
        self._log("A", f"spot {k}: re-check ok -> verified, closed")

    def _reopen(self, a):
        """B reported ok but the spot is still wrong: B handled the other object (2(d), 4(b)). Void what B learned
        there and send TASK v2 (target: N, C, crop; the other object: N', C')."""
        k = a.spot
        a.reported = False
        if not a.wrong:
            self.pending.append(a)
            return
        self.stats["wrong_ops"] += 1
        self.stats["v2"] += 1
        self.r -= WRONG_OP
        self._knows("on_wrong", k)
        self._measure(a)
        items = {NORM, CUR, IMG}
        n = BASE_BYTES + sum(ITEM_BYTES[i] for i in items) + V2_EXTRA
        self._send(n, True, "v2")
        self._log("A", f"spot {k}: re-check -> B moved the other object (unreported) -> TASK v2 with crop ({n} B)")
        payload = dict(items=items, cur_rep=a.cur_rep.copy(), cur_ver=a.cur_ver, v2=True, undo=a.wrong_obj)
        a.sent = dict(payload=payload, n=n, tries=0, deadline=self.t + LAYOUT["ack_timeout"] + self._delay(n))
        a.hist.on_send(items)
        a.wrong = False
        self._core("send", k, payload, items, 1)
        self._post(a, "cmd", payload, n)

    # ----- channel -----
    def _tick_inbox(self):
        for msg in [m for m in self.inbox if m[0] <= self.t]:
            self.inbox.remove(msg)
            _, kind, a, p = msg
            if self.anom.get(a.spot) is not a:
                continue  # already closed
            k = a.spot
            if kind == "pose":
                serving = self.b_job is not None and self.b_job[0] is a
                job = self.b_job[1] if serving else a.job  # the job B is actually carrying out
                if job is not None:
                    job.update(cur_rep=p["cur_rep"], cur_ver=p["cur_ver"])
                    if serving and self.b_state == "wait_pose":
                        self._log("B", f"spot {k}: POSE_UPDATE v{p['cur_ver']} -> search again")
                        self._check(a, job)
                continue
            self._learn(a, p["items"])
            if kind == "resp":
                if a.job is not None:
                    a.job["items"] |= p["items"]
                if self._reply():
                    self._knows("on_ack", k, p["items"])
                    self._core("ack", p, self.a_epoch)
                continue
            # kind == "cmd" (task, retransmit or TASK v2)
            job = a.job
            # a resent message carries the same payload object: same job, not a new one
            fresh = job is None or job["failed"] or (p.get("v2") and job.get("src") is not p)
            if fresh:
                job = a.job = dict(items=set(p["items"]), failed=False, started=False, asked=False,
                                   cur_rep=p["cur_rep"], cur_ver=p["cur_ver"], v2=p.get("v2", False),
                                   undo=p.get("undo"), src=p)
                if a not in self.queue and not (self.b_job and self.b_job[0] is a):
                    self.queue.append(a)
            elif not job["started"]:
                job["items"] |= p["items"]  # duplicate or more info before B left
            need_n = NORM not in job["items"] and not self.b_normal[k] and not job["started"]
            if self._reply():  # ACK carries B's memory epoch and what B still needs before leaving (2(b))
                self._epoch_seen()
                a.hist.on_ack()
                self._knows("on_ack", k, p["items"])
                self._core("ack", p, self.a_epoch)
                if need_n and not job.get("pulled"):
                    job["pulled"] = True
                    self.stats["requests"] += 1
                    self.stats["pulls"] += 1
                    self.r -= REQUEST
                    self._knows("on_missing", k, {NORM})
                    n = BASE_BYTES + ITEM_BYTES[NORM]
                    self._send(n, False, "info")
                    self._log("B", f"spot {k}: ACK + needs_info(N) -> A sends N before B leaves")
                    resp = dict(items={NORM})
                    self._core("send", k, resp, {NORM}, 1)
                    self._post(a, "resp", resp, n)

    def _learn(self, a, items):
        if NORM in items:
            self.b_normal[a.spot] = True
        if IMG in items:  # a fresh crop replaces whatever appearance B stored
            self.b_obj[a.spot], self.b_wrong[a.spot] = True, False

    def _tick_moves(self):
        """4(d): the worker moves the target again while B is on its way."""
        for a in self.anom.values():
            if a.move_at is not None and self.t >= a.move_at:
                a.move_at = None
                a.obj.pos = a.obj.pos + self._offset(*LAYOUT["move_off"], a.rng)
                a.moved += 1
                self._log("!", f"spot {a.spot}: target moved again while B is on its way")

    # ----- Robot B -----
    def _drive(self):
        """Move one frame; returns True at the end of the leg. Recognises spots in view (truth, hidden from A)."""
        pos, heading, vis = self.b_frames[self.b_i]
        self.b_pose = (pos, heading)
        for j in vis:
            if self.rng.random() < LAYOUT["p_frame"]:
                if not self.b_obj[j]:
                    self.b_obj[j], self.b_wrong[j] = True, self.rng.random() > LAYOUT["q_learn"]
                if j in self.anom and any(o.deviates() for o in self.spots[j].objs):
                    self.anom[j].b_saw = True
                else:
                    self.b_normal[j] = True
                self.b_seen.add(j)
                self.b_trip.add(j)  # truth per trip, for measuring p_acq (knowledge_estimation calibration)
        self.b_i += 1
        return self.b_i >= len(self.b_frames)

    def _pick(self, a, items, cur):
        """Which object B would point the worker at; None = unsure (asks A); NOT_FOUND = nothing near `cur`.
        Look-alikes differ only by a sticker, so text never separates them: B needs the crop, its own memory of the
        look (from handling or passing), or a report that leaves only one candidate within 2 sigma."""
        s = self.spots[a.spot]
        if a.atype in (TOPPLED, WRONG) or s.n_similar == 1 or a.b_saw:  # odd one out, or B saw it happen
            return a.obj
        cands = s.objs if CUR not in items else [o for o in s.objs
                                                 if math.hypot(*(o.pos - cur)) <= 2 * a.pos_err / 100]
        if not cands:
            return NOT_FOUND
        if IMG in items and a.img_match:
            return a.obj if a.obj in cands else NOT_FOUND
        if self.b_obj[a.spot]:  # B's stored look; wrong if it learned the other one as this (2(d))
            want = next(o for o in s.objs if o is not a.obj) if self.b_wrong[a.spot] else a.obj
            if want in cands:
                return want
        if CUR in items and len(cands) == 1:
            return cands[0]
        return None

    def _instruction(self, a, target):
        k, o = a.spot, a.obj
        side = ("왼쪽 " if target.normal[0] < LAYOUT["spots"][k][0] else "오른쪽 ") if self.spots[k].pair else ""
        name = f"{side}{COLORS_KO[o.cls]} 물건"
        if a.atype == TOPPLED:
            return f"스팟 {k}에서 넘어진 {name}을 세워서 제자리에 놓아주세요"
        if a.atype == WRONG:
            return f"스팟 {k}의 {COLORS_KO[o.color]} 물건을 {name}으로 바꿔주세요"
        dx, dy = (o.normal - target.pos) * 100
        parts = [f"{'오른쪽' if dx > 0 else '왼쪽'}으로 {abs(dx):.0f}cm" if abs(dx) >= 1 else "",
                 f"{'앞쪽' if dy > 0 else '뒤쪽'}으로 {abs(dy):.0f}cm" if abs(dy) >= 1 else ""]
        return f"스팟 {k}에서 {name}을 " + ", ".join(p for p in parts if p) + " 옮겨주세요"

    def _worker_fix(self, a, job):
        if job.get("undo") is not None:  # TASK v2: first put back the object moved by mistake
            u, job["undo"] = job["undo"], None
            u.pos = u.normal + a.rng.normal(0, 0.005, 2)
        target = job["target"]
        if target is not a.obj:  # worker follows the instruction: the other object goes to the target's place
            target.pos = a.obj.normal + a.rng.normal(0, 0.005, 2)
            self._log("W", f"spot {a.spot}: worker moved the object B pointed at (the wrong one)")
            return
        o = a.obj
        o.toppled, o.color = False, o.cls
        if a.rng.random() < LAYOUT["p_wrong"]:
            o.pos = o.normal + self._offset(*LAYOUT["wrong_off"], a.rng)
            self._log("W", f"spot {a.spot}: worker moved it, but to the wrong place")
        else:
            o.pos = o.normal + a.rng.normal(0, 0.005, 2)
            self._log("W", f"spot {a.spot}: worker fixed it")

    def _b_done(self, a, job):
        """What B sees: the object it pointed at sits at the target's normal place (B cannot tell P from P')."""
        target = job["target"]
        if target is a.obj:
            return not target.deviates()
        return math.hypot(*(target.pos - a.obj.normal)) <= LAYOUT["pos_tol"]

    def _tick_b(self):
        L, st = LAYOUT, self.b_state
        if st == "idle":
            if not self.queue:
                return
            a = self.queue.pop(0)
            job = a.job
            job.update(started=True, t_start=self.t)
            self.b_job, self.b_seen, self.b_trip = (a, job), set(), set()
            self.b_frames, self.b_i, self.b_state = routes(self.occluder)[a.spot][0], 0, "out"
            if a.atype == MISPLACED and not job["v2"] and a.rng.random() < L["p_move_again"]:
                a.move_at = self.t + a.rng.uniform(1.0, 5.0)  # 4(d)
            self._log("B", f"drives standby -> spot {a.spot} with {sorted(ITEMS[i] for i in job['items'])}")
            return
        a, job = self.b_job
        if st == "out":
            if self._drive():
                self._log("B", f"arrives at spot {a.spot}, passed and saw spots {sorted(self.b_seen)} (hidden from A)")
                self._check(a, job)
        elif st == "wait":
            if self.t >= job["wait_until"] or not self._missing(a, job)[0]:
                self._check(a, job)
        elif st == "wait_pose":
            if self.t >= job["wait_until"]:
                self._give_up(a, job, NEED_ID, "no POSE_UPDATE in time")
        elif st == "watch":
            if self._b_done(a, job):
                self._report_ok(a, job)
            elif self.t >= job["worker_at"]:
                job["worker_at"], job["fixed_at"] = math.inf, self.t
                self._worker_fix(a, job)
            elif self.t - job["fixed_at"] >= L["remind_after"]:
                self.stats["reminders"] += 1
                self._log("B", "-> 노트북: 물건 위치를 수정해주세요")
                job["fixed_at"], job["worker_at"] = math.inf, self.t + a.rng.uniform(*L["worker_delay"])
        elif st == "back":
            if self._drive():
                self.b_state, self.b_job = "idle", None
                self._log("B", f"back at standby, saw spots {sorted(self.b_seen)} on the trip (hidden from A)")
                open_ = {j for j, x in self.anom.items() if x.detected and x.hist.t0 <= job["t_start"]}
                self.stats["trips"].append((a.spot, {j: (j in self.b_trip, j in self.b_trip and not self.b_wrong[j])
                                                     for j in range(len(self.spots)) if j != a.spot}))
                self.stats["trip_open"].append(sorted(open_))  # spots A knew were abnormal (not a normal pass)
                if self._reply("result"):  # trip report: A knows the route from its own command
                    if job["failed"]:
                        a.hist.on_result(job["reason"])
                    self._knows("on_pass", visibility_table()[a.spot], open_)
                    self._core("trip", a.spot, open_, sorted(visibility_table()[a.spot]))  # A's route map

    def _missing(self, a, job):
        pick = self._pick(a, job["items"], job["cur_rep"])
        miss = set()
        if pick is None:
            miss.add(IMG)
        if NORM not in job["items"] and not self.b_normal[a.spot]:
            miss.add(NORM)
        return miss, pick

    def _check(self, a, job):
        """At the spot: not_found -> wait for A's POSE_UPDATE; missing info -> ask A once; then instruct or give up."""
        miss, pick = self._missing(a, job)
        k = a.spot
        if pick == NOT_FOUND and not miss:
            self.stats["not_found"] += 1
            self.r -= REOBSERVE
            job.update(wait_until=self.t + LAYOUT["pose_wait"])
            self.b_state = "wait_pose"
            self._log("B", f"spot {k}: nothing matches near the reported position -> not_found")
            if self._reply("result"):
                a.need_pose = True
                self._knows("on_missing", k, {CUR})
            return
        if miss and not job["asked"]:
            job["asked"], job["wait_until"], self.b_state = True, self.t + LAYOUT["req_timeout"], "wait"
            self.stats["requests"] += 1
            self.r -= REQUEST
            self._log("B", f"spot {k}: missing {[ITEMS[i] for i in sorted(miss)]} -> asks A")
            if self._reply():
                self._knows("on_missing", k, miss)  # 3B occluded: the request itself says B does not hold them
                n = BASE_BYTES + sum(ITEM_BYTES[i] for i in miss)
                self._send(n, IMG in miss, "info")
                a.hist.items |= miss
                resp = dict(items=miss)
                self._core("send", k, resp, miss, 1)
                self._post(a, "resp", resp, n)
            return
        if miss or pick == NOT_FOUND:
            self._give_up(a, job, NEED_ID if IMG in miss or pick == NOT_FOUND else NEED_NORMAL, "cannot instruct")
            return
        job["target"], job["manip"] = pick, True
        undo = "잘못 옮긴 물건을 제자리로 되돌리고, " if job.get("undo") is not None else ""
        self._log("B", "-> 노트북: " + undo + self._instruction(a, pick) + (" (사진 첨부)" if job["v2"] else ""))
        job.update(worker_at=self.t + a.rng.uniform(*LAYOUT["worker_delay"]), fixed_at=math.inf)
        self.b_state = "watch"

    def _give_up(self, a, job, reason, why):
        job["reason"], job["failed"] = reason, True
        self.r -= 3  # wasted trip; must stay < unresolved penalty or sending nothing wins
        self._log("B", f"spot {a.spot}: {why} ({['ok', 'need id', 'need normal'][reason]}), driving back")
        self._go_back(a)

    def _report_ok(self, a, job):
        """B believes the repair is done and reports ok; A trusts it only after its own re-check."""
        k, target = a.spot, job["target"]
        a.wrong, a.wrong_obj = target is not a.obj, target
        self.b_normal[k] = True
        if a.wrong:
            self.b_obj[k], self.b_wrong[k] = True, True  # it stored the other object's look as the target's
        elif not self.b_obj[k]:
            self.b_obj[k], self.b_wrong[k] = True, self.rng.random() > LAYOUT["q_learn"]
        job["reason"] = OK
        if self._reply("result"):
            a.reported, a.t_report = True, self.t
            a.hist.on_result(OK)
        self._log("B", f"spot {k}: repair looks done -> RESULT ok, driving back")
        self._go_back(a)

    def _go_back(self, a):
        self.b_frames, self.b_i, self.b_state = routes(self.occluder)[a.spot][1], 0, "back"


def scenario_world(k, shift, pair=False, pos_err=2.0, cur_rep=None, true_err=None):
    """A world with one misplaced object at spot k (left one if pair), already detected and confirmed by A."""
    w = World(np.random.default_rng(0), "base", trace=True)
    w.next_event = w.next_reset = math.inf
    w.spots[k] = Spot(k, pair)
    o = w.spots[k].objs[0]
    o.pos = o.normal + np.array(shift)
    a = w.anom[k] = Anomaly(k, o, MISPLACED, 0.0)
    a.detected, a.confirmed, a.hist = True, True, History(0)
    a.pos_err, a.true_err, a.class_ok, a.img_match, a.conf, a.occ = pos_err, true_err or pos_err, True, True, .9, 0
    a.cur_rep = o.pos.copy() if cur_rep is None else np.array(cur_rep)
    w.stats["anomalies"] = 1
    return w, a


def run_scenario(w, a, act, ticks=1500):
    w.pending.append(a)
    w.decide(act)
    for _ in range(ticks):
        w.tick()
        if w.pending:
            w.pending.clear()  # A's repeat decisions: send nothing
        if w.stats["fixed"] and w.b_state == "idle":
            break
    return w


if __name__ == "__main__":
    L = LAYOUT
    for k, phi in enumerate(obs_points()):
        pos, _, _ = ellipse_pose(phi)
        assert math.hypot(*np.subtract(L["spots"][k], pos)) >= 0.4, k
    assert sum(p is not None for p in obs_points2()) >= 3, obs_points2()  # 4(c) needs a second view
    seen = lambda occl, k, j: sum(j in v for o_b in routes(occl)[k] for _, _, v in o_b)  # frames, out + back
    assert seen(False, 2, 1) > 0 and seen(False, 2, 3) > 0, "S2 trip should show S1 and S3"
    assert seen(True, 2, 1) < seen(False, 2, 1) / 2, "occluder should hide most of S1 on the S2 trip"
    assert visibility_table()[2][1] > 0.9  # A's table does not know about the occluder
    E, N, C, I = 1, 1 + 4, 1 + 2 + 4, 1 + 2 + 4 + 8
    p_wrong, L["p_wrong"], L["p_move_again"] = L["p_wrong"], 0.0, 0.0

    # 1(a)+2(b): E only, B lacks N -> needs_info at ACK -> N before leaving; A closes only after its re-check
    w, a = scenario_world(1, [0.06, 0.0])
    run_scenario(w, a, E)
    assert w.stats["pulls"] == 1 and w.stats["fixed"] == 1 and w.stats["verify_delay"], w.stats
    assert w.knows["cache_result"].normal[1] == 1 and w.knows["cache_result"].obj[1] == 0.9 and w.b_state == "idle", w.log

    # worker gets it wrong once: B keeps watching and reminds after 10 s
    L["p_wrong"] = 1.0
    w, a = scenario_world(1, [0.06, 0.0])
    w.pending.append(a), w.decide(N)
    for _ in range(1500):
        w.tick()
        w.pending.clear()
        if any("수정해주세요" in s for s in w.log):
            L["p_wrong"] = 0.0
        if w.stats["fixed"] and w.b_state == "idle":
            break
    assert any("왼쪽으로 6cm" in s for s in w.log) and w.stats["reminders"] >= 1 and w.stats["fixed"] == 1, w.log

    # 4(b): look-alike pair, A trusts sigma 1 cm but 'cur' lands on the other object -> B points at it, reports ok
    # (it cannot tell) -> A's re-check reopens -> wrong op, knowledge void -> TASK v2 with crop -> verified
    w, a = scenario_world(2, [0.0, 0.06], pair=True, pos_err=1.0, cur_rep=object_slots(2)[2])
    run_scenario(w, a, C)
    assert w.stats["wrong_ops"] == 1 and w.stats["v2"] == 1 and w.stats["fixed"] == 1, (w.stats, w.log)
    assert any("오른쪽 파란 물건" in s for s in w.log) and any("RESULT ok" in s for s in w.log), w.log

    # 1(b): first message lost -> resend after the ACK deadline
    w, a = scenario_world(1, [0.06, 0.0])
    w.loss = 1.0
    w.pending.append(a), w.decide(N)
    for _ in range(40):
        w.tick()
    assert w.stats["retx"] >= 1 and w.stats["bytes_by"]["retx"] > 0, w.stats

    # 2(b): memory reset with STATUS lost -> A finds out from the next ACK's epoch and voids its estimates
    w, a = scenario_world(1, [0.06, 0.0])
    w.knows["cache_result"].normal[:] = 1.0  # team BKnowledge rules (A uses knowledge_estimation)
    w.b_normal[:] = True
    w.next_reset, L["status_loss"] = 0.0, 1.0
    w.tick()
    assert w.a_epoch == 0 and w.b_epoch == 1 and w.knows["cache_result"].normal[1] == 1.0  # A still believes
    run_scenario(w, a, E)
    assert w.a_epoch == 1 and w.stats["pulls"] == 1 and w.stats["fixed"] == 1, (w.stats, w.log)
    L["status_loss"] = 0.5

    # 4(d): target moved again while B drives -> B cannot find it -> not_found -> A re-observes -> POSE_UPDATE
    w, a = scenario_world(2, [0.0, 0.06], pair=True, pos_err=1.0)
    L["p_move_again"] = 1.0
    run_scenario(w, a, C + 8, ticks=3000)
    assert w.stats["fixed"] == 1 and (w.stats["pose_updates"] or w.stats["not_found"]), (w.stats, w.log)
    L["p_move_again"], L["p_wrong"] = 0.15, p_wrong

    # 3A: pass-by normal evidence is held until A sees the spot normal, dropped if A finds it abnormal
    kn = BKnowledge(4)
    kn.on_pass({0: 0.8, 1: 0.8}, set())
    kn.on_spot_seen(0, True), kn.on_spot_seen(1, False)
    assert kn.normal[0] == 0.8 and kn.normal[1] == 0.0 and kn.hold.sum() == 0
    print("ok | views:", [round(p, 2) for p in obs_points()], [p and round(p, 2) for p in obs_points2()],
          "| S2 trip sees S1 frames:", seen(False, 2, 1), "-> with occluder:", seen(True, 2, 1))
