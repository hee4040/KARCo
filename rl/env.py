"""Event-level simulator of the A -> B anomaly repair loop (team scenarios SC-1..3).

One step = Robot A observes an anomaly at a spot and picks what to send:
0 = nothing, 1..16 = base (event id, spot, anomaly, repair request) + a subset of items (features.ITEMS).
Between steps A keeps patrolling. B (line tracking, D455) drives from standby to the spot, asks A once for
missing info, sends a text repair instruction to the laptop, a human fixes the object, B checks the fix
with its camera, then drives back to standby and replies to A.
B's true knowledge is tracked here and hidden from A; A only sees its own estimate (features.BKnowledge).
"""
import math
import gymnasium as gym
import numpy as np
from features import (BELIEF, CUR, ID, IMG, ITEMS, NEED_ID, NEED_NORMAL, NORM, OBS_DIM, OK, BKnowledge, History,
                      items_of, make_obs, route_spots)

N_ACTIONS = 17
FULL = 16                       # base + every item: the Full Sharing baseline
BASE_BYTES = 60
ITEM_BYTES = {ID: 40, CUR: 60, NORM: 60, IMG: 30_000}
REPLY_BYTES = 50                # ACK, task_result, info request
STEP_S = 5.0                    # seconds per patrol visit
P_FIX = 0.95                    # human fix passes B's check the first time; otherwise the human redoes it (not a comm failure)
HOME = -1                       # B's standby position, before spot 0 on the line
TOPPLED = 1                     # anomaly types: 0 misplaced, 1 toppled, 2 wrong part
ATYPES = ["misplaced", "toppled", "wrong part"]

# ponytail: hand-set ranges, replace with stats from real trial logs
# multi: P(spot holds 2-4 look-alikes), identical: look-alikes are exact copies (only position/image tells them apart)
# lookalike: P(A may misclassify), dist: A-to-tag (m), vis: P(B sees a spot it drives past), weights: where anomalies happen
_BASE = dict(multi=0.2, identical=0.5, spacing=(10, 20), lookalike=0.1, occ=(0, .2), dist=(.4, .8),
             bw=(200, 1000), loss=(0, .02), fix=(2, 4), vis=0.7, weights=(1, 1, 1, 1))
# fix: steps for drive + text report + human fix + B's check  (ponytail: measure on the real robots)
CONDITIONS = {
    "normal":        _BASE,
    "repeat":        {**_BASE, "weights": (6, 1, 1, 1)},                  # SC-2: same object keeps coming back wrong
    "pass_clear":    {**_BASE, "weights": (3, 1, 1, 3), "vis": 0.9},      # SC-3: B drives past spots 1, 2 in plain view
    "pass_occluded": {**_BASE, "weights": (3, 1, 1, 3), "vis": 0.3},      # SC-3: that path is partly blocked
    "clustered":     {**_BASE, "multi": 1.0, "identical": 1.0, "spacing": (3, 6), "occ": (0, .3), "dist": (.6, 1.2)},
    "limited":       {**_BASE, "multi": 0.3, "bw": (5, 50), "loss": (.1, .3)},
}


class Spot:
    def __init__(self, rng, c):
        self.n_similar = int(rng.integers(2, 5)) if rng.random() < c["multi"] else 1
        self.identical = float(rng.random() < c["identical"])
        self.spacing = rng.uniform(*c["spacing"])


class Anomaly:
    def __init__(self, rng, c, ws, t):
        self.ws, self.atype = ws, int(rng.integers(3))
        self.occ = rng.uniform(*c["occ"])
        # D455 + tag pose error grows with distance and occlusion
        self.pos_err = 1 + 1.5 * rng.uniform(*c["dist"]) + 4 * self.occ + abs(rng.normal(0, 0.5))
        acc = np.clip(0.98 - 0.4 * self.occ - 0.35 * (rng.random() < c["lookalike"]), 0.05, 1)
        self.class_ok = rng.random() < acc
        self.conf = float(np.clip(acc + rng.normal(0, 0.08), 0, 1))
        self.img_match = rng.random() < 0.92 - 0.4 * self.occ  # ponytail: replace with B's measured match rate
        self.hist, self.job = History(t), None
        self.b_saw = False  # truth: B drove past after it happened and saw it


class CommEnv(gym.Env):
    def __init__(self, condition=None, belief=True, alpha=0.1, beta=1.0, gamma=1.0,
                 n_ws=4, horizon=40, p_new=0.3, trace=False):
        self.condition, self.belief, self.trace = condition, belief, trace
        self.alpha, self.beta, self.gamma = alpha, beta, gamma  # cost per KB, per step-waiting, per redundant msg
        self.n_ws, self.horizon, self.p_new = n_ws, horizon, p_new
        self.observation_space = gym.spaces.Box(-1, 3, (OBS_DIM,), np.float32)
        # Uncertainty Only baseline: no "send nothing" option, no knowledge/history features
        self.action_space = gym.spaces.Discrete(N_ACTIONS if belief else N_ACTIONS - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        name = self.condition or list(CONDITIONS)[r.integers(len(CONDITIONS))]  # domain randomization
        self.c = CONDITIONS[name]
        self.bw, self.loss = r.uniform(*self.c["bw"]), r.uniform(*self.c["loss"])
        w = np.array(self.c["weights"], float)
        self.p_spot = np.clip(self.p_new * w / w.mean(), 0, 0.9)
        self.spots = [Spot(r, self.c) for _ in range(self.n_ws)]
        self.know = BKnowledge(self.n_ws)                                   # A's estimate
        self.b_obj, self.b_normal = np.zeros(self.n_ws, bool), np.zeros(self.n_ws, bool)  # B's truth
        self.t, self.ws, self.b_spot = 0, -1, HOME
        self.anom, self.inbox, self.queue, self.busy = {}, [], [], None
        self.stats = dict(condition=name, anomalies=0, fixed=0, times=[], bytes=0, msgs=0, images=0,
                          redundant=0, requests=0, know_err=[])
        self._r, self.log = 0.0, []
        if self._advance():  # whole patrol without a single anomaly: nothing to decide, redraw
            return self.reset()
        return self._obs(), {}

    def step(self, action):
        act = int(action) + (0 if self.belief else 1)
        a = self.anom[self.ws]
        self._r = 0.0
        self.stats["know_err"].append(abs(self.know.normal[a.ws] - self.b_normal[a.ws]))
        self._log("A", f"spot {a.ws}: {ATYPES[a.atype]} | B knows obj {self.know.obj[a.ws]:.2f} normal {self.know.normal[a.ws]:.2f} "
                       f"current {self.know.cur[a.ws]:.2f} | send: {describe(act)}")
        if act:
            redundant = a.job is not None and not a.job["failed"]
            self.stats["redundant"] += redundant
            self._r -= self.gamma * redundant
            items = items_of(act)
            n = BASE_BYTES + sum(ITEM_BYTES[i] for i in items)
            self._send(n, IMG in items)
            a.hist.on_send(items)
            if self.np_random.random() < self.loss:
                self._log("", f"  message lost ({n} B)")
            else:
                self.inbox.append((self.t + math.ceil(n / 1000 / self.bw / STEP_S), a, items))
        done = self._advance()
        if done:
            self._r -= 10 * len(self.anom)  # unresolved at end of patrol
            self._log("--", f"patrol over, unresolved at spots {sorted(self.anom)}")
        return self._obs(), self._r, done, False, {"stats": self.stats} if done else {}

    # --- internals ---
    def _log(self, who, msg):
        if self.trace:
            self.log.append(f"{self.t * STEP_S:5.0f}s  {who:<2} {msg}")

    def _spend(self, n):
        self.stats["bytes"] += n
        self._r -= self.alpha * n / 1000

    def _send(self, n, image):
        self._spend(n)
        self.stats["msgs"] += 1
        self.stats["images"] += image

    def _reply(self):
        """B -> A message (ACK / result / info request); costs bytes, may be lost."""
        self._spend(REPLY_BYTES)
        return self.np_random.random() >= self.loss

    def _learn(self, a, items):
        if ID in items:
            self.b_obj[a.ws] = True
        if NORM in items:
            self.b_normal[a.ws] = True

    def _identified(self, a, items):
        """Can B single out the target at the spot?"""
        s = self.spots[a.ws]
        if a.atype == TOPPLED or s.n_similar == 1 or a.b_saw:
            return True
        if CUR in items and a.pos_err <= s.spacing / 2:
            return True
        if IMG in items and a.img_match:
            return True
        # look-alikes that differ in detail: a correct description or B's own past look is enough
        return not s.identical and ((ID in items and a.class_ok) or self.b_obj[a.ws])

    def _advance(self):
        """Run time forward until A stands at a spot with an anomaly."""
        r = self.np_random
        while True:
            self._tick_b()
            self.t += 1
            if self.t >= self.horizon:
                return True
            for a in self.anom.values():
                a.hist.tick()
            self._r -= self.beta * len(self.anom)
            self.ws = (self.ws + 1) % self.n_ws
            if self.ws not in self.anom and r.random() < self.p_spot[self.ws]:
                self.anom[self.ws] = Anomaly(r, self.c, self.ws, self.t)
                self.know.on_new_anomaly(self.ws)
                s = self.spots[self.ws]
                self._log("!", f"spot {self.ws}: {ATYPES[self.anom[self.ws].atype]} happens "
                               f"(look-alikes {s.n_similar}{', identical' if s.identical and s.n_similar > 1 else ''})")
                self.stats["anomalies"] += 1
            if self.ws in self.anom:
                return False

    def _tick_b(self):
        r = self.np_random
        for msg in [m for m in self.inbox if m[0] <= self.t]:
            self.inbox.remove(msg)
            _, a, items = msg
            if self.anom.get(a.ws) is not a:
                continue  # already repaired
            self._learn(a, items)
            if self._reply():
                a.hist.on_ack()
                self.know.on_ack(a.ws, items)
            if a.job is None or a.job["failed"]:
                a.job = dict(items=set(items), failed=False, end=None, asked=False)
                self.queue.append(a)
            elif a.job["end"] is None:
                a.job["items"] |= items  # more info arrived before B started
        if self.busy and self.t >= self.busy.job["end"]:
            a = self.busy
            job = a.job
            missing = set()
            if not self._identified(a, job["items"]):
                missing.add(IMG)
            if NORM not in job["items"] and not self.b_normal[a.ws]:
                missing.add(NORM)
            if missing and not job["asked"]:
                # SC-3: B asks once; A answers with what is missing (latest stored crop / normal pose)
                job["asked"], job["end"] = True, self.t + 1
                self.stats["requests"] += 1
                self._log("B", f"spot {a.ws}: missing {[ITEMS[i] for i in missing]} -> asks A")
                if self._reply():
                    self._send(BASE_BYTES + sum(ITEM_BYTES[i] for i in missing), IMG in missing)
                    a.hist.items |= missing
                    if r.random() >= self.loss:
                        job["items"] |= missing
                        self._learn(a, missing)
                return
            reason = OK if not missing else NEED_ID if IMG in missing else NEED_NORMAL
            if reason == OK and not job.get("reported"):
                job["reported"] = True  # B -> laptop text instruction; not counted as A<->B traffic
                self._log("B", f"spot {a.ws}: sends repair text to laptop, human fixes")
            if reason == OK and r.random() > P_FIX:  # B's check fails: human redoes it, B checks again
                self._log("B", f"spot {a.ws}: check failed, human redoes the fix")
                job["end"] = self.t + 1
                return
            self.busy = None
            seen_back = self._drive_past(a.ws)  # drives back to standby along the same line
            self.b_spot = HOME
            self._log("B", f"spot {a.ws}: " + ("fix confirmed" if reason == OK else f"gives up ({['ok', 'need id', 'need normal'][reason]})")
                      + f", back to standby, sees spots {seen_back} (hidden from A)")
            if reason == OK:
                del self.anom[a.ws]
                self.b_obj[a.ws] = self.b_normal[a.ws] = True
                self.stats["fixed"] += 1
                self.stats["times"].append((self.t - a.hist.t0) * STEP_S)
                self._r += 10
            else:
                job["failed"] = True
                self._r -= 3  # wasted B trip; must stay < unresolved penalty or sending nothing wins
            if self._reply():  # result carries the route, so A can update what B may have seen
                a.hist.on_result(reason)
                if reason == OK:
                    self.know.on_success(a.ws)
                seen = {k for k, x in self.anom.items() if x.hist.t0 <= job["t_start"]}
                q = self.c["vis"]  # A's visibility table; B passes each spot twice (out and back)
                passed = route_spots(HOME, a.ws)
                self.know.on_pass({k: 1 - (1 - q) ** 2 for k in passed}, seen)
                for k in passed:  # ponytail: event level, A is taken to have seen these spots normal since
                    self.know.on_spot_seen(k, k not in seen)
        if self.busy is None and self.queue:
            a = self.queue.pop(0)
            a.job["t_start"] = self.t
            seen = self._drive_past(a.ws)
            a.job["end"] = self.t + int(r.integers(*self.c["fix"], endpoint=True))
            self._log("B", f"drives standby -> {a.ws} with {sorted(ITEMS[i] for i in a.job['items'])}, "
                           f"sees spots {seen} on the way (hidden from A)")
            self.busy, self.b_spot = a, a.ws

    def _drive_past(self, to):
        """Truth: B sees each spot between standby and `to` with probability vis. Hidden from A."""
        seen = []
        for k in route_spots(HOME, to):
            if self.np_random.random() < self.c["vis"]:
                seen.append(k)
                self.b_obj[k] = True
                if k in self.anom:
                    self.anom[k].b_saw = True
                else:
                    self.b_normal[k] = True
        return seen

    def _obs(self):
        a = self.anom.get(self.ws)
        if a is None:
            return np.zeros(OBS_DIM, np.float32)
        s = self.spots[self.ws]
        o = make_obs(a.conf, a.occ, s.n_similar, a.pos_err, s.spacing, s.identical, a.atype, self.ws, a.hist,
                     self.know, self.b_spot == self.ws, self.busy is not None, len(self.queue), self.bw, self.loss)
        if not self.belief:
            o[BELIEF] = 0
        return o


def describe(act):
    return "nothing" if act == 0 else "base" + "".join(" + " + ITEMS[i] for i in sorted(items_of(act)))


def run(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, done, _, info = env.step(policy(obs))
        total += r
    return total, info["stats"]


def fix_rate(S):
    return sum(x["fixed"] for x in S) / max(sum(x["anomalies"] for x in S), 1)


if __name__ == "__main__":
    # self-check: logic breaks -> one of these fails
    from features import BKnowledge as K, History as H
    r, s = run(CommEnv("normal"), lambda o: 0, 0)
    assert s["bytes"] == 0 and s["fixed"] == 0 and r < 0, s
    full = [run(CommEnv("normal"), lambda o: FULL, i)[1] for i in range(50)]
    assert fix_rate(full) > 0.6, fix_rate(full)
    # clustered identical look-alikes: base-only must lean on info requests, full sharing must not
    base = [run(CommEnv("clustered"), lambda o: 1, i)[1] for i in range(50)]
    full_c = [run(CommEnv("clustered"), lambda o: FULL, i)[1] for i in range(50)]
    assert sum(x["requests"] for x in base) > 3 * max(sum(x["requests"] for x in full_c), 1)
    # A's estimate: a certain pass marks the in-between spot as seen, the anomaly there as known
    k = K(4)
    k.on_pass({j: 1.0 for j in route_spots(HOME, 3)}, {2})
    k.on_spot_seen(0, True), k.on_spot_seen(1, True)
    assert k.obj[0] == k.normal[1] == 1 and k.normal[2] == 0 and k.cur[2] == 1
    # B always ends a job back at standby
    e = CommEnv("normal"); e.reset(seed=0); done = False
    while not done:
        _, _, done, _, _ = e.step(FULL)
    assert e.busy is not None or e.b_spot == HOME
    env = CommEnv(belief=False)
    o, _ = env.reset(seed=1)
    assert env.action_space.n == 16 and not o[BELIEF].any()
    print("ok | normal full fix:", round(fix_rate(full), 2), "| clustered fix base/full:",
          round(fix_rate(base), 2), round(fix_rate(full_c), 2))
