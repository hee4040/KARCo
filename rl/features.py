"""State vector + A's estimate of what B knows. Shared by the simulator and the real Robot A,
so the policy sees the same features in both. Everything here uses only what A can observe:
its own sends, B's ACK / result / info-request messages, and the route B took (from A's own command).

A -> B message = base (event id, spot, anomaly, repair request) + any of ITEMS
B -> A: ACK {msg_id, stored items, memory_epoch, needs_info}, result {msg_id, ok | need_id | need_normal | not_found},
        info request {msg_id, spot, need: [...]}, STATUS {memory_epoch}
"""
import numpy as np

ID, CUR, NORM, IMG = range(4)                 # optional items: target id/description, current pose, normal pose, image crop
ITEMS = ["id", "cur", "norm", "img"]
OK, NEED_ID, NEED_NORMAL = 0, 1, 2            # B's task_result reasons
OBS_DIM = 27
BELIEF = slice(8, 22)                         # features zeroed for the "Uncertainty Only" baseline
Q_LEARN = 0.9                                 # P(B learns the right appearance from handling an object)
R_CROP = 0.95                                 # P(B recognises the target from a received crop)
# estimator variants (scenario doc comparison methods); all share ACK handling and the event gate
MODES = ("proposed", "cache", "cache_result", "optimistic")


def items_of(action):
    """Action 1..16 -> set of optional items (bit i = ITEMS[i]). Action 0 = send nothing."""
    return {i for i in range(4) if (action - 1) >> i & 1}


def route_spots(frm, to):
    """Spots B drives past between two points on the line (standby = -1, before spot 0).
    ponytail: spots on one straight line; replace with a table from the real line layout."""
    lo, hi = sorted((frm, to))
    return range(lo + 1, hi)


def noisy_or(p, q):
    return 1 - (1 - p) * (1 - q)


class History:
    """A's record of one anomaly: what it sent and what B answered."""

    def __init__(self, t):
        self.t0, self.n_sent, self.items, self.ack, self.since = t, 0, set(), False, 0
        self.result, self.reason = 0, OK  # result: -1 fail, 0 none, 1 ok

    def on_send(self, items):
        self.n_sent += 1
        self.items, self.ack, self.since, self.result, self.reason = set(items), False, 0, 0, OK

    def on_ack(self):
        self.ack = True

    def on_result(self, reason):
        self.result, self.reason = (1 if reason == OK else -1), reason

    def tick(self):
        self.since += 1


class BKnowledge:
    """A's per-spot estimate of B's knowledge: target appearance (obj), normal layout (normal), and the current state
    of the open anomaly there (cur). mode picks the estimator compared in the scenario doc:
    proposed     -- messages + verified results + what B passed (held until A sees the spot normal again)
    cache        -- Confirmed Cache: only items B acknowledged
    cache_result -- Confirmed Cache + Result: also what A verified B handled
    optimistic   -- 'B passed it, so B knows it' (no hold, no probability)"""

    def __init__(self, n_spots, mode="proposed"):
        self.mode = mode
        self.obj, self.normal, self.cur = np.zeros(n_spots), np.zeros(n_spots), np.zeros(n_spots)
        self.hold = np.zeros(n_spots)  # pass-by evidence waiting for A to confirm the spot was normal

    def on_ack(self, spot, items):
        if NORM in items:
            self.normal[spot] = 1.0
        if items & {CUR, IMG}:
            self.cur[spot] = 1.0
        if IMG in items:
            self.obj[spot] = max(self.obj[spot], R_CROP)

    def on_success(self, spot):
        """A verified the repair B reported."""
        if self.mode != "cache":
            self.obj[spot] = max(self.obj[spot], Q_LEARN)
            self.normal[spot] = 1.0

    def on_new_anomaly(self, spot):
        self.cur[spot] = 0.0

    def on_pass(self, qs, open_spots):
        """B drove past spots; it saw spot k with probability qs[k] (A's visibility table)."""
        if self.mode in ("cache", "cache_result"):
            return
        for k, q in qs.items():
            if self.mode == "optimistic":
                self.obj[k] = self.normal[k] = 1.0
                if k in open_spots:
                    self.cur[k] = 1.0
            elif k in open_spots:  # B saw the anomaly, not the normal layout
                self.cur[k] = noisy_or(self.cur[k], q)
            else:
                self.hold[k] = noisy_or(self.hold[k], q)

    def on_spot_seen(self, spot, normal):
        """A observed the spot: normal -> the held pass-by snapshot was a normal one; abnormal -> drop it."""
        if normal:
            self.obj[spot] = noisy_or(self.obj[spot], self.hold[spot])
            self.normal[spot] = noisy_or(self.normal[spot], self.hold[spot])
        self.hold[spot] = 0.0

    def on_missing(self, spot, items):
        """B asked for items (or reported not_found): it does not hold them."""
        if IMG in items:
            self.obj[spot] = 0.0
        if NORM in items:
            self.normal[spot] = 0.0
        if CUR in items:
            self.cur[spot] = 0.0

    def on_wrong(self, spot):
        """A's re-check found B handled the wrong object: what B learned there is void."""
        self.obj[spot] = 0.0

    def reset(self):
        """B's memory epoch changed: everything B learned is gone."""
        for v in (self.obj, self.normal, self.cur, self.hold):
            v[:] = 0.0


def pending(h, loss):
    """P(B already holds this task), from history only. Without an ACK: P(message arrived | no ACK) with
    message and ACK each lost with probability `loss` (scenario doc 1(b))."""
    if h.n_sent == 0 or h.result == -1:
        return 0.0
    p = 1 - loss
    got = 0.9 if h.ack else (p * (1 - p) / (1 - p * p) if p < 1 else 0.5)
    return got * 0.9 ** h.since  # ponytail: hand-tuned decay, fit to real logs


def make_obs(conf, occ, n_similar, pos_err_cm, spacing_cm, identical, atype, spot, h, know,
             b_near, b_busy, queue, bw_kbps, loss):
    """conf: A's class confidence. n_similar: look-alike objects at the spot. identical: look-alikes are exact copies.
    pos_err_cm: A's position uncertainty. spacing_cm: gap to the nearest look-alike."""
    v = np.zeros(OBS_DIM, np.float32)
    v[0:5] = conf, occ, min(n_similar, 4) / 4, min(pos_err_cm / spacing_cm, 2) / 2, identical
    v[5 + atype] = 1
    v[8:12] = know.obj[spot], know.normal[spot], know.cur[spot], pending(h, loss)
    v[12:16] = min(h.n_sent, 3) / 3, h.ack, min(h.since, 10) / 10, h.result
    for i in h.items:
        v[16 + i] = 1
    v[20] = h.reason == NEED_ID
    v[21] = h.reason == NEED_NORMAL
    v[22:27] = b_near, b_busy, min(queue, 3) / 3, np.log10(bw_kbps) / 3, loss
    return v
