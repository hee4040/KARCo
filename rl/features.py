"""State vector + A's estimate of what B knows. Shared by the simulator and the real Robot A,
so the policy sees the same features in both. Everything here uses only what A can observe:
its own sends, B's ACK / result / info-request messages, and the route B took.

A -> B message = base (event id, spot, anomaly, repair request) + any of ITEMS
B -> A: ACK {msg_id, b_spot, b_busy, queue}, result {msg_id, ok | need_id | need_normal, from_spot, t_start},
        info request {msg_id, spot, need: [...]}
"""
import numpy as np

ID, CUR, NORM, IMG = range(4)                 # optional items: target id/description, current pose, normal pose, image crop
ITEMS = ["id", "cur", "norm", "img"]
OK, NEED_ID, NEED_NORMAL = 0, 1, 2            # B's task_result reasons (grasp failures B retries itself)
OBS_DIM = 27
BELIEF = slice(8, 22)                         # features zeroed for the "Uncertainty Only" baseline
Q_LEARN = 0.9                                 # P(B learns the right appearance from handling an object)


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
        self.est_current = 0.0            # P(B already knows this anomaly's current state)

    def on_send(self, items):
        self.n_sent += 1
        self.items, self.ack, self.since, self.result, self.reason = set(items), False, 0, 0, OK

    def on_ack(self):
        self.ack = True
        if self.items & {CUR, IMG}:
            self.est_current = 1.0

    def on_result(self, reason):
        self.result, self.reason = (1 if reason == OK else -1), reason

    def tick(self):
        self.since += 1


class BKnowledge:
    """A's per-spot estimate of B's knowledge: does B know the target object / the normal layout."""

    def __init__(self, n_spots):
        self.obj = np.zeros(n_spots)
        self.normal = np.zeros(n_spots)

    def on_ack(self, spot, items):
        if ID in items:
            self.obj[spot] = 1.0
        if NORM in items:
            self.normal[spot] = 1.0

    def on_success(self, spot):
        self.obj[spot] = max(self.obj[spot], Q_LEARN)  # B handled it; may still have learned the wrong look
        self.normal[spot] = 1.0

    def on_route(self, frm, to, q, open_anomalies):
        """B drove frm -> to; it saw each spot on the way with probability q (A's visibility table)."""
        self.on_pass({k: q for k in route_spots(frm, to)}, open_anomalies)

    def on_pass(self, qs, open_anomalies):
        """B drove past spots; it saw spot k with probability qs[k]."""
        for k, q in qs.items():
            self.obj[k] = noisy_or(self.obj[k], q)
            if k in open_anomalies:  # B saw the anomaly, not the normal layout
                open_anomalies[k].est_current = noisy_or(open_anomalies[k].est_current, q)
            else:
                self.normal[k] = noisy_or(self.normal[k], q)


def pending(h, loss):
    """P(B already holds this task), from history only."""
    if h.n_sent == 0 or h.result == -1:
        return 0.0
    return (0.9 if h.ack else 0.5 * (1 - loss)) * 0.9 ** h.since  # ponytail: hand-tuned decay, fit to real logs


def make_obs(conf, occ, n_similar, pos_err_cm, spacing_cm, identical, atype, spot, h, know,
             b_near, b_busy, queue, bw_kbps, loss):
    """conf: A's class confidence. n_similar: look-alike objects at the spot. identical: look-alikes are exact copies.
    pos_err_cm: A's position uncertainty. spacing_cm: gap to the nearest look-alike."""
    v = np.zeros(OBS_DIM, np.float32)
    v[0:5] = conf, occ, min(n_similar, 4) / 4, min(pos_err_cm / spacing_cm, 2) / 2, identical
    v[5 + atype] = 1
    v[8:12] = know.obj[spot], know.normal[spot], h.est_current, pending(h, loss)
    v[12:16] = min(h.n_sent, 3) / 3, h.ack, min(h.since, 10) / 10, h.result
    for i in h.items:
        v[16 + i] = 1
    v[20] = h.reason == NEED_ID
    v[21] = h.reason == NEED_NORMAL
    v[22:27] = b_near, b_busy, min(queue, 3) / 3, np.log10(bw_kbps) / 3, loss
    return v
