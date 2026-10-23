"""Rule-based expected-cost policy (rule_based_comm, unchanged) with sizes, costs and outcome model calibrated to sim/.

The rule's structure stays: requirements -> 8 E/N/C/I candidates -> expected cost -> argmin. Only the numbers change:
- message sizes and cost weights = the simulator's reward (bytes, waiting time, wrong-object penalty, failed trip)
- outcome model = how B in sim/world.py picks its target (World._pick), using only what A can observe
The estimate of B's knowledge it reads is chosen by `mode` (features.MODES): this is how the scenario doc's
Confirmed Cache / Cache + Result / optimistic / proposed methods differ -- same decision rule, different estimator.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "sim"), str(ROOT / "rl")]
from rule_based_comm.config import CostWeights, MessageSizes, PolicyConfig  # noqa: E402
from rule_based_comm.cost_model import ExpectedCostModel  # noqa: E402
from rule_based_comm.policy_adapter import ACTIONS, RulePolicyAdapter  # noqa: E402
from rule_based_comm.rule_policy import RuleBasedCommunicationPolicy  # noqa: E402
from rule_based_comm.schemas import Action, OutcomeEstimate  # noqa: E402
from features import CUR, ID, IMG, NORM  # noqa: E402
from world import BASE_BYTES, ITEM_BYTES, LAYOUT, REOBSERVE as REOBS_COST, REQUEST, TOPPLED, WRONG, WRONG_OP  # noqa

BIT = {"E": 1, "N": 4, "C": 2, "I": 8}  # E carries the target id -> our base + ID
FAIL = 3 + 30 * 1.0 / 5         # B gives up: -3, then ~30 s (one patrol lap) of waiting until A resends
EXTRA_WRONG = 50 * 1.0 / 5      # unreported wrong object: ~30 s until A's re-check + ~20 s second trip
REOBSERVE = REOBS_COST + 30 * 1.0 / 5  # not_found: A re-observes, B waits ~30 s for the POSE_UPDATE
_Z = np.random.default_rng(0).standard_normal((4000, 2))
_BIG = np.where(np.random.default_rng(1).random(4000) < LAYOUT["p_overconf"], LAYOUT["overconf"], 1.0)
_ang = np.random.default_rng(2).uniform(0, 2 * np.pi, 4000)
_MOVE = np.stack([np.cos(_ang), np.sin(_ang)], 1) * np.random.default_rng(3).uniform(*LAYOUT["move_off"], 4000)[:, None] * 100


def to_action(items):
    return 1 + sum(BIT[i] for i in items)


def cur_cue(sigma, dist, p_moved=0.0):
    """Who lands within B's 2-sigma search radius around the reported 'cur' (World._pick), with A's error model
    (sigma sometimes 3x too small) and, with probability p_moved, the target moved again after the report (4(d)):
    P(nobody), P(target only), P(other only), P(both)."""
    if not np.isfinite(dist):
        return dict(none=0.0, t=1.0, o=0.0, both=0.0)
    rep = _Z * (sigma * _BIG / np.sqrt(2))[:, None]  # reported position (sigma = RMS distance), target at 0 (cm)
    moved = np.arange(len(rep)) < p_moved * len(rep)  # the samples are i.i.d., so the first share stands for 'moved'
    tgt = np.where(moved[:, None], _MOVE, 0.0)
    t = np.hypot(*(rep - tgt).T) <= 2 * sigma
    o = np.hypot(rep[:, 0] - dist, rep[:, 1]) <= 2 * sigma
    return dict(none=float(np.mean(~t & ~o)), t=float(np.mean(t & ~o)), o=float(np.mean(o & ~t)),
                both=float(np.mean(t & o)))


class SimOutcomeModel(ExpectedCostModel):
    ctx = {}

    def predict_outcome(self, candidate, event, belief, requirements):
        c = self.ctx
        no_img = "I" not in candidate
        if c["trivial"]:  # toppled / recoloured / single object: B sees which one
            correct, wrong, unsure, reobs = 1.0, 0.0, 0.0, 0.0
        else:  # World._pick: saw it happen -> (search radius if C) -> crop -> own memory of the look -> unique -> ask
            img = 0.0 if no_img else c["img_p"]
            look = img + (1 - img) * belief.b_I  # b_I folds in wrongly learned looks
            if "C" in candidate:
                r = c["cue"]
                ok = r["t"] + r["both"] * look            # target alone in the radius, or told apart there
                wr = r["o"] * (1 - img)                   # only the other one in the radius
                nf = r["none"] + r["o"] * img             # nothing there, or the crop matches nobody there
                un = r["both"] * (1 - look)
            else:
                ok, wr, nf, un = look, 0.0, 0.0, 1 - look
            seen = belief.b_C                             # B saw it happen while passing
            correct, wrong, unsure = seen + (1 - seen) * ok, (1 - seen) * wr, (1 - seen) * un
            reobs = (1 - seen) * nf                       # not_found -> A re-observes -> POSE_UPDATE
        need_n = 0.0 if "N" in candidate else 1 - belief.b_N
        p_req = 1 - (1 - need_n) * (1 - unsure)  # N is asked at the ACK, the crop on site
        fail = unsure * ((1 - c["img_p"]) if no_img else 1.0)  # a fetched crop may still not match
        return OutcomeEstimate(
            expected_requests=p_req, p_wrong=wrong, p_unresolved=fail, expected_reobservations=reobs,
            expected_followup_bytes=need_n * ITEM_BYTES[NORM] + (unsure + wrong) * no_img * ITEM_BYTES[IMG])


def rule_calibrated(env, mode="proposed"):
    """mode: estimator of B's knowledge (features.MODES), or None = scene-only rule."""
    w = env.world
    a = w.pending[0]
    s = w.spots[a.spot]
    inp = env.rule_input(mode)
    others = [o for o in s.objs if o is not a.obj]
    dist = min((np.hypot(*(a.cur_rep - o.normal)) * 100 for o in others), default=np.inf)  # A's measurement vs layout
    cue = cur_cue(a.pos_err, dist, LAYOUT["p_move_again"] * (a.atype == 0))
    cfg = PolicyConfig(
        message_sizes=MessageSizes(E=BASE_BYTES + ITEM_BYTES[ID], N=ITEM_BYTES[NORM], C=ITEM_BYTES[CUR],
                                   I=ITEM_BYTES[IMG]),
        costs=CostWeights(transmission_per_kib=1.024 * (env.alpha + env.beta / 5 / w.bw),  # bytes + transfer time
                          request_roundtrip=REQUEST + 0.1, wrong_action=WRONG_OP + EXTRA_WRONG, reobservation=REOBSERVE,
                          unresolved=FAIL))
    policy = RuleBasedCommunicationPolicy(cfg)
    policy.cost_model = SimOutcomeModel(cfg)
    policy.cost_model.ctx = dict(trivial=a.atype in (TOPPLED, WRONG) or s.n_similar == 1,
                                 img_p=float(np.clip(0.92 - 0.4 * a.occ, 0, 1)), cue=cue)
    d = policy.select(inp, inp)
    return to_action(d.selected_items) if d.action is Action.SEND else 0


_default = RulePolicyAdapter()


def rule_default(env, mode="proposed"):
    """The rule exactly as on heewoo/rule-based (its own sizes and probabilities)."""
    act, _ = _default.predict(env.rule_input(mode))
    return 0 if act is None else to_action(ACTIONS[act])


if __name__ == "__main__":
    c1, c3 = cur_cue(1.0, 4.0), cur_cue(3.0, 4.0)
    assert abs(sum(c1.values()) - 1) < 1e-9 and c1["t"] > 0.7 and c1["o"] < 0.1, c1  # sigma 1 cm: cur alone works
    assert c3["both"] > 0.4, c3                                                      # sigma 3 cm: both in radius
    print("ok | cur cue sigma 1 cm:", {k: round(v, 2) for k, v in c1.items()},
          "| sigma 3 cm:", {k: round(v, 2) for k, v in c3.items()})
