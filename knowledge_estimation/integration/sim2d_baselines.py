"""Prior-work receiver-knowledge estimators vs this estimator, under the same team rule policy (team 2D simulator).

python -m knowledge_estimation.integration.sim2d_baselines --seed0 600000 --episodes 200

Every method is the team rule (rule_ppo_compare/calibrated.py) reading a different estimate of B's knowledge; the
simulator, costs and seeds are the same. The two prior-work estimators are simplified re-implementations of their
belief-update rules in this setting (what was kept and dropped is listed in SIM2D_RESULTS.md section 9):

  mue     Message utility estimation (arXiv 2304.06873): A keeps the set of data it believes B holds. A broadcast
          is assumed received (no handshake); data in messages received from B is added (ACK), and B's own
          messages that say it lacks data (info request) or lost it (memory epoch) remove it. No pass-by, no
          content validity, no expiry.
  tom     Theory-of-mind assistive communication (Buehler et al., arXiv 2109.01355): per-item belief probability,
          Bayesian update from B's observations with a perception model (here the same planned-frame model as
          ours, independent union), B's actions as evidence (request -> lacks it, verified repair -> holds it,
          wrong object -> wrong look); a communication certainly updates the belief. No message loss, no
          forgetting, no wrong learning while passing, no calibration.
  core_raw  this estimator without the b_cal score calibration (ablation)
  oracle    B's true holding at read time (upper bound; not available to A)

The simulator is not modified: the estimators are attached to World at construction and receive the same event
calls as the core estimator (World._knows for all, World._core for the modes listed in CORE_MODES).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl"), str(ROOT / "rule_ppo_compare"), str(ROOT)]

from knowledge_estimation.integration.sim2d import CUR, IMG, NORM  # noqa: E402

OUT = ROOT / "knowledge_estimation" / "data" / "baselines_v1"
TAU = 0.8
MODES = ("cache", "optimistic", "mue", "tom", "core", "core_raw", "oracle")
NAMES = {"cache": "Confirmed Cache", "optimistic": "optimistic", "mue": "message utility (2023)",
         "tom": "ToM Bayesian (Buehler 2021)", "core": "ours", "core_raw": "ours without b_cal", "oracle": "oracle"}


class _Base:
    """Per-spot arrays obj (I), normal (N), cur (C) like features.BKnowledge, plus the World event interface."""

    def __init__(self, n):
        self.obj, self.normal, self.cur = np.zeros(n), np.zeros(n), np.zeros(n)

    def _sent(self, k, items, p=1.0):
        if NORM in items:
            self.normal[k] = max(self.normal[k], p)
        if items & {CUR, IMG}:
            self.cur[k] = max(self.cur[k], p)
        if IMG in items:
            self.obj[k] = max(self.obj[k], p)

    # World._knows (every estimator)
    def on_ack(self, spot, items):
        self._sent(spot, set(items))

    def on_new_anomaly(self, spot):
        self.cur[spot] = 0.0  # a new event: its C is new data B cannot hold yet

    def on_missing(self, spot, items):
        if IMG in items:
            self.obj[spot] = 0.0
        if NORM in items:
            self.normal[spot] = 0.0
        if CUR in items:
            self.cur[spot] = 0.0

    def on_pass(self, qs, open_spots):
        pass

    def on_spot_seen(self, spot, normal):
        pass

    def on_success(self, spot):
        pass

    def on_wrong(self, spot):
        pass

    def reset(self):
        pass

    # World._core (CORE_MODES)
    def send(self, k, payload, items, attempt):
        pass

    def ack(self, payload, epoch):
        pass

    def trip(self, k, open_spots, route_spots):
        pass

    def pose_measured(self, k, cur_ver, observed_at):
        self.cur[k] = 0.0  # a new C version

    def decision(self, k):
        pass


class MessageUtility(_Base):
    """Sent = held (no handshake); B's messages add (ACK) or remove (request, memory epoch) data."""

    def send(self, k, payload, items, attempt):
        self._sent(k, set(items))

    def reset(self):  # B's STATUS / ACK carried a new memory epoch: B says its data is gone
        for v in (self.obj, self.normal, self.cur):
            v[:] = 0.0


class TheoryOfMind(_Base):
    """Bayesian per-item belief from B's observations and actions; communication is a certain update."""

    def __init__(self, n, world):
        super().__init__(n)
        self.world = world

    def send(self, k, payload, items, attempt):
        self._sent(k, set(items))

    def trip(self, k, open_spots, route_spots):
        core = self.world.knows["core"].core
        frames = core.acquisition.frames
        entry = core.parameters.entry("p_frame", dict(receiver_id="B", kind="", site_id="", route_id="",
                                                      condition_id="in_view"))
        for j in route_spots:
            if j == k:
                continue
            p = 1 - (1 - entry["value"]) ** frames.get((f"to{k}", f"S{j}"), 0)
            if j in open_spots:  # B observed the abnormal state, not the normal layout
                self.cur[j] = 1 - (1 - self.cur[j]) * (1 - p)
            else:
                self.normal[j] = 1 - (1 - self.normal[j]) * (1 - p)
                self.obj[j] = 1 - (1 - self.obj[j]) * (1 - p)

    def on_success(self, spot):  # B's action: it handled the object, so it knows the layout and the look
        self.normal[spot] = self.obj[spot] = 1.0

    def on_wrong(self, spot):  # B's action: it picked the other object, so its look is wrong
        self.obj[spot] = 0.0


class _Truth:
    def __init__(self, world, kind):
        self.world, self.kind = world, kind

    def __getitem__(self, k):
        w, k = self.world, int(k)
        if self.kind == "N":
            return float(w.b_normal[k])
        if self.kind == "I":
            return float(w.b_obj[k] and not w.b_wrong[k])
        a = w.anom.get(k)  # same definition as World._record's K for C
        if a is None:
            return 0.0
        job = a.job
        return float(a.b_saw or bool(job and not job["failed"] and job["items"] & {CUR, IMG}
                                     and job["cur_ver"] == a.moved))

    def __len__(self):
        return len(self.world.spots)


class Oracle(_Base):
    def __init__(self, n, world):
        super().__init__(n)
        self.obj, self.normal, self.cur = _Truth(world, "I"), _Truth(world, "N"), _Truth(world, "C")

    def on_ack(self, spot, items):
        pass

    def on_new_anomaly(self, spot):
        pass

    def on_missing(self, spot, items):
        pass

    def pose_measured(self, k, cur_ver, observed_at):
        pass


def attach():
    """Give every World the comparison estimators and route the core-only events to them too."""
    import world as W
    from knowledge_estimation.integration.sim2d import CoreKnowledge
    if getattr(W.World, "_baselines_attached", False):
        return
    W.World._baselines_attached = True
    init = W.World.__init__

    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        n, L = len(self.spots), W.LAYOUT
        self.knows["mue"] = MessageUtility(n)
        self.knows["tom"] = TheoryOfMind(n, self)
        self.knows["core_raw"] = CoreKnowledge(self, W.CORE_PARAMETERS, L["spots"], None, W.CORE_CAPS, level="raw",
                                               plans=W.a_plans(), camera=W.Camera(L["fov"], *L["depth"]))
        self.knows["oracle"] = Oracle(n, self)

    W.World.__init__ = patched
    W.CORE_MODES = ("core", "core_table", "mue", "tom", "core_raw", "oracle")


def run(job):
    mode, cond, seeds = job
    attach()
    from env2d import Sim2DEnv
    from calibrated import rule_calibrated
    eps, decisions = [], []
    for seed in seeds:
        env = Sim2DEnv(cond)
        env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            _, r, done, _, info = env.step(rule_calibrated(env, mode))
            total += r
        s = info["stats"]
        eps.append(dict(mode=mode, condition=cond, seed=seed, reward=total, kb=s["bytes"] / 1000,
                        fixed=s["fixed"], anomalies=s["anomalies"], wrong=s["wrong_ops"], requests=s["requests"],
                        images=s["images"], pulls=s["pulls"]))
        if mode == "core":  # every estimator's value at the same decisions, for paired accuracy
            for d in s["dec"]:
                decisions.append(dict(condition=cond, seed=seed, K=d["K"],
                                      est={m: d["est"][m] for m in MODES}))
    return eps, decisions


def paired(eps, mode, key, cond=None):
    ref = {(e["condition"], e["seed"]): e[key] for e in eps if e["mode"] == "core"}
    d = np.array([e[key] - ref[(e["condition"], e["seed"])] for e in eps
                  if e["mode"] == mode and (cond is None or e["condition"] == cond)])
    se = d.std(ddof=1) / math.sqrt(len(d))
    p = math.erfc(abs(d.mean() / se) / math.sqrt(2)) if se > 0 else float("nan")
    return float(d.mean()), float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se), p


def accuracy(decisions, rng, reps=1000):
    """Brier per item and high-estimate-unheld (b >= TAU, K = 0) on the same decisions, episode bootstrap CI."""
    out = {}
    episodes = sorted({(d["condition"], d["seed"]) for d in decisions})
    index = {e: i for i, e in enumerate(episodes)}
    for mode in MODES:
        out[mode] = {}
        for j, kind in enumerate("NIC"):
            sums = np.zeros((len(episodes), 2))
            hi = [0, 0]
            for d in decisions:
                b, k = float(d["est"][mode][j]), float(d["K"][j])
                row = sums[index[(d["condition"], d["seed"])]]
                row += ((b - k) ** 2, 1)
                if b >= TAU:
                    hi[0] += 1
                    hi[1] += k == 0
            boot = sums[rng.integers(0, len(sums), (reps, len(sums)))]
            out[mode][kind] = dict(brier=float(sums[:, 0].sum() / sums[:, 1].sum()),
                                   ci=[float(np.percentile(boot[..., 0].sum(1) / boot[..., 1].sum(1), q)) for q in (2.5, 97.5)],
                                   high_unheld=hi[1] / hi[0] if hi[0] else None, n_high=hi[0])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed0", type=int, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    from env2d import CONDITIONS
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    jobs = [(m, c, seeds) for m in MODES for c in CONDITIONS]
    with Pool(min(len(jobs), 20)) as pool:
        results = pool.map(run, jobs)
    eps = [e for r, _ in results for e in r]
    decisions = [d for _, r in results for d in r]
    acc = accuracy(decisions, np.random.default_rng(0))

    print(f"seeds {seeds[0]}..{seeds[-1]} x {len(CONDITIONS)} conditions, team rule reading each estimate\n")
    print("estimator accuracy on the same decisions (rule reading ours), Brier [95% CI] / high-estimate unheld")
    for mode in MODES:
        a = acc[mode]
        print(f"  {NAMES[mode]:<28}" + "  ".join(
            f"{k} {a[k]['brier']:.3f} [{a[k]['ci'][0]:.3f},{a[k]['ci'][1]:.3f}] "
            f"{'-' if a[k]['high_unheld'] is None else format(a[k]['high_unheld'], '.3f')}" for k in "NIC"))
    table = {}
    for cond in CONDITIONS:
        print(f"\n== {cond} ==  {'method':<28}{'reward':>8}{'KB':>7}{'wrong':>7}{'reqs':>6}{'imgs':>6}{'succ':>6}"
              f"   reward vs ours [95% CI] p        KB vs ours")
        for mode in MODES:
            E = [e for e in eps if e["mode"] == mode and e["condition"] == cond]
            row = dict(reward=np.mean([e["reward"] for e in E]), kb=np.mean([e["kb"] for e in E]),
                       wrong=np.mean([e["wrong"] for e in E]), requests=np.mean([e["requests"] for e in E]),
                       images=np.mean([e["images"] for e in E]),
                       success=sum(e["fixed"] for e in E) / max(sum(e["anomalies"] for e in E), 1))
            line = (f"   {'':<10}{NAMES[mode]:<28}{row['reward']:>8.2f}{row['kb']:>7.1f}{row['wrong']:>7.3f}"
                    f"{row['requests']:>6.2f}{row['images']:>6.2f}{row['success']:>6.3f}")
            if mode != "core":
                row["vs_ours_reward"], row["vs_ours_kb"] = paired(eps, mode, "reward", cond), paired(eps, mode, "kb", cond)
                r, k = row["vs_ours_reward"], row["vs_ours_kb"]
                line += f"   {r[0]:+6.2f} [{r[1]:+.2f},{r[2]:+.2f}] p={r[3]:.3f}   {k[0]:+6.1f} p={k[3]:.3f}"
            print(line)
            table.setdefault(cond, {})[mode] = {k: v for k, v in row.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(dict(seeds=[seeds[0], seeds[-1]], accuracy=acc, comm=table),
                                                      indent=1, default=float), encoding="utf-8")
    with (args.out / "episodes.json").open("w", encoding="utf-8") as f:
        json.dump(eps, f)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
