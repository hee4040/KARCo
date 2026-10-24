"""Choose the b_safe caps (Notion section 12 policy parameter) for the team rule policy on validation seeds.

python -m knowledge_estimation.integration.sim2d_rule_select --episodes 200

The team rule (rule_ppo_compare/calibrated.py) reads b_safe from this estimator. Each candidate cap on b_I is run
on the same validation seeds (120000+, never the final ones), and the cap with the best mean team reward (the
scenario document's cost weights) is written to sim2d_policy_v2.json, with the paired difference to "no cap".
The I cap is the only candidate because I is where high estimates were most often unheld (SIM2D_RESULTS.md).
Ties keep the first candidate ("none"): a cap that changes no decision is not adopted.
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl"), str(ROOT / "rule_ppo_compare"), str(ROOT)]

CANDIDATES = {"none": {}, "I<=0.9": {"b_I": 0.9}, "I<=0.8": {"b_I": 0.8}, "I<=0.6": {"b_I": 0.6}}


def run(job):
    name, caps, seeds = job
    import world as W
    W.CORE_CAPS = caps
    from env2d import CONDITIONS, Sim2DEnv
    from calibrated import rule_calibrated
    out = []
    for cond in CONDITIONS:
        for seed in seeds:
            env = Sim2DEnv(cond)
            env.reset(seed=seed)
            done, total = False, 0.0
            while not done:
                _, r, done, _, info = env.step(rule_calibrated(env, "core"))
                total += r
            s = info["stats"]
            out.append(dict(key=f"{cond}/{seed}", reward=total, kb=s["bytes"] / 1000, fixed=s["fixed"],
                            anomalies=s["anomalies"], wrong=s["wrong_ops"], requests=s["requests"], images=s["images"]))
    return name, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed0", type=int, default=120_000)
    args = p.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    with Pool(len(CANDIDATES)) as pool:
        results = dict(pool.map(run, [(n, c, seeds) for n, c in CANDIDATES.items()]))
    base = {e["key"]: e for e in results["none"]}
    rows = {}
    print(f"validation seeds {seeds[0]}..{seeds[-1]} x 5 conditions, team rule reading b_safe\n")
    print(f"{'cap':<9}{'reward':>8}{'vs none [95% CI]':>24}{'success':>9}{'KB':>7}{'imgs':>6}{'reqs':>6}{'wrong':>7}")
    for name, eps in results.items():
        d = np.array([e["reward"] - base[e["key"]]["reward"] for e in eps])
        se = d.std(ddof=1) / math.sqrt(len(d)) if name != "none" else 0.0
        tot = lambda k: sum(e[k] for e in eps)
        rows[name] = dict(caps=CANDIDATES[name], reward=float(np.mean([e["reward"] for e in eps])), diff=float(d.mean()),
                          ci=[float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                          success=tot("fixed") / max(tot("anomalies"), 1), kb=tot("kb") / len(eps),
                          images=tot("images") / len(eps), requests=tot("requests") / len(eps), wrong=tot("wrong") / len(eps))
        r = rows[name]
        print(f"{name:<9}{r['reward']:>8.2f}{r['diff']:>+9.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]{r['success']:>9.3f}"
              f"{r['kb']:>7.1f}{r['images']:>6.2f}{r['requests']:>6.2f}{r['wrong']:>7.3f}")
    best = max(rows, key=lambda n: rows[n]["reward"])
    from knowledge_estimation.integration.sim2d import POLICY as out
    out.write_text(json.dumps(dict(caps=CANDIDATES[best], chosen=best, rule="max mean team reward on validation",
                                   seeds=[seeds[0], seeds[-1]], candidates=rows), indent=1), encoding="utf-8")
    print(f"\nchosen: {best} -> {out.name}")


if __name__ == "__main__":
    main()
