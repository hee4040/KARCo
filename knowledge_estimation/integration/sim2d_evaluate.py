"""Estimator evaluation in the team 2D world, Notion section 11 (decision-time records only).

python -m knowledge_estimation.integration.sim2d_evaluate --seed0 400000 --episodes 200
python -m knowledge_estimation.integration.sim2d_evaluate --seed0 400000 --episodes 200 --b-speed 0.30   # OOD

Use seeds disjoint from parameter (100000+), score-calibration (110000+) and validation (120000+) seeds.
Decisions are made by the team's rule policy reading this estimator and by Minimal Push+Pull, so the estimate
is judged on states two different policies create.

Per item N / I / C, on the same decisions:
  Brier            mean((b - K)^2) for b_raw, b_cal, b_safe, the p_acq-table variant (table_raw, table_cal) and
                   the section 10 baselines; 95% CI by bootstrap over episodes (the independent unit), plus split
                   by condition and by A's evidence sources
  paired           Brier(b_cal) - Brier(table_cal) on the same decisions, episode bootstrap 95% CI
  reliability      b_cal bins: decisions, mean estimate, actual holding rate with a Wilson 95% interval
  high-est unheld  among decisions with b >= tau (fixed 0.8), the share where B did not hold the item; N/A if none
K is B's validated holding at the decision (World._record), never shown to A or the policy.

--b-speed changes B's driving speed for the whole run (out of distribution: parameters, score maps and the p_acq
table were all measured at the layout's 0.15 m/s). A plans with the new speed, as it commands it.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl"), str(ROOT / "rule_ppo_compare"), str(ROOT)]

TAU = 0.8  # fixed in advance (Notion section 11)
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ESTIMATES = ("raw", "cal", "safe", "table_raw", "table_cal", "cache", "cache_result", "optimistic")
OUT = ROOT / "knowledge_estimation" / "data" / "sim2d_eval_v2"


def frozen():
    """sha256 of the estimator code and every artifact it reads, recorded before and checked after a run."""
    from knowledge_estimation.integration import sim2d
    from knowledge_estimation.probabilistic import estimator
    files = [sim2d.PARAMETERS, sim2d.SCORE_MODEL, sim2d.SCORE_MODEL_TABLE, sim2d.POLICY, Path(sim2d.__file__)]
    out = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    out["core_code"] = estimator._CORE_CODE_FINGERPRINT
    return out


def collect(episodes, seed0):
    from env2d import CONDITIONS, Sim2DEnv
    from calibrated import rule_calibrated
    policies = {"rule": lambda env: rule_calibrated(env, "core"), "minimal": lambda env: 1}
    rows = []
    for cond in CONDITIONS:
        for seed in range(seed0, seed0 + episodes):
            for pname, policy in policies.items():
                env = Sim2DEnv(cond)
                env.reset(seed=seed)
                done = False
                while not done:
                    _, _, done, _, info = env.step(policy(env))
                kn, table = env.world.knows["core"], env.world.knows["core_table"]
                for rec, t, d in zip(kn.decisions, table.decisions, info["stats"]["dec"]):
                    for j, kind in enumerate("NIC"):
                        r = rec[kind]
                        row = dict(condition=cond, policy=pname, episode=f"{cond}/{pname}/{seed}", kind=kind,
                                   K=int(d["K"][j]), raw=r["raw"], cal=r["cal"], safe=r["safe"], reason=r["reason"],
                                   table_raw=t[kind]["raw"], table_cal=t[kind]["cal"],
                                   sources="+".join(sorted(set(r["sources"]))) or "none")
                        for m in ("cache", "cache_result", "optimistic"):
                            row[m] = float(d["est"][m][j])
                        rows.append(row)
        print(f"collected {cond}", flush=True)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def brier_ci(rows, est, rng, reps=1000):
    """Brier with an episode-level bootstrap interval; None estimates are excluded and counted."""
    by_ep = defaultdict(lambda: [0.0, 0])
    missing = 0
    for r in rows:
        if r[est] is None:
            missing += 1
            continue
        e = by_ep[r["episode"]]
        e[0] += (r[est] - r["K"]) ** 2
        e[1] += 1
    sums = np.array([v for v in by_ep.values()], float)
    if not len(sums):
        return float("nan"), (float("nan"), float("nan")), missing
    point = sums[:, 0].sum() / sums[:, 1].sum()
    idx = rng.integers(0, len(sums), (reps, len(sums)))
    boot = sums[idx, 0].sum(1) / sums[idx, 1].sum(1)
    return point, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), missing


def paired_ci(rows, est, ref, rng, reps=1000):
    """Brier(est) - Brier(ref) on decisions where both are known; episode bootstrap 95% CI."""
    by_ep = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        if r[est] is None or r[ref] is None:
            continue
        e = by_ep[r["episode"]]
        e[0] += (r[est] - r["K"]) ** 2
        e[1] += (r[ref] - r["K"]) ** 2
        e[2] += 1
    sums = np.array(list(by_ep.values()), float)
    diff = lambda s: (s[..., 0].sum(-1) - s[..., 1].sum(-1)) / s[..., 2].sum(-1)
    boot = diff(sums[rng.integers(0, len(sums), (reps, len(sums)))])
    return float(diff(sums)), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), int(sums[:, 2].sum())


def high_unheld(rows, est):
    hi = [r for r in rows if r[est] is not None and r[est] >= TAU]
    return (sum(r["K"] == 0 for r in hi) / len(hi), len(hi)) if hi else (None, 0)


def report(rows):
    rng = np.random.default_rng(0)
    summary = dict(tau=TAU, decisions=len(rows) // 3, items={})
    for kind in "NIC":
        R = [r for r in rows if r["kind"] == kind]
        print(f"\n=== {kind}  ({len(R)} decisions, actual holding rate {np.mean([r['K'] for r in R]):.3f}) ===")
        print(f"{'estimate':<14}{'Brier':>8}{'95% CI':>18}{'high-est unheld':>18}{'(n >= tau)':>12}{'None':>6}")
        item = {}
        for est in ESTIMATES:
            b, ci, miss = brier_ci(R, est, rng)
            hu, n_hi = high_unheld(R, est)
            item[est] = dict(brier=b, ci=ci, high_unheld=hu, n_high=n_hi, missing=miss)
            hu_s = "N/A" if hu is None else f"{hu:.3f}"
            print(f"{est:<14}{b:>8.4f}   [{ci[0]:.4f}, {ci[1]:.4f}]{hu_s:>18}{n_hi:>12}{miss:>6}")
        d, dci, n = paired_ci(R, "cal", "table_cal", rng)
        item["paired_cal_minus_table_cal"] = dict(diff=d, ci=dci, n=n)
        print(f"  paired Brier b_cal - table_cal  {d:+.4f}  [{dci[0]:+.4f}, {dci[1]:+.4f}]  (n={n})")
        print(f"  reliability of b_cal  {'bin':<12}{'n':>6}{'mean b':>8}{'actual':>8}{'Wilson 95%':>18}")
        rel = []
        for lo, hi in zip(BINS, BINS[1:]):
            inb = [r for r in R if r["cal"] is not None and (lo <= r["cal"] < hi or (hi == 1.0 and r["cal"] == 1.0))]
            if not inb:
                continue
            k = sum(r["K"] for r in inb)
            w = wilson(k, len(inb))
            rel.append(dict(bin=[lo, hi], n=len(inb), mean=float(np.mean([r["cal"] for r in inb])), actual=k / len(inb), ci=w))
            print(f"  {'':<22}[{lo:.1f},{hi:.1f}){len(inb):>8}{rel[-1]['mean']:>8.3f}{k / len(inb):>8.3f}   [{w[0]:.3f}, {w[1]:.3f}]")
        by_cond = {}
        for c in dict.fromkeys(r["condition"] for r in R):
            by_cond[c] = {est: float(np.mean([(r[est] - r["K"]) ** 2 for r in R if r["condition"] == c and r[est] is not None]))
                          for est in ("cal", "table_cal", "cache")}
        print("  Brier by condition (cal / table_cal / Cache)  " + "  ".join(
            f"{c}: {v['cal']:.3f} / {v['table_cal']:.3f} / {v['cache']:.3f}" for c, v in by_cond.items()))
        item["by_condition"] = by_cond
        by_src = defaultdict(list)
        for r in R:
            if r["cal"] is not None:
                by_src[r["sources"]].append((r["cal"] - r["K"]) ** 2)
        top = sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:6]
        print("  Brier(b_cal) by A's evidence  " + "  ".join(f"{s}: {np.mean(v):.3f} (n={len(v)})" for s, v in top))
        item["reliability"] = rel
        item["by_source"] = {s: dict(n=len(v), brier=float(np.mean(v))) for s, v in by_src.items()}
        summary["items"][kind] = item
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed0", type=int, required=True,
                   help="first seed; use fresh seeds for a final evaluation, 120000+ (validation) while choosing")
    p.add_argument("--b-speed", type=float, default=None, help="B's speed in m/s for an out-of-distribution run")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    import world as W
    if args.b_speed is not None:
        W.LAYOUT["b_speed"] = args.b_speed
        W.routes.cache_clear()
        W.visibility_table.cache_clear()
        print(f"OOD: B drives at {args.b_speed} m/s (measured at 0.15 m/s)")
    before = frozen()
    rows = collect(args.episodes, args.seed0)
    if frozen() != before:
        raise RuntimeError("estimator code or artifacts changed during the evaluation")
    summary = report(rows)
    summary.update(seeds=[args.seed0, args.seed0 + args.episodes - 1], b_speed=W.LAYOUT["b_speed"], frozen=before)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
