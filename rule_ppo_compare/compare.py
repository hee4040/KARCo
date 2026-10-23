"""Rule-based vs PPO under the scenario doc's conditions, methods and metrics (same simulator, events, costs).

python compare.py                  # 200 episodes per condition -> tables + results.csv + episodes.csv
python compare.py --episodes 20    # quick look
PPO models come from: python ../rl/train.py --env sim2d [--no-belief], and imitate.py
"""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from calibrated import rule_calibrated, rule_default
from env2d import CONDITIONS, FULL, Sim2DEnv
from stable_baselines3 import PPO

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--ref", default="Rule (proposed)", help="method the paired differences are taken against")
p.add_argument("--out", default=None, help="folder for results.csv / episodes.csv (default: this folder)")
p.add_argument("--seed0", type=int, default=0, help="first seed (default 0 = seeds 0..episodes-1)")
p.add_argument("--only", default=None, help="comma-separated method names to run (default: all)")
args = p.parse_args()
here = Path(__file__).resolve().parent
out_dir = Path(args.out) if args.out else here
out_dir.mkdir(parents=True, exist_ok=True)
models = here.parent / "rl" / "models"
TAU = 0.8  # 'high estimate' threshold, fixed in advance (scenario doc)


def ppo(name):
    path = models / f"sim2d_{name}_a0p1.zip"
    if not path.exists():
        return None
    m = PPO.load(path)
    return lambda env: int(m.predict(env.obs(), deterministic=True)[0])


def gated(act):  # the common event gate: nothing while the same event is being processed
    return lambda env: 0 if env.rule_input(None)["already_processing"] else act


# name: (policy(env) -> action, env exposes belief features, estimator whose quality is scored or None)
METHODS = {
    "Periodic Full":         (lambda env: FULL, True, None),          # weak baseline: everything, every time
    "Event-gated Full":      (gated(FULL), True, None),
    "Minimal Push+Pull":     (gated(1), True, None),
    "Fixed State":           (lambda env: 1 + 1 + 2 + 4, True, None),
    "Scene-only rule":       (lambda env: rule_calibrated(env, None), True, None),
    "Rule (Confirmed Cache)": (lambda env: rule_calibrated(env, "cache"), True, "cache"),
    "Rule (Cache+Result)":   (lambda env: rule_calibrated(env, "cache_result"), True, "cache_result"),
    "Rule (optimistic)":     (lambda env: rule_calibrated(env, "optimistic"), True, "optimistic"),
    "Rule (proposed)":       (lambda env: rule_calibrated(env, "core"), True, "core"),  # knowledge_estimation
    "Rule-default":          (lambda env: rule_default(env, "core"), True, "core"),
    "PPO (no B)":            (ppo("unc"), False, None),
    "PPO":                   (ppo("proposed"), True, "core"),
    "BC (rule copy)":        (ppo("bc"), True, "core"),
    "PPO from rule":         (ppo("bcppo"), True, "core"),
}
METHODS = {k: v for k, v in METHODS.items() if v[0] is not None}
if args.only:
    METHODS = {k: METHODS[k] for k in args.only.split(",")}


def decision_metrics(S, mode):
    """Scenario doc metrics over every decision A made (records written by World._record)."""
    D = [d for s in S for d in s["dec"] if d["act"]]
    n = max(len(D), 1)
    sent = lambda d, i: i in d["sent"]  # item ids: 1 cur, 2 norm, 3 img
    out = dict(
        crop_omit=np.mean([not sent(d, 3) for d in D if d["need_i"]] or [0]),
        omit_fail_ask=sum((not sent(d, 3) and d["need_i"] and d["without_img"] in ("ask", "not_found"))
                          or (not sent(d, 2) and d["need_n"]) for d in D) / n,
        omit_fail_wrong=sum(not sent(d, 3) and d["need_i"] and d["without_img"] == "wrong" for d in D) / n,
        unneeded_kb=sum(60 * (sent(d, 2) and not d["need_n"]) + 30_000 * (sent(d, 3) and not d["need_i"])
                        for d in D) / 1000 / max(len(S), 1))
    if mode:
        e = np.array([d["est"][mode] for d in D] or [(0, 0, 0)], float)  # (N, I, C)
        K = np.array([d["K"] for d in D] or [(0, 0, 0)], float)
        out.update({f"brier_{x}": float(np.mean((e[:, j] - K[:, j]) ** 2)) for j, x in enumerate("NIC")})
        hi = e[:, :2] >= TAU
        out["high_unheld"] = float(((K[:, :2] == 0) & hi).sum() / hi.sum()) if hi.sum() else float("nan")
    return out


rows, eps = [], []
for cond in CONDITIONS:
    print(f"\n== {cond} ==")
    print(f"{'method':<24}{'reward':>7}{'succ':>6}{'time':>6}{'late':>6}{'KB':>7}{'retx':>6}{'pull':>6}{'v2':>6}"
          f"{'msgs':>6}{'imgs':>5}{'reqs':>5}{'dup':>5}{'wrong':>6}{'crop-':>6}{'f-ask':>6}{'f-wr':>6}{'unKB':>6}")
    for name, (policy, belief, mode) in METHODS.items():
        env = Sim2DEnv(cond, belief=belief)
        R, S = [], []
        for seed in range(args.seed0, args.seed0 + args.episodes):  # same seeds -> same events for every method
            env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                _, r, done, _, info = env.step(policy(env))
                total += r
            R.append(total)
            S.append(info["stats"])
            st = info["stats"]
            eps.append(dict(condition=cond, method=name, seed=seed, reward=total, kb=st["bytes"] / 1000,
                            fixed=st["fixed"], anomalies=st["anomalies"], wrong=st["wrong_ops"]))
        tot = lambda k: sum(s[k] for s in S)
        kb = lambda kind: sum(s["bytes_by"][kind] for s in S) / 1000 / len(S)
        times = [t for s in S for t in s["times"]]
        row = dict(condition=cond, method=name, reward=np.mean(R), success=tot("fixed") / max(tot("anomalies"), 1),
                   time_s=np.mean(times) if times else 0, deadline_miss=tot("deadline_miss") / max(tot("anomalies"), 1),
                   kb_per_ep=tot("bytes") / 1000 / len(S), kb_task=kb("task"), kb_retx=kb("retx"), kb_info=kb("info"),
                   kb_v2=kb("v2"), kb_pose=kb("pose"), kb_control=kb("ack") + kb("result") + kb("status"),
                   msgs=tot("msgs") / len(S), imgs=tot("images") / len(S), reqs=tot("requests") / len(S),
                   pulls=tot("pulls") / len(S), redundant=tot("redundant") / max(tot("msgs"), 1),
                   wrong=tot("wrong_ops") / len(S),
                   verify_delay=np.mean([v for s in S for v in s["verify_delay"]] or [0]),
                   **decision_metrics(S, mode))
        rows.append(row)
        print(f"{name:<24}{row['reward']:>7.1f}{row['success']:>6.2f}{row['time_s']:>6.1f}{row['deadline_miss']:>6.2f}"
              f"{row['kb_per_ep']:>7.1f}{row['kb_retx']:>6.2f}{row['pulls']:>6.2f}{row['kb_v2']:>6.1f}{row['msgs']:>6.1f}"
              f"{row['imgs']:>5.1f}{row['reqs']:>5.1f}{row['redundant']:>5.2f}{row['wrong']:>6.2f}{row['crop_omit']:>6.2f}"
              f"{row['omit_fail_ask']:>6.2f}{row['omit_fail_wrong']:>6.2f}{row['unneeded_kb']:>6.1f}")
    print(f"\n  estimator quality ({cond}, decision-time, tau {TAU})")
    print(f"  {'method':<24}{'Brier N':>8}{'Brier I':>8}{'Brier C':>8}{'high-est unheld':>16}")
    for row in rows:
        if row["condition"] == cond and "brier_N" in row:
            print(f"  {row['method']:<24}{row['brier_N']:>8.3f}{row['brier_I']:>8.3f}{row['brier_C']:>8.3f}"
                  f"{row['high_unheld']:>16.2f}")

# paired differences against the reference method (same seeds = same events)
print(f"\npaired difference vs {args.ref} (mean, 95% CI, two-sided p by normal approx.)")
for cond in CONDITIONS:
    ref = {e["seed"]: e for e in eps if e["condition"] == cond and e["method"] == args.ref}
    for name in METHODS:
        if name == args.ref or not ref:
            continue
        for key in ("reward", "kb"):
            d = np.array([e[key] - ref[e["seed"]][key] for e in eps if e["condition"] == cond and e["method"] == name])
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            pval = math.erfc(abs(d.mean() / se) / math.sqrt(2)) if se > 0 else float("nan")
            if name in ("PPO from rule", "PPO", "BC (rule copy)", "Rule (Confirmed Cache)", "Rule (optimistic)"):
                print(f"  {cond:<9}{name:<24}{key:<7}{d.mean():>+8.2f}  [{d.mean() - 1.96 * se:+.2f}, "
                      f"{d.mean() + 1.96 * se:+.2f}]  p={pval:.3f}")

for fname, data in (("results.csv", rows), ("episodes.csv", eps)):
    with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))  # union: some rows lack Brier
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in data)
print("\nsaved results.csv, episodes.csv")
