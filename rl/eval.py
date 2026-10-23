"""Compare communication methods on each scenario condition.

python eval.py [--alpha 0.1] [--episodes 200] [--env sim2d]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from env import CONDITIONS, FULL, CommEnv, run

p = argparse.ArgumentParser()
p.add_argument("--alpha", type=float, default=0.1)
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--env", default="event", choices=["event", "sim2d"], help="event = rl/env.py, sim2d = sim/")
args = p.parse_args()
if args.env == "sim2d":  # 2D line-tracking world in ../sim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
    from env2d import CONDITIONS, FULL, Sim2DEnv as CommEnv  # noqa: F811
prefix = "sim2d_" if args.env == "sim2d" else ""
models = Path(__file__).parent / "models"


def ppo(name):
    m = PPO.load(models / f"{prefix}{name}_a{str(args.alpha).replace('.', 'p')}")
    return lambda o: int(m.predict(o, deterministic=True)[0])


# action = 1 + item bits (id=1, cur=2, norm=4, img=8)
METHODS = {  # name: (policy, uses belief env)
    "Full Sharing":  (lambda o: FULL, True),
    "Fixed Minimal": (lambda o: 1, True),            # base only, B must ask for the rest
    "Fixed Text":    (lambda o: 1 + 1 + 4, True),    # + id + normal pose
    "Fixed State":   (lambda o: 1 + 1 + 2 + 4, True),  # + id + current pose + normal pose
    "Uncert. Only":  (ppo("unc"), False),
    "Proposed":      (ppo("proposed"), True),
}
if args.env == "sim2d":  # rule-based expected-cost policy (heewoo/rule-based branch), for the ablation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rule_based_comm.policy_adapter import ACTIONS, RulePolicyAdapter
    BIT = {"E": 1, "N": 4, "C": 2, "I": 8}  # E carries the target id -> our base + ID

    def rule(belief):
        adapter = RulePolicyAdapter()

        def policy(o):  # reads the module-level `env` of the loop below (the rule needs the world, not the vector)
            act, _ = adapter.predict(env.rule_input("core" if belief else None))
            return 0 if act is None else 1 + sum(BIT[i] for i in ACTIONS[act])  # NO_COMM / REOBSERVE -> send nothing
        return policy

    METHODS = {"Rule (no belief)": (rule(False), True), "Rule": (rule(True), True), **METHODS}

for cond in CONDITIONS:
    print(f"\n== {cond} ==")
    print(f"{'method':<17}{'success':>8}{'time(s)':>9}{'KB/ep':>8}{'msgs':>7}{'imgs':>6}{'reqs':>6}{'redund':>8}{'wrong':>7}")
    for name, (policy, belief) in METHODS.items():
        env = CommEnv(cond, belief=belief, alpha=args.alpha)
        S = [run(env, policy, seed)[1] for seed in range(args.episodes)]  # same seeds for every method
        tot = lambda k: sum(s[k] for s in S)
        times = [t for s in S for t in s["times"]]
        print(f"{name:<17}{tot('fixed') / max(tot('anomalies'), 1):>8.2f}{np.mean(times) if times else 0:>9.1f}"
              f"{tot('bytes') / 1000 / len(S):>8.1f}{tot('msgs') / len(S):>7.1f}{tot('images') / len(S):>6.1f}"
              f"{tot('requests') / len(S):>6.1f}{tot('redundant') / max(tot('msgs'), 1):>8.2f}"
              f"{sum(s.get('wrong_ops', 0) for s in S) / len(S):>7.2f}")
    err = [e for s in S for e in s["know_err"]]
    print(f"A's estimate of 'B knows the normal layout': mean abs error {np.mean(err):.2f}")
