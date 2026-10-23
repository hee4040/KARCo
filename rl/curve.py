"""Did training converge? Plots the training log and evaluates every checkpoint on the same seeds.

python curve.py sim2d_proposed_a0p1 sim2d_proposed_a0p1_s1 sim2d_proposed_a0p1_s2   # several seeds overlay
python curve.py sim2d_unc_a0p1 --episodes 50
Converged = reward curve flat, entropy stops falling, and checkpoint metrics stop changing (and seeds agree).
"""
import argparse
import csv
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from env import CommEnv, fix_rate, run

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+", help="model names under models/ (training must have run with logging)")
p.add_argument("--episodes", type=int, default=30, help="eval episodes per checkpoint")
p.add_argument("--no-show", action="store_true", help="only save the png")
args = p.parse_args()
if args.no_show:
    matplotlib.use("Agg")
models = Path(__file__).parent / "models"
if any(n.startswith("sim2d_") for n in args.names):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
    from env2d import Sim2DEnv

KEYS = [("rollout/ep_rew_mean", "episode reward"), ("train/entropy_loss", "entropy loss (-entropy)"),
        ("train/approx_kl", "approx KL per update")]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for name in args.names:
    rows = list(csv.DictReader(open(models / "logs" / name / "progress.csv")))
    step = [int(r["time/total_timesteps"]) for r in rows]
    for ax, (k, title) in zip(axes[0], KEYS):
        ax.plot(step, [float(r[k]) if r.get(k) else np.nan for r in rows], label=name)
        ax.set_title(title)

    # every checkpoint (+ final model) on the same eval seeds
    belief = "_unc_" not in name
    env = (Sim2DEnv if name.startswith("sim2d_") else CommEnv)(belief=belief, alpha=float(name.split("_a")[1].split("_")[0].replace("p", ".")))
    ckpts = sorted((models / "ckpt" / name).glob("step_*_steps.zip"), key=lambda f: int(f.stem.split("_")[1]))
    pts = []
    print(f"\n{name}\n{'steps':>8}{'reward':>9}{'success':>9}{'KB/ep':>8}{'redund':>8}")
    for f in ckpts:
        m = PPO.load(f)
        R, S = zip(*(run(env, lambda o: int(m.predict(o, deterministic=True)[0]), 10_000 + s)
                     for s in range(args.episodes)))  # seeds unseen in training
        pts.append((int(f.stem.split("_")[1]), np.mean(R), fix_rate(S), sum(s["bytes"] for s in S) / 1000 / len(S),
                    sum(s["redundant"] for s in S) / max(sum(s["msgs"] for s in S), 1)))
        print(f"{pts[-1][0]:>8}{pts[-1][1]:>9.1f}{pts[-1][2]:>9.2f}{pts[-1][3]:>8.1f}{pts[-1][4]:>8.2f}")
    if pts:
        x = [q[0] for q in pts]
        for ax, i, title in zip(axes[1], (1, 3, 4), ("checkpoint eval reward", "checkpoint KB / episode", "checkpoint redundant ratio")):
            ax.plot(x, [q[i] for q in pts], "o-", label=name)
            ax.set_title(title)

for ax in axes.flat:
    ax.set_xlabel("steps")
    ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=8)
fig.tight_layout()
out = models / "logs" / f"curve_{args.names[0]}.png"
fig.savefig(out)
print("saved", out)
if not args.no_show:
    plt.show()
