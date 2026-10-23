"""Print one simulated patrol as a timeline, to see what each method does in the same episode.

python replay.py                                  # Proposed vs Full Sharing, condition normal, seed 0
python replay.py --condition clustered --seed 3
python replay.py --methods proposed               # proposed | unc | full | minimal (comma separated)
Legend: !  anomaly happens   A  Robot A decides   B  Robot B acts   -- end of patrol
"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from env import CONDITIONS, FULL, CommEnv

p = argparse.ArgumentParser()
p.add_argument("--condition", default="normal", choices=list(CONDITIONS))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--methods", default="proposed,full")
p.add_argument("--alpha", type=float, default=0.1)
args = p.parse_args()


def ppo(name):
    m = PPO.load(Path(__file__).parent / "models" / f"{name}_a{str(args.alpha).replace('.', 'p')}")
    return lambda o: int(m.predict(o, deterministic=True)[0])


for name in args.methods.split(","):
    policy, belief = {"proposed": (ppo, True), "unc": (ppo, False),
                      "full": (lambda o: FULL, True), "minimal": (lambda o: 1, True)}[name]
    if policy is ppo:
        policy = ppo(name)
    env = CommEnv(args.condition, belief=belief, alpha=args.alpha, trace=True)
    # same seed -> same start; one RNG drives everything, so events drift apart once the methods act differently
    obs, _ = env.reset(seed=args.seed)
    total, done = 0.0, False
    while not done:
        obs, r, done, _, info = env.step(policy(obs))
        total += r
    s = info["stats"]
    print(f"\n===== {name} | {args.condition} | seed {args.seed} =====")
    print("\n".join(env.log))
    print(f"-> fixed {s['fixed']}/{s['anomalies']}, {s['bytes'] / 1000:.1f} KB, {s['msgs']} msgs, "
          f"{s['images']} images, {s['requests']} info requests, {s['redundant']} redundant, reward {total:.1f}")
