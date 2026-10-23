"""Train the modality-selection policy (PPO) on the simulator.

python train.py                # Proposed (uncertainty + receiver belief)
python train.py --no-belief    # Uncertainty Only baseline
python train.py --alpha 0.5    # heavier byte cost -> another point on the trade-off curve
python train.py --env sim2d    # train in the 2D line-tracking world (sim/)
Logs -> models/logs/<name>/progress.csv, checkpoints every 50k steps -> models/ckpt/<name>/ (check with curve.py)
"""
import argparse
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from env import CommEnv

p = argparse.ArgumentParser()
p.add_argument("--no-belief", action="store_true")
p.add_argument("--alpha", type=float, default=0.1, help="reward cost per KB sent")
p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--env", default="event", choices=["event", "sim2d"], help="event = rl/env.py, sim2d = sim/")
args = p.parse_args()
if args.env == "sim2d":  # 2D line-tracking world in ../sim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
    from env2d import Sim2DEnv as CommEnv  # noqa: F811
prefix = "sim2d_" if args.env == "sim2d" else ""
name = f"{prefix}{'unc' if args.no_belief else 'proposed'}_a{str(args.alpha).replace('.', 'p')}"
name += f"_s{args.seed}" if args.seed else ""  # extra seeds for the convergence check; seed 0 keeps the eval.py name
models = Path(__file__).parent / "models"

# PPO over DQN: rewards of several anomalies interleave in one episode and DQN underfit (lost to a hand rule)
env = make_vec_env(lambda: CommEnv(belief=not args.no_belief, alpha=args.alpha), n_envs=8, seed=args.seed)
model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]), gamma=0.95, n_steps=256,
            batch_size=256, ent_coef=0.01, seed=args.seed, verbose=0)
model.set_logger(configure(str(models / "logs" / name), ["csv"]))  # one row per PPO update (2048 steps)
model.learn(args.steps, callback=CheckpointCallback(50_000 // env.num_envs, str(models / "ckpt" / name), "step"))
out = models / name
model.save(out)
print("saved", f"{out}.zip")
