"""Gymnasium wrapper around the 2D world, with the same observation (27) and action (17) spaces as rl/env.py,
so rl/train.py, rl/eval.py and rl/replay.py can use it via --env sim2d."""
import gymnasium as gym
import numpy as np
from world import ATYPES, CONDITIONS, FULL, N_ACTIONS, World
from features import BELIEF, OBS_DIM, make_obs

__all__ = ["Sim2DEnv", "CONDITIONS", "FULL"]


class Sim2DEnv(gym.Env):
    def __init__(self, condition=None, belief=True, alpha=0.1, beta=1.0, gamma=1.0, trace=False):
        self.condition, self.belief, self.trace = condition, belief, trace
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.observation_space = gym.spaces.Box(-1, 3, (OBS_DIM,), np.float32)
        # Uncertainty Only baseline: no "send nothing" option, no knowledge/history features
        self.action_space = gym.spaces.Discrete(N_ACTIONS if belief else N_ACTIONS - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        name = self.condition or list(CONDITIONS)[self.np_random.integers(len(CONDITIONS))]  # domain randomization
        self.world = World(self.np_random, name, self.alpha, self.beta, self.gamma, self.trace)
        if self.world.run_until_decision():  # whole patrol without a decision: redraw
            return self.reset()
        self.world.take_reward()
        return self.obs(), {}

    def step(self, action):
        self.world.decide(int(action) + (0 if self.belief else 1))
        done = self.world.run_until_decision()
        return self.obs(), self.world.take_reward(), done, False, {"stats": self.world.stats} if done else {}

    @property
    def log(self):
        return self.world.log

    def rule_input(self, mode="core"):
        """The pending decision in the rule-based policy's input format (rule_based_comm.schemas.Event + BeliefState).
        mode = which estimator of B's knowledge (features.MODES), or None for a scene-only rule (no estimate).
        The 'same event is being processed' gate is common to every method (scenario doc 1)."""
        w = self.world
        a = w.pending[0]
        s = w.spots[a.spot]
        kn = w.knows[mode] if mode else None
        return dict(event_id=f"{a.spot}@{a.hist.t0:.1f}", spot_id=str(a.spot), anomaly_type=ATYPES[a.atype],
                    has_similar_objects=s.n_similar > 1, similar_object_count=s.n_similar - 1,
                    object_distance_cm=s.spacing_cm, position_uncertainty_cm=a.pos_err,
                    already_processing=a.job is not None and not a.job["failed"],
                    b_N=kn.normal[a.spot] if kn else 0.0, b_I=kn.obj[a.spot] if kn else 0.0,
                    b_C=kn.cur[a.spot] if kn else 0.0)

    def obs(self):
        w = self.world
        if not w.pending:
            return np.zeros(OBS_DIM, np.float32)
        a = w.pending[0]
        s = w.spots[a.spot]
        b_near = w.b_job is not None and w.b_job[0].spot == a.spot
        o = make_obs(a.conf, a.occ, s.n_similar, a.pos_err, s.spacing_cm, float(s.identical), a.atype, a.spot,
                     a.hist, w.know, b_near, w.b_state != "idle", len(w.queue), w.bw, w.loss)
        if not self.belief:
            o[BELIEF] = 0
        return o


if __name__ == "__main__":
    import time
    from gymnasium.utils.env_checker import check_env
    check_env(Sim2DEnv(), skip_render_check=True)
    # fuzz: random actions in every condition must never crash (rare protocol paths: TASK v2 resent, POSE_UPDATE...)
    rng = np.random.default_rng(0)
    for cond in CONDITIONS:
        for seed in range(60):
            env = Sim2DEnv(cond)
            env.reset(seed=seed)
            done = False
            while not done:
                _, _, done, _, _ = env.step(int(rng.integers(N_ACTIONS)))
    t0, n = time.time(), 0
    for seed in range(20):
        env = Sim2DEnv("base")
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(FULL)
            n += 1
    s = info["stats"]
    print(f"ok | {n} decisions in {time.time() - t0:.1f}s | last episode: fixed {s['fixed']}/{s['anomalies']}, "
          f"{s['requests']} requests, {s['reminders']} reminders")
