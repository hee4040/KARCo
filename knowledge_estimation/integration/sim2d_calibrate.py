"""Measure the core estimator's parameters in the team 2D world (sim/), on calibration seeds only.

python -m knowledge_estimation.integration.sim2d_calibrate --episodes 300            # 1. sources (seeds 100000+)
python -m knowledge_estimation.integration.sim2d_calibrate --scores --episodes 200   # 2. b_cal (seeds 110000+)

Step 2 freezes the step 1 parameters, replays decisions with them, and fits one PAV map per item on the
decision-time b_raw against B's truth at that moment (Notion sections 5, 11, 12). It fits the perspective
estimator ("core") and the p_acq-table variant ("core_table") on the same decisions. Seed split: sources
100000+, scores 110000+, validation 120000+, final evaluations on fresh seeds (see SIM2D_RESULTS.md).

Every probability is successes / trials from the world's own log, never a chosen value, as ParameterStore
requires. Conditions are drawn per episode as in PPO training. The log is written to knowledge_estimation/data/
(git-ignored); the artifact keeps its sha256.

  p_recv   (channel)              A -> B message delivered                           per send attempt
  p_ack    (channel)              B -> A reply delivered (ACK, request)              per reply
  F_recv   (channel)              delivery delay of delivered messages
  p_frame  (in_view)              B recognises a spot in one camera frame while the  per frame in view
                                  spot is in its view (sensor rate; the perspective model's only pass input)
  p_acq    (kind, spot, route)    B stored N / the right look I of a spot it passed  per trip and spot on A's route
                                  map, only spots A did not know to be abnormal (table variant, comparison only)
  q_learn  (I, verified)          B holds the right look after A verified a repair   per verified repair
  q_learn  (I, pass)              B holds the right look of a spot it recognised     per trip and spot recognised
                                  while passing (verified repairs over-represent right looks: a wrong look
                                  leads to the wrong object, which A does not verify)
  r_crop   (I, content)           a received crop leaves B with the right look        per received crop
  r_current(C, content)           a received C still matches the object's position   per received C
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl"), str(ROOT)]

from knowledge_estimation.integration.sim2d import (  # noqa: E402
    CONTENT_CONDITION, CUR, IMG, LEARNING_CONDITION, PARAMETERS, PASS_CONDITION, SCORE_MODEL, SCORE_MODEL_TABLE,
    channel_condition)

LOG = ROOT / "knowledge_estimation" / "data" / "sim2d_calibration_v2" / "log.csv"
FRAME_CONDITION = "in_view"  # perspective.PerspectiveAcquisition reads p_frame under this condition
SCORE_OUTPUTS = {"core": SCORE_MODEL, "core_table": SCORE_MODEL_TABLE}


def collect(episodes, seed0):
    import world as W
    from env2d import FULL, Sim2DEnv

    rows = []
    orig_post, orig_reply, orig_verified, orig_learn = W.World._post, W.World._reply, W.World._verified, W.World._learn

    def post(self, a, kind, payload, n):
        before = len(self.inbox)
        orig_post(self, a, kind, payload, n)
        rows.append(dict(source="p_recv", context=channel_condition(self.loss), success=int(len(self.inbox) > before),
                         duration=self._delay(n)))

    def reply(self, kind="ack"):
        ok = orig_reply(self, kind)
        if kind == "ack":
            rows.append(dict(source="p_ack", context=channel_condition(self.loss), success=int(ok), duration=""))
        return ok

    def verified(self, a):
        k, counted = a.spot, a.detected
        orig_verified(self, a)
        if counted:
            rows.append(dict(source="q_learn", context=LEARNING_CONDITION, success=int(self.b_obj[k] and not self.b_wrong[k]),
                             duration=""))

    def learn(self, a, items):
        orig_learn(self, a, items)
        if IMG in items:
            rows.append(dict(source="r_crop", context=CONTENT_CONDITION,
                             success=int(self.b_obj[a.spot] and not self.b_wrong[a.spot]), duration=""))
        if CUR in items:
            rows.append(dict(source="r_current", context=CONTENT_CONDITION, success=int(a.cur_ver == a.moved), duration=""))

    W.World._post, W.World._reply, W.World._verified, W.World._learn = post, reply, verified, learn
    try:
        for seed in range(seed0, seed0 + episodes):
            env = Sim2DEnv()  # condition drawn per episode, as in training
            env.reset(seed=seed)
            done = False
            while not done:
                _, _, done, _, info = env.step(FULL)
            s = info["stats"]
            trials, hits = s["frames"]
            rows += [dict(source="p_frame", context=FRAME_CONDITION, success=int(i < hits), duration="")
                     for i in range(trials)]
            for (k, truth), open_ in zip(s["trips"], s["trip_open"]):
                for j in W.visibility_table()[k]:  # A's route map (no occluder)
                    if j in open_:
                        continue
                    seen, right = truth[j]
                    rows.append(dict(source="p_acq", context=f"N|S{j}|to{k}", success=int(seen), duration=""))
                    rows.append(dict(source="p_acq", context=f"I|S{j}|to{k}", success=int(right), duration=""))
                    if seen:  # B's look of a spot it recognised while passing, whatever the route
                        rows.append(dict(source="q_learn", context=PASS_CONDITION, success=int(right), duration=""))
    finally:
        W.World._post, W.World._reply, W.World._verified, W.World._learn = orig_post, orig_reply, orig_verified, orig_learn
    return rows


def build_artifact(rows, digest, ack_timeout):
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"], r["context"]].append(r)
    entries = []

    def ctx(kind="", site="", route="", condition=""):
        return dict(receiver_id="B", kind=kind, site_id=site, route_id=route, condition_id=condition)

    def rate(name, context, rs, **extra):
        s, n = sum(r["success"] for r in rs), len(rs)
        entries.append(dict(name=name, parameter_id=f"sim2d:{name}:{':'.join(context.values())}", context=context,
                            evidence_ids=["sim2d-calibration-log"], input_sha256=digest, trials=n, successes=s,
                            value=s / n, **extra))

    for (source, context), rs in sorted(groups.items()):
        if source == "p_recv":
            delays = [float(r["duration"]) for r in rs if r["success"]]
            deadline = ack_timeout + max(delays)  # A waits the transfer time plus the ACK timeout
            rate("p_recv", ctx(condition=context), rs, deadline_seconds=deadline)
            entries.append(dict(name="F_recv", parameter_id=f"sim2d:F_recv:{context}", context=ctx(condition=context),
                                evidence_ids=["sim2d-calibration-log"], input_sha256=digest, trials=len(delays),
                                samples_seconds=delays, observation_horizon_seconds=deadline))
        elif source == "p_ack":
            rate("p_ack", ctx(condition=context), rs)
        elif source == "p_frame":
            rate("p_frame", ctx(condition=context), rs)
        elif source == "p_acq":
            kind, site, route = context.split("|")
            rate("p_acq", ctx(kind, site, route, PASS_CONDITION), rs)
        elif source == "q_learn":
            rate("q_learn", ctx("I", condition=context), rs)
        elif source == "r_crop":
            rate("r_crop", ctx("I", condition=context), rs)
        elif source == "r_current":
            rate("r_current", ctx("C", condition=context), rs)
    return dict(schema_version=1, origin="simulation", parameters=entries,
                note="measured in the team 2D world (sim/) on calibration seeds; not robot measurements")


SCORE_POLICIES = ("rule", "minimal", "fixed_state", "full")


def collect_decisions(episodes, seed0):
    """Decision-time records under the frozen parameters, per estimator variant: [(kind, b_raw, K)]."""
    import world as W
    W.SCORE_MODEL = W.SCORE_MODEL_TABLE = None  # collect raw scores; no earlier map is used while collecting
    from env2d import Sim2DEnv
    sys.path.insert(0, str(ROOT / "rule_ppo_compare"))
    from calibrated import rule_calibrated
    # decisions come from the team's policies, so the fit sees the states those policies create
    policies = {"rule": lambda env: rule_calibrated(env, "core"), "minimal": lambda env: 1,
                "fixed_state": lambda env: 1 + 1 + 2 + 4, "full": lambda env: 16}
    rows = {mode: [] for mode in SCORE_OUTPUTS}
    fingerprints = {mode: set() for mode in SCORE_OUTPUTS}
    decisions = 0
    for seed in range(seed0, seed0 + episodes):
        for name in SCORE_POLICIES:
            env = Sim2DEnv()  # condition drawn per episode, as in training
            env.reset(seed=seed)
            done = False
            while not done:
                _, _, done, _, info = env.step(policies[name](env))
            dec = info["stats"]["dec"]
            decisions += len(dec)
            for mode in SCORE_OUTPUTS:
                kn = env.world.knows[mode]
                fingerprints[mode].add(kn.core.fingerprint)
                assert len(dec) == len(kn.decisions), "decision records out of step"
                for rec, d in zip(kn.decisions, dec):  # both written once per World.decide, before sending
                    for kind, truth in zip("NIC", d["K"]):
                        r = rec[kind]
                        if r["raw"] is not None and r["reason"] not in ("hard_invalid_zero", "no_item"):
                            rows[mode].append((kind, r["raw"], int(truth)))
    for mode, prints in fingerprints.items():
        if len(prints) != 1:
            raise ValueError(f"score calibration of {mode} needs one frozen raw estimator/parameter set")
    return {mode: (rows[mode], fingerprints[mode].pop()) for mode in SCORE_OUTPUTS}, decisions


def build_score_model(rows, fingerprint, decisions, seed0, episodes):
    """One pooled PAV map per item in the projection.validate_model format."""
    from knowledge_estimation.probabilistic.projection import model_id, monotone_map, validate_model
    pairs = {k: [(raw, y) for kind, raw, y in rows if kind == k] for k in "NIC"}
    model = dict(schema_version=1, origin="simulation", estimator_fingerprint=fingerprint, receiver_ids=["B"],
                 maps={k: monotone_map(v) for k, v in pairs.items()}, fit_split="calibration",
                 method="isotonic_PAV_no_prior_no_extrapolation", decisions=decisions,
                 pairs={k: len(v) for k, v in pairs.items()}, seeds=[seed0, seed0 + episodes - 1],
                 policies=list(SCORE_POLICIES), note="decision-time records in the team 2D world; not robot data")
    model["model_id"] = model_id(model)
    validate_model(model)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed0", type=int, default=None, help="default 100000 (sources) / 110000 (--scores)")
    p.add_argument("--scores", action="store_true", help="fit b_cal on decision-time records instead of sources")
    p.add_argument("--output", type=Path, default=None, help="sources only (default: sim2d_parameters_v2.json)")
    args = p.parse_args()
    if args.scores:
        seed0 = 110_000 if args.seed0 is None else args.seed0
        fitted, decisions = collect_decisions(args.episodes, seed0)
        for mode, (rows, fingerprint) in fitted.items():
            model = build_score_model(rows, fingerprint, decisions, seed0, args.episodes)
            SCORE_OUTPUTS[mode].write_text(json.dumps(model, indent=1), encoding="utf-8")
            print(f"{mode:<11} {decisions} decisions, pairs {model['pairs']} -> {SCORE_OUTPUTS[mode].name}")
        return
    args.seed0 = 100_000 if args.seed0 is None else args.seed0
    args.output = args.output or PARAMETERS
    import world as W
    rows = collect(args.episodes, args.seed0)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["source", "context", "success", "duration"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_bytes(data)
    artifact = build_artifact(rows, hashlib.sha256(data).hexdigest(), W.LAYOUT["ack_timeout"])
    artifact.update(episodes=args.episodes, seeds=[args.seed0, args.seed0 + args.episodes - 1], log=str(LOG.relative_to(ROOT)))
    args.output.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    from knowledge_estimation.parameters import ParameterStore
    ParameterStore.load(args.output, allow_simulation=True)  # validates every entry
    print(f"{len(rows)} log rows -> {len(artifact['parameters'])} parameters -> {args.output.name}")
    for e in artifact["parameters"]:
        if "value" in e:
            print(f"  {e['name']:<9} {'/'.join(v for v in e['context'].values() if v and v != 'B'):<22} "
                  f"{e['successes']:>5}/{e['trials']:<5} = {e['value']:.3f}" + (f"  deadline {e['deadline_seconds']:.2f}s"
                                                                              if 'deadline_seconds' in e else ""))


if __name__ == "__main__":
    main()
