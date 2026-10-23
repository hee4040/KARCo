"""Does the estimator follow the Notion design inside the team 2D world? Trace + branch coverage + rule checks.

python -m knowledge_estimation.integration.sim2d_trace --trace base 3      # one episode, world log + core updates
python -m knowledge_estimation.integration.sim2d_trace --episodes 40       # coverage and rule checks, all conditions

Rule checks run on the estimator's own records after every episode (A-side information only):
  send alone never stores           evidence created by a send has p_stored 0 (F_recv at t=0) or None, never 1
  pass waits for normal (section 8) every pass_by evidence belongs to a pass confirmed normal before and after
  RESULT ok waits for A             every verified_task evidence has a task_verified record before it
  no-ACK posterior                  every ack_timeout evidence equals d(1-a)/(1-da) of the measured channel
  unmeasured stays None             estimates that are None are counted, never turned into a number by the core
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl"), str(ROOT)]

from knowledge_estimation.probabilistic.probabilities import receipt_given_no_ack  # noqa: E402

BRANCHES = {  # core history action -> Notion branch
    "send": "6 send (attempt per retransmit)", "ack": "6 ACK stores items", "ack_timeout": "6 no-ACK posterior",
    "needs_info": "6 needs_info / request invalidates", "missing": "2(d) wrong look invalidated",
    "diagnostic:not_found": "4(d) not_found kept as diagnostic", "pass_by": "3 pass recorded",
    "pass_confirmed": "8 pass confirmed normal", "pass_abnormal": "8 pass dropped (abnormal)",
    "result_pending": "1(a) RESULT ok pending", "task_verified": "1(a) verified by A",
    "evidence_revoked": "2(d) verified evidence revoked", "memory_reset": "2(b) new memory epoch",
    "version": "5 new item version (C re-measured)",
}


POLICIES = {"full": 16, "minimal": 1, "fixed_state": 1 + 1 + 2 + 4}  # team action ids (base + item bits)


def run(condition, seed, trace=False, policy="full"):
    """Fixed team policies only drive the episode; every branch the estimator sees comes from the team world."""
    from env2d import Sim2DEnv
    env = Sim2DEnv(condition, trace=trace)
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, _, info = env.step(POLICIES[policy])
    return env.world, info["stats"]


def check(world):
    kn = world.knows["core"]
    core, problems = kn.core, []
    verified_at = {r.reference: r.timestamp for r in core.history if r.action == "task_verified"}
    params = core.parameters
    channel = dict(receiver_id="B", kind="", site_id="", route_id="", condition_id=kn.channel)
    posterior = receipt_given_no_ack(params.value("p_recv", channel), params.value("p_ack", channel))
    for evidences in core._storage.values():
        for ev in evidences.values():
            if ev.source == "send" and ev.p_stored not in (0.0, None):
                problems.append(f"send stored {ev.p_stored}")
            if ev.source == "pass_by":
                visit = core._passes[ev.provenance_ids[0]]
                if not (visit.normal_before and visit.normal_after is True):
                    problems.append(f"pass used before normal confirmation: {ev.evidence_id}")
            if ev.source == "verified_task" and ev.provenance_ids[0] not in verified_at:
                problems.append(f"task evidence without verification: {ev.evidence_id}")
            if ev.source == "ack_timeout" and abs(ev.p_stored - posterior) > 1e-12:
                problems.append(f"no-ACK posterior {ev.p_stored} != {posterior}")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", nargs=2, metavar=("CONDITION", "SEED"))
    p.add_argument("--policy", choices=list(POLICIES), default="minimal", help="policy driving the traced episode")
    p.add_argument("--episodes", type=int, default=40)
    args = p.parse_args()
    from env2d import CONDITIONS
    if args.trace:
        world, stats = run(args.trace[0], int(args.trace[1]), trace=True, policy=args.policy)
        rows = [(float(line[:6]), 0, line) for line in world.log]
        for r in world.knows["core"].core.history:
            if r.action in ("version",) and r.timestamp == 0:
                continue
            items = ", ".join(f"{i.key.kind.value}:{i.key.object_id}v{i.version}" for i in r.items)
            rows.append((r.timestamp, 1, f"{r.timestamp:6.1f}s  KE {r.action:<16} {r.reference:<10} {items}"))
        for _, _, line in sorted(rows, key=lambda x: (x[0], x[1])):
            print(line)
        for d in stats["dec"]:
            print("decision:", "b(N,I,C) =", tuple(round(float(v), 3) for v in d["est"]["core"]), "truth K =", d["K"])
        print("rule checks:", check(world) or "all passed")
        return
    coverage, problems, unsupported = Counter(), [], 0
    for cond in CONDITIONS:
        for seed in range(args.episodes):
            for policy in POLICIES:
                world, _ = run(cond, 50_000 + seed, policy=policy)
                kn = world.knows["core"]
                coverage.update(r.action for r in kn.core.history if not (r.action == "version" and r.timestamp == 0))
                unsupported += kn.unsupported
                problems += [f"{cond}/{seed}/{policy}: {x}" for x in check(world)]
    print(f"{args.episodes} episodes x {len(CONDITIONS)} conditions x policies {list(POLICIES)} (seeds 50000+)\n")
    print(f"{'Notion branch':<40}{'core action':<22}{'count':>7}")
    for action, label in BRANCHES.items():
        print(f"{label:<40}{action:<22}{coverage.get(action, 0):>7}")
    other = {a: c for a, c in coverage.items() if a not in BRANCHES}
    if other:
        print("other:", other)
    print(f"\nestimates read as None (unmeasured): {unsupported}")
    print("rule checks:", "all passed" if not problems else f"{len(problems)} problems, e.g. {problems[:5]}")


if __name__ == "__main__":
    main()
