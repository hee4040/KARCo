"""The core KnowledgeEstimator inside the team 2D world (knowledge_estimation/integration/sim2d.py).

Event semantics are checked on a tiny stand-in world (only the A-side fields the adapter reads), so they do
not depend on the simulator; the end-to-end check runs only when the team sim/ is present in this checkout.
"""
import math
from pathlib import Path
import sys
import unittest

from knowledge_estimation.integration.sim2d import CUR, IMG, NORM, CoreKnowledge, load_parameters
from knowledge_estimation.parameters import ParameterStore
from knowledge_estimation.probabilistic.probabilities import receipt_given_no_ack

ROOT = Path(__file__).resolve().parents[2]
SPOTS = [(0.6, 0.95), (0.9, 0.65), (1.2, 0.95), (1.5, 0.65)]


class StandIn:
    def __init__(self, loss=0.0):
        self.t, self.loss, self.a_epoch = 0.0, loss, 0


def store():
    return load_parameters()


@unittest.skipIf(load_parameters() is None, "sim2d parameters not measured yet")
class Adapter(unittest.TestCase):
    def setUp(self):
        self.world = StandIn()
        self.kn = CoreKnowledge(self.world, store(), SPOTS)
        self.params = store()

    def value(self, name, **ctx):
        base = dict(receiver_id="B", kind="", site_id="", route_id="", condition_id="")
        return self.params.value(name, {**base, **ctx})

    def test_parameters_are_measured_rates(self):
        art = self.params.artifact
        self.assertEqual(art["origin"], "simulation")
        for e in art["parameters"]:
            if "value" in e:
                self.assertEqual(e["value"], e["successes"] / e["trials"])
        with self.assertRaises(Exception):
            ParameterStore(art)  # simulation numbers are refused unless explicitly allowed

    def test_ack_confirms_storage_and_content_quality(self):
        kn = self.kn
        kn.pose_measured(1, 0, 0.0)
        payload = dict(items={NORM, CUR, IMG})
        kn.send(1, payload, payload["items"], 1)
        self.assertEqual(kn.normal[1], 0.0)  # sent is not stored
        self.world.t = 0.5
        kn.ack(payload, 0)
        self.assertEqual(kn.normal[1], 1.0)
        self.assertAlmostEqual(kn.obj[1], self.value("r_crop", kind="I", condition_id="content"))
        self.assertAlmostEqual(kn.cur[1], self.value("r_current", kind="C", condition_id="content"))

    def test_missing_ack_gives_the_delivery_posterior_after_the_deadline(self):
        kn = self.kn
        payload = dict(items={NORM})
        kn.send(2, payload, payload["items"], 1)
        self.world.t = kn.deadline - 0.1
        self.assertEqual(kn.normal[2], 0.0)
        self.world.t = kn.deadline + 0.1
        d = self.value("p_recv", condition_id="link_good")
        a = self.value("p_ack", condition_id="link_good")
        self.assertAlmostEqual(kn.normal[2], receipt_given_no_ack(d, a))

    def test_perspective_pass_uses_planned_frames_and_sensor_rates(self):
        from knowledge_estimation.probabilistic.perspective import Camera
        # B drives past S2 facing +y: the first 3 planned frames see it (0.5 m ahead), the last 2 are beyond 1.2 m
        plan = [(1.2, 0.45, math.pi / 2)] * 3 + [(1.2, 2.3, -math.pi / 2)] * 2
        kn = CoreKnowledge(self.world, store(), SPOTS, plans={"to0": plan}, camera=Camera(math.radians(87), 0.4, 1.2))
        kn.trip(0, set(), [2])
        p = self.value("p_frame", condition_id="in_view")
        look = (1 - (1 - p) ** 3) * self.value("q_learn", kind="I", condition_id="pass")
        self.assertEqual(kn.normal[2], 0.0)       # N waits for A's normal confirmation (section 8) ...
        self.assertAlmostEqual(kn.obj[2], look)  # ... the look does not
        kn.on_spot_seen(2, True)
        self.assertAlmostEqual(kn.normal[2], 1 - (1 - p) ** 3)
        self.assertAlmostEqual(kn.obj[2], look)  # same pass, not counted twice
        self.assertNotEqual(kn.core.fingerprint, self.kn.core.fingerprint)  # table / perspective maps cannot be swapped
        kn.pose_measured(2, 0, 0.0)  # A detected an event at S2 and measured the pose
        kn.trip(0, {2}, [2])         # B passed it while A knew it was abnormal: B may have seen the event
        self.assertAlmostEqual(kn.cur[2], 1 - (1 - p) ** 3)
        kn.pose_measured(2, 1, 0.0)  # a new C version voids the sighting
        self.assertEqual(kn.cur[2], 0.0)

    def test_pass_waits_for_a_normal_confirmation(self):
        kn = self.kn  # p_acq table variant
        kn.trip(0, set(), [1, 2, 3])
        self.assertEqual(kn.normal[2], 0.0)  # section 8: no pass evidence before A sees the spot normal
        kn.on_spot_seen(2, True)
        self.assertAlmostEqual(kn.normal[2], self.value("p_acq", kind="N", site_id="S2", route_id="to0",
                                                        condition_id="pass"))
        kn.trip(0, set(), [1, 2, 3])
        kn.on_spot_seen(3, False)  # abnormal when A looked: the snapshot is not a normal one
        self.assertEqual(kn.normal[3], 0.0)
        fresh = CoreKnowledge(self.world, store(), SPOTS)
        fresh.trip(0, {1}, [1, 2, 3])  # a spot A knew was abnormal is not a normal pass
        fresh.on_spot_seen(1, True)
        self.assertEqual(fresh.normal[1], 0.0)
        fresh.on_spot_seen(2, True)  # the same trip still counts for a spot that was normal
        self.assertGreater(fresh.normal[2], 0.0)

    def test_requests_reset_and_wrong_handling_invalidate(self):
        kn = self.kn
        kn.on_success(1)  # A verified a repair at spot 1
        self.assertEqual(kn.normal[1], 1.0)
        self.assertAlmostEqual(kn.obj[1], self.value("q_learn", kind="I", condition_id="verified"))
        kn.on_wrong(1)
        self.assertEqual(kn.obj[1], 0.0)
        kn.on_missing(1, {NORM})
        self.assertEqual(kn.normal[1], 0.0)
        kn.on_success(2)
        self.world.a_epoch = 1
        kn.reset()
        self.assertEqual(kn.normal[2], 0.0)
        self.assertEqual(kn.obj[2], 0.0)

    def test_not_found_is_diagnostic_and_a_new_pose_voids_old_c(self):
        kn = self.kn
        kn.pose_measured(3, 0, 0.0)
        payload = dict(items={CUR})
        kn.send(3, payload, payload["items"], 1)
        kn.ack(payload, 0)
        before = kn.cur[3]
        kn.on_missing(3, {CUR})  # not_found: cause unknown
        self.assertEqual(kn.cur[3], before)
        kn.pose_measured(3, 1, 0.0)  # A re-measured after the object moved
        self.assertEqual(kn.cur[3], 0.0)


@unittest.skipUnless((ROOT / "sim" / "world.py").exists() and load_parameters() is not None,
                     "team sim/ not in this checkout")
class EndToEnd(unittest.TestCase):
    def test_core_runs_in_every_condition_without_unsupported_reads(self):
        sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl")]
        from env2d import CONDITIONS, FULL, Sim2DEnv
        for cond in CONDITIONS:
            env = Sim2DEnv(cond)
            env.reset(seed=3)
            done = False
            while not done:
                _, _, done, _, info = env.step(FULL)
            self.assertIs(env.world.know, env.world.knows["core"])  # the policy input is knowledge_estimation
            self.assertIsNotNone(env.world.know.core.acquisition)  # ... with the perspective pass-by model
            for mode in ("core", "core_table"):
                kn = env.world.knows[mode]
                self.assertEqual((kn.unsupported, kn.contradictions), (0, 0), (cond, mode))
                self.assertTrue(all(mode in d["est"] for d in info["stats"]["dec"]))
                self.assertEqual(len(kn.decisions), len(info["stats"]["dec"]))  # one decision-time record each
                for rec in kn.decisions:
                    self.assertEqual(rec["fingerprint"], kn.core.fingerprint)
                    for kind in "NIC":
                        self.assertTrue({"raw", "cal", "safe", "reason", "sources"} <= set(rec[kind]))

    def test_notion_rule_checks_hold(self):
        from knowledge_estimation.integration.sim2d_trace import check, run
        for cond in ("limited", "lookalike_occluded"):
            for policy in ("minimal", "fixed_state"):
                world, _ = run(cond, 7, policy=policy)
                self.assertEqual(check(world), [], (cond, policy))


if __name__ == "__main__":
    unittest.main()
