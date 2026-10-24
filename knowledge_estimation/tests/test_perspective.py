"""Perspective-taking pass-by model (probabilistic/perspective.py): geometry, rates, None contract."""
import math
import unittest

from knowledge_estimation.parameters import ParameterStore
from knowledge_estimation.probabilistic.perspective import (
    Camera, PerspectiveAcquisition, in_view, pass_probability, planned_frames)

CAMERA = Camera(math.radians(87), 0.4, 1.2)
SHA = "0" * 64


def rates(**values):
    """A fixture store with the given p_frame / q_learn successes and trials."""
    entries = []
    for name, (condition, kind, s, n) in values.items():
        entries.append(dict(name=name.split("__")[0], parameter_id=f"fx:{name}", evidence_ids=["fixture"],
                            input_sha256=SHA, trials=n, successes=s, value=s / n,
                            context=dict(receiver_id="B", kind=kind, site_id="", route_id="", condition_id=condition)))
    return ParameterStore(dict(schema_version=1, origin="fixture", parameters=entries), allow_fixture=True)


class Geometry(unittest.TestCase):
    def test_depth_window_and_field_of_view(self):
        self.assertTrue(in_view((0, 0, 0), (1.0, 0), CAMERA))
        self.assertFalse(in_view((0, 0, 0), (0.3, 0), CAMERA))     # closer than the depth window
        self.assertFalse(in_view((0, 0, 0), (1.3, 0), CAMERA))     # farther
        self.assertTrue(in_view((0, 0, 0), (0.7, 0.6), CAMERA))    # 40.6 deg off axis, inside 43.5
        self.assertFalse(in_view((0, 0, 0), (0.6, 0.7), CAMERA))   # 49.4 deg off axis
        self.assertFalse(in_view((0, 0, math.pi), (1.0, 0), CAMERA))  # forward mount: behind B

    def test_known_walls_block_and_count_frames(self):
        wall = ((0.5, -0.1), (0.5, 0.1))
        self.assertFalse(in_view((0, 0, 0), (1.0, 0), CAMERA, [wall]))
        plan = [(x / 10, 0, 0) for x in range(10)]  # driving toward a target at x = 1.5
        self.assertEqual(planned_frames(plan, (1.5, 0), CAMERA), 7)  # x = 0.3 .. 0.9: 1.2 .. 0.6 m away
        self.assertEqual(pass_probability(0, 0.3), 0)
        self.assertAlmostEqual(pass_probability(2, 0.3), 0.51)


class Acquisition(unittest.TestCase):
    PLAN = {"r": [(0, 0, 0)] * 4}
    SITES = {"S": (1.0, 0), "far": (5.0, 0)}

    def test_rates_combine_per_item(self):
        model = PerspectiveAcquisition(self.PLAN, self.SITES, CAMERA,
                                       rates(p_frame=("in_view", "", 1, 4), q_learn=("pass", "I", 9, 10)))
        ctx = dict(receiver_id="B", site_id="S", route_id="r", condition_id="pass")
        stored, usable, ids = model({**ctx, "kind": "N"})
        self.assertAlmostEqual(stored, 1 - 0.75 ** 4)
        self.assertEqual(usable, 1.0)
        self.assertEqual(ids, ("fx:p_frame",))
        stored, usable, ids = model({**ctx, "kind": "I"})
        self.assertAlmostEqual(usable, 0.9)
        self.assertEqual(ids, ("fx:p_frame", "fx:q_learn"))
        self.assertEqual(model({**ctx, "kind": "N", "site_id": "far"})[0], 0)  # never in view

    def test_unknown_plan_or_unmeasured_rate_is_none(self):
        ctx = dict(receiver_id="B", kind="I", site_id="S", route_id="r", condition_id="pass")
        model = PerspectiveAcquisition(self.PLAN, self.SITES, CAMERA, ParameterStore())
        self.assertEqual(model(ctx)[:2], (None, None))
        self.assertEqual(model({**ctx, "route_id": "new"})[:2], (None, None))
        same = lambda plan, camera: PerspectiveAcquisition(plan, self.SITES, camera, ParameterStore()).fingerprint
        self.assertEqual(same(self.PLAN, CAMERA), same({"r": [(0, 0, 0)] * 3}, CAMERA))  # a plan is an input
        self.assertNotEqual(same(self.PLAN, CAMERA), same(self.PLAN, Camera(math.radians(60), 0.4, 1.2)))


if __name__ == "__main__":
    unittest.main()
