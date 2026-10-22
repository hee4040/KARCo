"""Independent geometry oracle and corrupt-log detection for the data audit."""

from copy import deepcopy
import json
from math import pi
from pathlib import Path
import unittest

from knowledge_estimation.simulation.audit_acquisition import occluded, seen, audit_trace, independent_label, condition_metrics
from knowledge_estimation.simulation.world import run_pass
from knowledge_estimation.tests.fixtures import snapshots


class AcquisitionAuditTests(unittest.TestCase):
    def test_closed_segment_rectangles_include_touching_but_not_infinite_line(self):
        wall = [[1., 1., 2., 2.]]
        self.assertTrue(occluded((0., 0.), (3., 3.), wall))
        self.assertTrue(occluded((0., 1.), (3., 1.), wall))
        self.assertTrue(occluded((1.5, 1.5), (1.6, 1.6), wall))
        self.assertFalse(occluded((0., 0.), (.5, .5), wall))
        self.assertFalse(occluded((0., 0.), (3., 0.), wall))

    def test_camera_frame_oracle_rejects_behind_camera_and_occlusion(self):
        config = {"camera_min_depth_m": .6, "camera_range_m": 6., "camera_fov_degrees": 87.}
        self.assertTrue(seen((0., 0., 0.), (2., 0.), config, []))
        self.assertFalse(seen((0., 0., pi), (2., 0.), config, []))
        self.assertFalse(seen((0., 0., 0.), (2., 0.), config, [[1., -.1, 1.2, .1]]))
        self.assertFalse(seen((0., 0., 0.), (.5, 0.), config, []))

    def test_corrupted_visibility_and_acquisition_records_are_detected(self):
        config = json.loads((Path(__file__).parents[1]/"simulation"/"scenario.json").read_text(encoding="utf-8"))
        result = run_pass(config, 1., 90., False)
        trace = dict(speed_mps=config["speed_mps"], offset=1., heading=90., walls=[],
                     frames=result.frames, acquired_at=result.acquired_at, delivered=True)
        self.assertEqual(audit_trace(trace, config)[2], [])
        corrupt = deepcopy(trace)
        corrupt["frames"][0]["visible"] = not corrupt["frames"][0]["visible"]
        self.assertIn("visibility_dwell_or_stored", audit_trace(corrupt, config)[2])
        corrupt = deepcopy(trace)
        corrupt["acquired_at"] = {}
        self.assertIn("acquisition_time", audit_trace(corrupt, config)[2])

    def test_independent_labels_distinguish_expiry_and_version(self):
        snapshot, world = snapshots(7., stored=True)
        self.assertEqual(independent_label(snapshot, world), {"N": 1, "C": 0, "I": 1})
        world["versions"]["I"] = 2
        self.assertEqual(independent_label(snapshot, world)["I"], 0)

    def test_condition_statistics_do_not_hide_probability_rate_gap(self):
        row = dict(stage="test", kind="N", speed_mps=1., route_standoff_m=1., heading_degrees=90., walls=False, K=1, prediction=.5)
        result = condition_metrics([row])[0]
        self.assertEqual(result["unique_designs"], 1)
        self.assertEqual(result["signed_gap"], -.5)
        self.assertEqual(result["brier"], .25)
