"""Stochastic world: zero-noise equivalence, physical noise only, p* contract."""

from itertools import islice
import json
from math import radians
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.measurement import read_jsonl
from knowledge_estimation.simulation.collection_design import build_plan, allocations
from knowledge_estimation.simulation.collect_designed import sample_pass
from knowledge_estimation.simulation.collect_stochastic import (run, validate_collection, auroc,
                                                                build_stochastic_plan, seed_for, world_config)
from knowledge_estimation.simulation.stochastic_world import rollout, profile, detection_probability, focal_px


class StochasticWorldTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(seed=7)
        self.config = self.plan["config"]
        self.draws = list(islice(allocations(self.plan), 0, 2160, 9))

    def test_zero_noise_profile_reproduces_deterministic_world(self):
        none = profile("none")
        for draw in self.draws:
            old_frames, old = sample_pass(self.config, draw)
            frames, new = rollout(self.config, draw, none, 123)
            self.assertEqual(old, new)
            self.assertEqual([(f["time"], f["pose"], f["visible"], f["stored"]) for f in old_frames],
                             [(f["time"], f["pose"], f["visible"], f["stored"]) for f in frames])

    def test_same_seed_reproduces_and_different_seeds_vary(self):
        medium = profile("medium")
        outcomes = set()
        for draw in self.draws:
            self.assertEqual(rollout(self.config, draw, medium, 5), rollout(self.config, draw, medium, 5))
            outcomes.add(tuple(sorted(rollout(self.config, draw, medium, 6)[1].items())) !=
                         tuple(sorted(rollout(self.config, draw, medium, 5)[1].items())))
        self.assertIn(True, outcomes)

    def test_noise_never_creates_acquisition_without_geometric_visibility(self):
        high = profile("high")
        for draw in self.draws:
            frames, acquired = rollout(self.config, draw, high, 11)
            self.assertTrue(all(f["visible"] or not f["detected"] for f in frames))
            for kind, at in acquired.items():
                hit = next(f for f in frames if f["time"] == at)
                self.assertTrue(hit["detected"] and hit["contiguous_visible_seconds"] >= self.config["storage_dwell_seconds"][kind])

    def test_detection_factors_are_monotone_physical_terms(self):
        medium = profile("medium")
        args = dict(fov=radians(87), f_px=focal_px(self.config, medium["shared"]), det=medium["detection"], shared=medium["shared"])
        near = detection_probability(depth=1., bearing=0., bearing_rate=0., **args)
        far = detection_probability(depth=5., bearing=0., bearing_rate=0., **args)
        edge = detection_probability(depth=1., bearing=radians(43), bearing_rate=0., **args)
        blurred = detection_probability(depth=1., bearing=0., bearing_rate=4., **args)
        self.assertGreater(near, far)
        self.assertGreater(near, edge)
        self.assertGreater(near, blurred)
        self.assertLessEqual(near, medium["detection"]["p_max"])
        self.assertEqual(detection_probability(depth=5., bearing=0., bearing_rate=9., **{**args, "det": None}), 1.)

    def test_forward_mount_plan_is_separate_and_camera_follows_route(self):
        self.assertEqual(build_plan(seed=7, camera_mount="free"), self.plan)
        self.assertNotIn("camera_mount", self.plan)
        forward = build_plan(seed=7, camera_mount="forward")
        self.assertNotEqual(forward["plan_id"], self.plan["plan_id"])
        rows = list(islice(allocations(forward), 0, 2160, 9))
        self.assertEqual({r["side"] for r in rows}, {-1, 1})
        self.assertTrue(all(r["camera_mount"] == "forward" and "offset" not in r for r in rows))
        for draw in rows[:20]:
            frames, _ = rollout(forward["config"], draw, profile("none"), 1)
            self.assertTrue(all(abs(f["pose"][2]-radians(draw["heading_degrees"])) < 1e-12 for f in frames))
            # Mid-route closest approach at the drawn distance.
            mid = min(frames, key=lambda f: (f["pose"][0]-forward["config"]["object_xy"][0])**2+(f["pose"][1]-forward["config"]["object_xy"][1])**2)
            gap = ((mid["pose"][0]-forward["config"]["object_xy"][0])**2+(mid["pose"][1]-forward["config"]["object_xy"][1])**2)**.5
            self.assertAlmostEqual(gap, draw["distance_m"], delta=draw["speed_mps"]/30)

    def test_forward_draw_along_x_equals_free_draw_facing_forward(self):
        forward = dict(self.draws[0], camera_mount="forward", side=1, heading_degrees=0., walls=False)
        forward.pop("offset")
        free = dict(self.draws[0], heading_degrees=0., walls=False, offset=self.config["object_xy"][1]-self.draws[0]["distance_m"])
        none = profile("none")
        a, b = rollout(self.config, forward, none, 1), rollout(self.config, free, none, 1)
        self.assertEqual(a[1], b[1])
        self.assertTrue(all(abs(x-y) < 1e-9 for f, g in zip(a[0], b[0]) for x, y in zip(f["pose"], g["pose"])))

    def test_world_override_changes_only_the_true_sensor_with_paired_noise(self):
        nominal = build_stochastic_plan(self.plan, "medium", 4)
        narrow = build_stochastic_plan(self.plan, "medium", 4, {"camera_fov_degrees": 80.})
        self.assertNotEqual(narrow["stochastic_plan_id"], nominal["stochastic_plan_id"])
        draw = self.draws[0]
        self.assertEqual(seed_for(narrow, draw["trial_id"], 3), seed_for(nominal, draw["trial_id"], 3))
        self.assertEqual(world_config(self.config, narrow)["camera_fov_degrees"], 80.)
        self.assertEqual(world_config(self.config, nominal), self.config)
        self.assertEqual(self.config["camera_fov_degrees"], 87.)
        with self.assertRaises(ValueError):
            build_stochastic_plan(self.plan, "medium", 4, {"object_xy": [0., 0.]})

    def test_auroc_handles_ties_and_single_class(self):
        self.assertEqual(auroc([(.9, 1), (.1, 0)]), 1.)
        self.assertEqual(auroc([(.5, 1), (.5, 0)]), .5)
        self.assertIsNone(auroc([(.5, 1)]))

    def test_end_to_end_labels_p_star_and_frozen_estimator(self):
        # Relaxed precision only to keep the contract test fast.
        base = build_plan(seed=42, epsilon=.49, alpha=.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"collection"
            report = run(root, base, profile_name="medium", rollouts=4, workers=2)
            self.assertTrue(report["quality_passed"])
            labels = list(read_jsonl(root/"test"/"B_labels.private.jsonl"))
            self.assertTrue(all(set(r["p_star"]) == {"N", "I"} for r in labels))
            self.assertIsNone(next(read_jsonl(root/"train"/"B_labels.private.jsonl"))["p_star"])
            A = next(read_jsonl(root/"test"/"A_inputs.jsonl"))
            self.assertFalse({"p_star", "heading_degrees", "walls"} & set(A))
            self.assertFalse((root/"calibration"/"raw_predictions.jsonl").exists())
            with (root/"test"/"B_labels.private.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_collection(root, 2)


if __name__ == "__main__":
    unittest.main()
