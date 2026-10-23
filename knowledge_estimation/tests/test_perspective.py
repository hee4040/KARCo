"""Perspective-taking acquisition model: geometry sense, A-only views, reports."""

import csv
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.probabilistic.perspective import (acquisition_probability, _detect_probability,
                                                            A_ASSUMPTION, A_ASSUMPTION_SIZE)
from knowledge_estimation.simulation.collection_design import build_plan, allocations
from knowledge_estimation.simulation.collect_designed import A_input
from knowledge_estimation.simulation.compare_perspective import (view_for, reports_from, apply_map,
                                                                 planned_context, run, rescore)
from knowledge_estimation.simulation.collect_stochastic import run as collect


def view(**changes):
    base = dict(route_origin=[-3., 1.], route_direction_deg=0., route_length_m=6., planned_speed_mps=1.,
                object_xy=[0., 2.], walls=[],
                camera=dict(fov_deg=87., z_min_m=.6, z_max_m=6., dt_s=1/30),
                dwell_seconds={"N": .3, "I": .6}, planned_heading_deg=90., pose_reports=[])
    return {**base, **changes}


class PerspectiveTests(unittest.TestCase):
    def test_facing_object_is_likely_and_facing_away_is_impossible(self):
        self.assertGreater(acquisition_probability(view(), samples=64, seed=1)["N"], .9)
        self.assertEqual(acquisition_probability(view(planned_heading_deg=-90.), samples=64, seed=1), {"N": 0., "I": 0.})

    def test_wall_from_A_map_blocks_sight(self):
        wall = [[-3., 1.4, 3., 1.6]]
        self.assertEqual(acquisition_probability(view(walls=wall), samples=32, seed=2), {"N": 0., "I": 0.})

    def test_longer_dwell_item_is_never_more_likely(self):
        for heading in (0., 45., 90., 150.):
            p = acquisition_probability(view(planned_heading_deg=heading), samples=64, seed=3)
            self.assertLessEqual(p["I"], p["N"])

    def test_unknown_heading_marginalises_and_reports_need_a_plan(self):
        unknown = acquisition_probability(view(planned_heading_deg=None), samples=256, seed=4)["N"]
        self.assertTrue(.05 < unknown < .95)
        with self.assertRaises(ValueError):
            acquisition_probability(view(planned_heading_deg=None, pose_reports=[(0., -3., 1., 1.5)]), samples=4, seed=5)

    def test_view_contains_plan_and_map_only(self):
        plan = build_plan(seed=3)
        draw = next(allocations(plan))
        v = view_for(plan["config"], A_input(plan["config"], draw), draw, True, [])
        self.assertEqual(set(v), {"route_origin", "route_direction_deg", "route_length_m", "planned_speed_mps",
                                  "object_xy", "walls", "camera", "dwell_seconds", "planned_heading_deg", "pose_reports"})
        self.assertIsNone(view_for(plan["config"], A_input(plan["config"], draw), draw, False, [])["planned_heading_deg"])

    def test_forward_mount_view_points_camera_along_route(self):
        plan = build_plan(seed=3, camera_mount="forward")
        draw = next(allocations(plan))
        v = view_for(plan["config"], A_input(plan["config"], draw), draw, True, [])
        self.assertAlmostEqual(v["planned_heading_deg"], v["route_direction_deg"])
        self.assertAlmostEqual(v["route_direction_deg"], draw["heading_degrees"])

    def test_rotated_route_matches_rotated_world(self):
        # Same pass seen in a world rotated by 90 degrees about the object.
        straight = acquisition_probability(view(planned_heading_deg=30.), samples=64, seed=9)
        rotated = acquisition_probability(view(route_origin=[1., -1.], route_direction_deg=90.,
                                               planned_heading_deg=120.), samples=64, seed=9)
        for kind in "NI":
            self.assertAlmostEqual(straight[kind], rotated[kind], delta=.1)

    def test_reports_use_first_frame_at_or_after_each_report_time(self):
        frames = [{"time": t/10, "pose": [t, 0., 0.]} for t in range(1, 25)]
        self.assertEqual([r[0] for r in reports_from(frames, 1.)], [.1, 1., 2.])
        self.assertEqual(reports_from(frames, None), [])

    def test_size_assumption_lowers_far_detection_and_keeps_near(self):
        camera = view()["camera"]
        self.assertEqual(_detect_probability(A_ASSUMPTION, 5., camera), .9)
        near, far = _detect_probability(A_ASSUMPTION_SIZE, 1., camera), _detect_probability(A_ASSUMPTION_SIZE, 4., camera)
        self.assertEqual(near, .9)
        self.assertLess(far, near)
        self.assertEqual(_detect_probability(A_ASSUMPTION_SIZE, 60., camera), 0.)
        close = acquisition_probability(view(), samples=64, seed=1, assumption=A_ASSUMPTION_SIZE)["N"]
        distant = acquisition_probability(view(object_xy=[0., 4.5], route_origin=[-3., 1.]), samples=64, seed=1,
                                          assumption=A_ASSUMPTION_SIZE)["N"]
        self.assertLess(distant, close)

    def test_planned_context_is_the_planned_closest_distance(self):
        for mount in ("free", "forward"):
            plan = build_plan(seed=5, camera_mount=mount)
            for draw in list(allocations(plan))[:30]:
                A = A_input(plan["config"], draw)
                context = planned_context(plan["config"], A, draw)
                self.assertAlmostEqual(context["distance_m"], draw["distance_m"], places=9)
                self.assertEqual(context["speed_mps"], draw["speed_mps"])

    def test_selective_stage_and_rescore_reproduce_metrics(self):
        base = build_plan(seed=42, epsilon=.49, alpha=.5, camera_mount="forward", ood_distance_bins_m=[[2.5, 3.]])
        with tempfile.TemporaryDirectory() as directory:
            dataset, out = Path(directory)/"data", Path(directory)/"cmp"
            collect(dataset, base, profile_name="medium", rollouts=2, workers=2)
            run(dataset, out, samples=8, workers=2, validate=False, tiers=["L2"])
            with (out/"metrics.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["stage"] for r in rows if r["tier"] == "L2"}, {"raw", "calibrated", "selective"})
            coverage = {r["split"]: float(r["value"]) for r in rows if r["stage"] == "selective_coverage"}
            self.assertEqual(coverage["ood_distance"], 0.)  # every OOD-distance plan is outside support
            # Support is the observed calibration range, so edge-of-bin plans may fall outside.
            self.assertGreater(coverage["test"], .9)
            ood = [r for r in rows if r["split"] == "ood_distance" and r["tier"] == "L2" and r["factor"] == "all"]
            by_stage = {(r["stage"], r["kind"]): r["brier"] for r in ood}
            self.assertEqual(by_stage["selective", "N"], by_stage["raw", "N"])
            rescore(dataset, out, Path(directory)/"rescored")
            with (Path(directory)/"rescored"/"metrics.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(list(csv.DictReader(stream)), rows)

    def test_calibration_map_refuses_unsupported_scores(self):
        mapping = {"support": [.2, .8], "blocks": [{"upper": .5, "value": .1}, {"upper": .8, "value": .7}]}
        self.assertEqual(apply_map(mapping, .3), .1)
        self.assertEqual(apply_map(mapping, .8), .7)
        self.assertIsNone(apply_map(mapping, .9))


if __name__ == "__main__":
    unittest.main()
