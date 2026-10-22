"""Geometry tests; no receiver ground truth is used as an input."""

import math
import unittest

from knowledge_estimation.visibility import (
    Pose2D, Rectangle, VisibilityConfig, estimate_visibility,
)


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = VisibilityConfig(max_range_m=5, horizontal_fov_rad=math.pi / 2)
        self.pose = Pose2D(0, 0, 0)

    def test_range_and_fov_boundaries(self):
        for target in ((5, 0), (1, 1), (1, -1)):
            with self.subTest(target=target):
                self.assertEqual(estimate_visibility([self.pose], target, self.config).p_visible, 0.8)
        for target in ((5.001, 0), (-1, 0), (1, 1.001)):
            with self.subTest(target=target):
                self.assertEqual(estimate_visibility([self.pose], target, self.config).p_visible, 0)

    def test_heading_wrap_and_full_circle(self):
        result = estimate_visibility([Pose2D(0, 0, 2 * math.pi)], (2, 0), self.config)
        self.assertEqual(result.p_visible, 0.8)
        full_circle = VisibilityConfig(5, 2 * math.pi)
        self.assertEqual(estimate_visibility([self.pose], (-2, 0), full_circle).p_visible, 0.8)

    def test_occluder_crossing_tangent_and_containment_block(self):
        boxes = (
            Rectangle(1, -1, 2, 1),  # crosses the viewing segment
            Rectangle(1, 0, 2, 1),  # tangent along its boundary
            Rectangle(3, -1, 4, 1),  # contains the target
            Rectangle(0, -1, 0, 1),  # zero-width wall at the camera
        )
        for box in boxes:
            with self.subTest(box=box):
                self.assertEqual(estimate_visibility([self.pose], (3, 0), self.config, [box]).p_visible, 0)

    def test_obstacles_outside_segment_do_not_block(self):
        boxes = [Rectangle(4, -1, 5, 1), Rectangle(-2, -1, -1, 1), Rectangle(1, 1, 2, 2)]
        self.assertEqual(estimate_visibility([self.pose], (3, 0), self.config, boxes).p_visible, 0.8)

    def test_empty_and_repeated_samples(self):
        empty = estimate_visibility([], (2, 0), self.config)
        self.assertEqual((empty.p_visible, empty.visible_sample_count, empty.total_sample_count), (0, 0, 0))
        repeated = estimate_visibility([self.pose] * 10, (2, 0), self.config)
        self.assertEqual((repeated.p_visible, repeated.visible_sample_count, repeated.total_sample_count), (0.8, 10, 10))

    def test_only_sampled_poses_count(self):
        route = [Pose2D(-10, 0, 0), Pose2D(10, 0, math.pi)]
        result = estimate_visibility(iter(route), (0, 0), self.config)
        self.assertEqual((result.p_visible, result.total_sample_count), (0, 2))

    def test_coincident_target_and_zero_range(self):
        config = VisibilityConfig(0, math.pi / 4)
        self.assertEqual(estimate_visibility([Pose2D(0, 0, math.pi)], (0, 0), config).p_visible, 0.8)
        self.assertEqual(estimate_visibility([self.pose], (0, 0), config, [Rectangle(0, 0, 1, 1)]).p_visible, 0)

    def test_invalid_geometry_and_configuration(self):
        factories = (
            lambda: Pose2D(float("nan"), 0, 0),
            lambda: Pose2D(0, 0, float("inf")),
            lambda: Rectangle(2, 0, 1, 1),
            lambda: Rectangle(0, 0, 1, float("inf")),
            lambda: VisibilityConfig(-1, math.pi),
            lambda: VisibilityConfig(1, 0),
            lambda: VisibilityConfig(1, 2 * math.pi + 0.01),
            lambda: VisibilityConfig(1, math.pi, float("nan")),
            lambda: VisibilityConfig(1, math.pi, 1.01),
            lambda: estimate_visibility([], (float("inf"), 0), self.config),
        )
        for factory in factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()


if __name__ == "__main__":
    unittest.main()
