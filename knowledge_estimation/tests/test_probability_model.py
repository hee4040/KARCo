"""Independent latent-outcome checks of the probability model and folder split."""

from itertools import product
from math import prod
import unittest

from knowledge_estimation import EstimatorConfig, KnowledgeEstimator, KnowledgeItem, KnowledgeKey
from knowledge_estimation.probabilistic.probabilities import (
    acquisition_probability, independent_union, receipt_given_no_ack,
)
from knowledge_estimation.simulator.engine import Simulator


class ProbabilityModelTests(unittest.TestCase):
    def test_no_ack_bayes_matches_enumerated_delivery_ack_outcomes(self):
        for delivery, ack in product((0., .1, .5, .9, 1.), (0., .2, .9, 1.)):
            with self.subTest(delivery=delivery, ack=ack):
                # An ACK cannot occur without delivery. Enumerate the complete
                # joint distribution, then condition by discarding ACK cases.
                outcomes = ((False, False, 1-delivery),
                            (True, False, delivery*(1-ack)),
                            (True, True, delivery*ack))
                no_ack_mass = sum(mass for _, observed_ack, mass in outcomes if not observed_ack)
                if not no_ack_mass:
                    with self.assertRaises(ValueError):
                        receipt_given_no_ack(delivery, ack)
                else:
                    valid_mass = sum(mass for received, observed_ack, mass in outcomes
                                     if received and not observed_ack)
                    self.assertAlmostEqual(receipt_given_no_ack(delivery, ack), valid_mass/no_ack_mass)

    def test_retry_and_shared_crop_quality_match_joint_latent_enumeration(self):
        for attempts in (((.9, .9), (.9, .9)), ((.6, .8), (.8, .3)), ((1., 0.), (0., .9))):
            with self.subTest(attempts=attempts):
                item = KnowledgeItem(KnowledgeKey("I", "Q"))
                est = KnowledgeEstimator()
                no_ack_mass, valid_mass = 0., 0.
                worlds = [[(False, False, 1-d), (True, False, d*(1-a)), (True, True, d*a)]
                          for d, a in attempts]
                for deliveries in product(*worlds):
                    if any(ack for _, ack, _ in deliveries):
                        continue
                    for good_content, quality_mass in ((False, .05), (True, .95)):
                        weight = prod(outcome[2] for outcome in deliveries)*quality_mass
                        no_ack_mass += weight
                        if good_content and any(received for received, _, _ in deliveries):
                            valid_mass += weight
                for index, (delivery, ack) in enumerate(attempts, 1):
                    est.record_send("same-crop", [item], attempt_id=index, p_delivery=delivery, p_ack=ack)
                    est.observe_ack_timeout("same-crop", attempt_id=index)
                self.assertAlmostEqual(est.estimate(item).probability, valid_mass/no_ack_mass)

    def test_source_combination_matches_independent_acquisition_worlds(self):
        probabilities = (.72, .9, .4)
        expected = 0.
        for outcomes in product((False, True), repeat=3):
            mass = prod(p if success else 1-p for p, success in zip(probabilities, outcomes))
            if any(outcomes):
                expected += mass
        self.assertAlmostEqual(independent_union(probabilities), expected)
        self.assertAlmostEqual(independent_union((.72, .9)), .972)
        self.assertAlmostEqual(independent_union((.95, .9)), .995)
        self.assertEqual(independent_union(()), 0)
        self.assertEqual(independent_union((0, 1)), 1)
        self.assertGreater(independent_union((1e-20,)), 0)

    def test_pass_acquisition_is_a_conditional_chain(self):
        # The observation/storage probabilities below are conditional, not
        # probabilities of three independent events.
        worlds = {"no_pass": .4, "pass_no_view": .6*.2,
                  "view_no_store": .6*.8*.1, "valid_acquisition": .6*.8*.9}
        self.assertAlmostEqual(sum(worlds.values()), 1)
        self.assertAlmostEqual(acquisition_probability(.6, .8, .9), worlds["valid_acquisition"])

    def test_same_content_and_declared_dependent_sources_are_not_independent(self):
        item = KnowledgeItem(KnowledgeKey("I", "Q"))
        est = KnowledgeEstimator()
        for msg in ("crop1", "crop2"):
            est.record_send(msg, [item])
            est.observe_ack(msg, [item], memory_epoch=0)
        self.assertAlmostEqual(est.estimate(item).probability, .95)
        est = KnowledgeEstimator()
        est.record_pass_by("camera-pass", [item], p_visible=.8, p_store=.9, dependency_group="same-recognizer")
        est.record_result("camera-task", [item], dependency_group="same-recognizer")
        est.verify_task("camera-task", success=True)
        self.assertAlmostEqual(est.estimate(item).probability, .9)
        conservative = KnowledgeEstimator(config=EstimatorConfig(independent_sources=False))
        conservative.record_pass_by("pass", [item])
        conservative.record_result("task", [item])
        conservative.verify_task("task", success=True)
        self.assertAlmostEqual(conservative.estimate(item).probability, .9)

    def test_direct_probability_api_rejects_bool_string_and_nonfinite_values(self):
        for value in (True, "0.8", None, float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    receipt_given_no_ack(value, .9)
                with self.assertRaises(ValueError):
                    acquisition_probability(1., value, .9)
                with self.assertRaises(ValueError):
                    independent_union((value,))


class ObservableGeometryTests(unittest.TestCase):
    def test_known_out_of_range_and_out_of_fov_do_not_create_pass_evidence(self):
        for scene in ({"b_camera_range": .5}, {"b_camera_fov": .2}):
            with self.subTest(scene=scene):
                sim = Simulator(scene=scene)
                sim.step(6)
                passes = [event for event in sim.export_observations() if event["kind"] == "pass_by"]
                self.assertEqual(len(passes), 1)
                self.assertEqual(passes[0]["data"]["p_visible"], 0)
                self.assertEqual(sim.a_view("Q")["baseline"]["b_I"], 0)

    def test_pass_geometry_is_frozen_from_a_known_pose_at_that_opportunity(self):
        sim = Simulator()
        sim.step(6)
        before = sim.a_view("Q")["planned_pass"]
        # A may observe a different pose later; that does not retrospectively
        # change the viewing geometry of an earlier acquisition opportunity.
        sim._known_positions["Q"] = (11., 7.)
        self.assertEqual(sim.a_view("Q")["planned_pass"], before)

    def test_a_previously_observed_abnormal_pose_does_not_support_normal_layout(self):
        sim = Simulator()
        sim.move_object("Q")
        sim._known_normal["Q"] = False  # A-side sensor observation state.
        sim.step(6)
        passes = [event for event in sim.export_observations() if event["kind"] == "pass_by"]
        self.assertFalse(passes[0]["data"]["normal_before"])
        sim._observe("pass_confirmation", pass_id="route-S2", normal_after=True)
        self.assertEqual(sim.a_view("Q")["baseline"]["b_N"], 0)

    def test_compatibility_paths_resolve_to_one_implementation(self):
        from knowledge_estimation.estimator import KnowledgeEstimator as OldEstimator
        from knowledge_estimation.probabilistic import KnowledgeEstimator as NewEstimator
        from knowledge_estimation.dataset.collection import collect as old_collect
        from knowledge_estimation.learning.dataset.collection import collect as new_collect
        self.assertIs(OldEstimator, NewEstimator)
        self.assertIs(old_collect, new_collect)
        self.assertEqual(NewEstimator.__module__, "knowledge_estimation.probabilistic.estimator")


if __name__ == "__main__":
    unittest.main()
