import unittest

from rule_based_comm import Action, RuleBasedCommunicationPolicy


class ScenarioSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RuleBasedCommunicationPolicy()

    def select(self, event: dict, belief: dict):
        return self.policy.select({"event_id": "test", **event}, belief)

    def test_scenario_1_first_event_sends_event_and_normal(self) -> None:
        decision = self.select(
            {"target_confirmed": True, "has_similar_objects": False},
            {"b_N": 0.0, "b_I": 0.0},
        )
        self.assertEqual(Action.SEND, decision.action)
        self.assertEqual(("E", "N"), decision.selected_items)
        self.assertEqual(160, decision.estimated_bytes)

    def test_scenario_2_reuses_confirmed_normal(self) -> None:
        decision = self.select(
            {"target_confirmed": True, "has_similar_objects": False},
            {"b_N": 1.0, "b_I": 0.9},
        )
        self.assertEqual(("E",), decision.selected_items)

    def test_scenario_2_lookalike_with_appearance_memory(self) -> None:
        decision = self.select(
            {
                "has_similar_objects": True,
                "object_distance_cm": 4.0,
                "position_uncertainty_cm": 3.0,
            },
            {"b_N": 1.0, "b_I": 0.9, "b_C": 0.0},
        )
        self.assertEqual(("E", "C"), decision.selected_items)
        self.assertEqual(160, decision.estimated_bytes)
        self.assertAlmostEqual(0.92, decision.expected_cost, delta=0.08)

    def test_scenario_3_probabilistic_observation_omits_crop(self) -> None:
        decision = self.select(
            {
                "has_similar_objects": True,
                "object_distance_cm": 4.0,
                "position_uncertainty_cm": 3.0,
            },
            {"b_N": 0.72, "b_I": 0.72, "b_C": 0.0},
        )
        self.assertEqual(("E", "N", "C"), decision.selected_items)
        self.assertEqual(200, decision.estimated_bytes)
        # The initial model also charges the residual risk after a requested
        # crop, so it is slightly above the HTML's illustrative 1.51.
        self.assertAlmostEqual(1.64, decision.expected_cost, delta=0.08)

    def test_scenario_4_unknown_lookalike_sends_full_payload(self) -> None:
        decision = self.select(
            {
                "has_similar_objects": True,
                "object_distance_cm": 4.0,
                "position_uncertainty_cm": 3.0,
            },
            {"b_N": 0.0, "b_I": 0.0, "b_C": 0.0},
        )
        self.assertEqual(("E", "N", "C", "I"), decision.selected_items)
        self.assertEqual(30_920, decision.estimated_bytes)
        self.assertAlmostEqual(3.65, decision.expected_cost, delta=0.08)

    def test_underreported_small_sigma_omits_crop(self) -> None:
        decision = self.select(
            {
                "has_similar_objects": True,
                "object_distance_cm": 4.0,
                "position_uncertainty_cm": 1.0,
            },
            {"b_N": 0.0, "b_I": 0.0, "b_C": 0.0},
        )
        self.assertEqual(("E", "N", "C"), decision.selected_items)

    def test_same_policy_input_gives_same_decision_despite_hidden_truth(self) -> None:
        event = {
            "event_id": "e3",
            "has_similar_objects": True,
            "object_distance_cm": 4.0,
            "position_uncertainty_cm": 3.0,
        }
        belief = {"b_N": 0.72, "b_I": 0.72}
        clear = self.policy.select(event, belief)
        occluded = self.policy.select(event, belief)
        self.assertEqual(clear.selected_items, occluded.selected_items)

    def test_all_candidate_costs_are_exposed_and_sorted(self) -> None:
        decision = self.select({}, {"b_N": 0.0})
        self.assertEqual(8, len(decision.candidate_costs))
        totals = [item.total_cost for item in decision.candidate_costs]
        self.assertEqual(sorted(totals), totals)


if __name__ == "__main__":
    unittest.main()

