import unittest

from rule_based_comm.policy_adapter import RulePolicyAdapter
from rule_based_comm.schemas import Action, BeliefState, Event


class AdapterAndValidationTest(unittest.TestCase):
    def test_ppo_compatible_action_id(self) -> None:
        action_id, decision = RulePolicyAdapter().predict(
            {
                "event_id": "e1",
                "target_confirmed": True,
                "has_similar_objects": False,
                "b_N": 0.0,
                "b_I": 0.0,
            }
        )
        self.assertEqual(1, action_id)
        self.assertEqual(("E", "N"), decision.selected_items)

    def test_gate_is_outside_eight_action_space(self) -> None:
        action_id, decision = RulePolicyAdapter().predict(
            {"event_id": "e1", "already_processing": True}
        )
        self.assertIsNone(action_id)
        self.assertEqual(Action.NO_COMM, decision.action)

    def test_invalid_belief_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BeliefState(b_N=1.1)

    def test_invalid_distance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Event(event_id="e1", object_distance_cm=0.0)


if __name__ == "__main__":
    unittest.main()

