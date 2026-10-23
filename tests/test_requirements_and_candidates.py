import unittest

from rule_based_comm.candidates import generate_candidates
from rule_based_comm.requirements import determine_requirements
from rule_based_comm.schemas import Action, Event


class RequirementsAndCandidatesTest(unittest.TestCase):
    def test_generates_eight_unique_candidates_with_event(self) -> None:
        candidates = generate_candidates()
        self.assertEqual(8, len(candidates))
        self.assertEqual(8, len(set(candidates)))
        self.assertTrue(all("E" in candidate for candidate in candidates))

    def test_unconfirmed_target_requires_reobservation(self) -> None:
        result = determine_requirements(Event(event_id="e1", target_confirmed=False))
        self.assertEqual(Action.REOBSERVE, result.status)
        self.assertFalse(result.required)

    def test_unchanged_processing_event_is_suppressed(self) -> None:
        result = determine_requirements(Event(event_id="e1", already_processing=True))
        self.assertEqual(Action.NO_COMM, result.status)

    def test_changed_processing_event_is_reconsidered(self) -> None:
        result = determine_requirements(
            Event(event_id="e1", already_processing=True, scene_changed=True)
        )
        self.assertEqual(Action.SEND, result.status)

    def test_lookalikes_require_current_pose_and_identification(self) -> None:
        result = determine_requirements(
            Event(event_id="e1", has_similar_objects=True)
        )
        self.assertEqual(frozenset(("E", "N", "C", "I")), result.required)


if __name__ == "__main__":
    unittest.main()

