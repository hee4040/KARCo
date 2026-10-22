"""Protocol edge cases that matter when callbacks are retried or malformed."""

import unittest

from knowledge_estimation import (
    EstimatorConfig, KnowledgeEstimator, KnowledgeItem, KnowledgeKey,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.est = KnowledgeEstimator(memory_epoch=7)
        self.n = KnowledgeItem(KnowledgeKey("N", "P", "S1"))
        self.i = KnowledgeItem(KnowledgeKey("I", "P"))

    def test_old_needs_info_ack_replay_cannot_clear_recovered_knowledge(self):
        self.est.record_send("event-only", [])
        self.est.observe_ack("event-only", [], memory_epoch=7, needs_info=[self.n])
        self.est.record_send("recovery", [self.n])
        self.est.observe_ack("recovery", [self.n], memory_epoch=7)
        self.est.observe_ack("event-only", [], memory_epoch=7, needs_info=[self.n])
        self.assertEqual(self.est.estimate(self.n).probability, 1)

    def test_new_epoch_ack_replay_cannot_restore_explicitly_missing_item(self):
        self.est.record_send("crossing-reset", [self.i])
        self.est.observe_ack("crossing-reset", [self.i], memory_epoch=8)
        self.est.observe_missing(self.i)
        self.est.observe_ack("crossing-reset", [self.i], memory_epoch=8)
        self.assertEqual(self.est.estimate(self.i).probability, 0)

    def test_partial_ack_does_not_confirm_other_payload_items(self):
        self.est.record_send("partial", [self.n, self.i])
        self.est.observe_ack("partial", [self.n], memory_epoch=7)
        self.assertEqual(self.est.estimate(self.n).probability, 1)
        self.assertEqual(self.est.estimate(self.i).probability, 0)
        self.est.observe_ack("partial", [self.i], memory_epoch=7)
        self.assertAlmostEqual(self.est.estimate(self.i).probability, 0.95)

    def test_revoked_pending_pass_cannot_return_after_confirmation(self):
        self.est.record_pass_by("bad-pass", [self.n], normal_before=True)
        self.est.revoke_evidence("bad-pass")
        self.est.confirm_pass_by("bad-pass", normal_after=True)
        self.assertEqual(self.est.estimate(self.n).probability, 0)

    def test_declared_content_uncertainty_survives_storage_ack(self):
        self.est.record_send("uncertain-label", [self.n], content_quality={self.n.key: 0.4})
        self.est.observe_ack("uncertain-label", [self.n], memory_epoch=7)
        self.assertAlmostEqual(self.est.estimate(self.n).probability, 0.4)

    def test_invalid_contracts_raise_before_creating_false_positive_knowledge(self):
        with self.assertRaises(ValueError):
            self.est.record_pass_by("route", [self.n], normal_before="false")
        self.est.record_pass_by("route", [self.n], normal_before=True)
        with self.assertRaises(ValueError):
            self.est.confirm_pass_by("route", normal_after="false")
        self.est.record_result("task", [self.i])
        with self.assertRaises(ValueError):
            self.est.verify_task("task", success="false")
        self.assertEqual(self.est.estimate(self.i).probability, 0)
        self.assertEqual(self.est.estimate(self.n).probability, 0)

    def test_impossible_timeout_and_conflicting_message_identity(self):
        self.est.record_send("reliable", [self.n], p_delivery=1, p_ack=1)
        with self.assertRaises(ValueError):
            self.est.observe_ack_timeout("reliable")
        with self.assertRaises(ValueError):
            self.est.record_send("reliable", [self.i])
        with self.assertRaises(ValueError):
            self.est.observe_ack("reliable", [self.i], memory_epoch=7)
        self.assertEqual(self.est.estimate(self.n).probability, 0)
        with self.assertRaises(ValueError):
            EstimatorConfig(r_crop=float("nan"))


if __name__ == "__main__":
    unittest.main()
