"""Checks derived from the user's adaptive communication HTML examples.

These tests exercise A's public observations only. In particular, simulated B
ground truth and hidden occlusion are never passed to the estimator.
"""

from dataclasses import FrozenInstanceError
import unittest

from knowledge_estimation.estimator import KnowledgeEstimator
from knowledge_estimation.types import (
    EstimatorConfig, Information, KnowledgeItem, KnowledgeKey,
)


def normal(object_id="P", site_id="S1", version=1):
    return KnowledgeItem(KnowledgeKey(Information.NORMAL, object_id, site_id), version)


def current(object_id="P", version=1):
    return KnowledgeItem(KnowledgeKey(Information.CURRENT, object_id), version)


def appearance(object_id="P", version=1):
    return KnowledgeItem(KnowledgeKey(Information.APPEARANCE, object_id), version)


class HTMLScenarioTests(unittest.TestCase):
    def setUp(self):
        self.estimator = KnowledgeEstimator(memory_epoch=7)
        self.n = normal()
        self.c = current()
        self.i = appearance()
        self.time = 0

    def now(self):
        self.time += 1
        return float(self.time)

    def assert_belief(self, item, expected, estimator=None):
        actual = (estimator or self.estimator).estimate(item).probability
        self.assertAlmostEqual(actual, expected, places=12)

    def send_ack(self, msg_id, items, estimator=None):
        est = estimator or self.estimator
        est.record_send(msg_id, items, now=self.now())
        est.observe_ack(msg_id, items, memory_epoch=est.memory_epoch, now=self.now())

    def verify_learned(self, task_id, items, estimator=None, **kwargs):
        est = estimator or self.estimator
        est.record_result(task_id, items, now=self.now(), **kwargs)
        est.verify_task(task_id, success=True, now=self.now())

    def test_first_task_requires_ack_and_external_verification(self):
        est = self.estimator
        est.record_send("task-e1", [self.n], now=self.now())
        self.assert_belief(self.n, 0)
        est.observe_ack("task-e1", [self.n], memory_epoch=7, now=self.now())
        self.assert_belief(self.n, 1)
        self.assert_belief(self.i, 0)

        est.record_result("e1/attempt1", [self.i], now=self.now())
        self.assert_belief(self.i, 0)  # B's own ok is not A's verification.
        est.verify_task("e1/attempt1", success=True, now=self.now())
        self.assert_belief(self.i, 0.9)
        est.verify_task("e1/attempt1", success=True, now=self.now())
        self.assert_belief(self.i, 0.9)  # Duplicate verification is not new learning.

    def test_ack_loss_bayesian_update_and_retry(self):
        est = self.estimator
        est.record_send("e1", [self.n], p_delivery=0.9, p_ack=0.9, now=self.now())
        est.observe_ack_timeout("e1", now=self.now())
        self.assert_belief(self.n, (0.9 * 0.1) / (1 - 0.9 * 0.9))
        est.observe_ack_timeout("e1", now=self.now())
        self.assert_belief(self.n, 0.47368421052631576)
        est.record_send("e1", [self.n], attempt_id=2, now=self.now())
        est.observe_ack("e1", [self.n], memory_epoch=7, now=self.now())
        self.assert_belief(self.n, 1)
        est.observe_ack_timeout("e1", attempt_id=2, now=self.now())
        self.assert_belief(self.n, 1)

    def test_retry_timeout_updates_receipt_without_duplicating_crop_quality(self):
        est = self.estimator
        posterior = 0.9 * 0.1 / (1 - 0.9 * 0.9)
        est.record_send("crop", [self.i], now=self.now())
        est.observe_ack_timeout("crop", now=self.now())
        est.record_send("crop", [self.i], attempt_id=2, now=self.now())
        est.observe_ack_timeout("crop", attempt_id=2, now=self.now())
        self.assert_belief(self.i, (1 - (1 - posterior) ** 2) * 0.95)
        est.observe_ack("crop", [self.i], memory_epoch=7, now=self.now())
        self.assert_belief(self.i, 0.95)

    def test_reuse_and_epoch_reset_distinguish_persistent_cache_from_boot(self):
        est = self.estimator
        self.send_ack("original-n", [self.n])
        self.verify_learned("original-task", [self.i])
        self.assert_belief(self.n, 1)
        self.assert_belief(self.i, 0.9)
        est.observe_status(7, boot_id="boot-new", now=self.now())
        self.assert_belief(self.n, 1)
        self.assert_belief(self.i, 0.9)

        # An unreceived STATUS is deliberately absent from A's observations.
        decision = est.snapshot("P", "S1", now=self.now())
        est.record_send("event-only", [], now=self.now())
        est.observe_ack(
            "event-only", [], memory_epoch=8, needs_info=[self.n], now=self.now(),
        )
        self.assertEqual(est.memory_epoch, 8)
        self.assert_belief(self.n, 0)
        self.assert_belief(self.i, 0)
        self.assertEqual(decision.as_policy_input(), {"b_N": 1, "b_C": 0, "b_I": 0.9})

        # Late old-epoch confirmations must neither resurrect nor clear new data.
        self.send_ack("new-n", [self.n])
        est.observe_ack("original-n", [self.n], memory_epoch=7, now=self.now())
        self.assertEqual(est.memory_epoch, 8)
        self.assert_belief(self.n, 1)
        self.assert_belief(self.i, 0)

    def test_empty_ack_does_not_confirm_unrelated_appearance(self):
        self.send_ack("n", [self.n])
        self.send_ack("event-only", [])
        self.assert_belief(self.n, 1)
        self.assert_belief(self.i, 0)

    def test_ack_with_new_epoch_confirms_explicitly_stored_new_cache_items(self):
        est = self.estimator
        self.verify_learned("old-task", [self.i])
        est.record_send("normal-before-reset-known", [self.n], now=self.now())
        est.observe_ack(
            "normal-before-reset-known", [self.n], memory_epoch=8, now=self.now(),
        )
        self.assertEqual(est.memory_epoch, 8)
        self.assert_belief(self.n, 1)
        self.assert_belief(self.i, 0)

    def test_delayed_old_epoch_feedback_does_not_clear_fresh_items(self):
        est = self.estimator
        est.observe_status(8, now=self.now())
        self.send_ack("fresh", [self.n, self.c, self.i])
        est.observe_missing(self.i, memory_epoch=7, now=self.now())
        est.record_result(
            "old-search", status="not_found", current_item=self.c,
            memory_epoch=7, now=self.now(),
        )
        est.record_result("old-learning", [appearance(version=2)], memory_epoch=7, now=self.now())
        self.assert_belief(self.n, 1)
        self.assert_belief(self.c, 1)
        self.assert_belief(self.i, 0.95)
        self.assertEqual(est.memory_epoch, 8)

    def test_omitted_processing_time_reuses_latest_time(self):
        est = self.estimator
        est.record_send("normal", [self.n], now=10)
        est.observe_ack("normal", [self.n], memory_epoch=7)
        self.assert_belief(self.n, 1)
        self.assertEqual(est.snapshot("P", "S1").timestamp, 10)
        with self.assertRaises(ValueError):
            est.snapshot("P", "S1", now=9)

    def test_pass_by_normal_pose_waits_for_before_and_after_confirmation(self):
        est = self.estimator
        n = normal("Q", "S2")
        est.record_pass_by("visit", [n], normal_before=True, now=self.now())
        self.assert_belief(n, 0)
        est.confirm_pass_by("visit", normal_after=True, now=self.now())
        self.assert_belief(n, 0.72)
        est.confirm_pass_by("visit", normal_after=True, now=self.now())
        self.assert_belief(n, 0.72)

    def test_pass_by_abnormal_or_unverified_before_is_not_normal_knowledge(self):
        for normal_before, normal_after in ((True, False), (False, True), (False, False)):
            with self.subTest(before=normal_before, after=normal_after):
                est = KnowledgeEstimator()
                n = normal("Q", "S2")
                est.record_pass_by("visit", [n], normal_before=normal_before, now=1)
                est.confirm_pass_by("visit", normal_after=normal_after, now=2)
                self.assert_belief(n, 0, est)

    def test_joint_snapshot_and_independent_handling_evidence(self):
        est = self.estimator
        n, i = normal("Q", "S2"), appearance("Q")
        est.record_pass_by("visit", [n, i], normal_before=True, now=self.now())
        self.assert_belief(n, 0)
        self.assert_belief(i, 0.72)
        est.confirm_pass_by("visit", normal_after=True, now=self.now())
        self.assert_belief(n, 0.72)
        self.assert_belief(i, 0.72)
        self.send_ack("normal", [n])
        est.record_result("e3/attempt1", [i], now=self.now())
        self.assert_belief(i, 0.72)
        est.verify_task("e3/attempt1", success=True, now=self.now())
        self.assert_belief(n, 1)
        self.assert_belief(i, 1 - 0.28 * 0.1)

    def test_hidden_occlusion_has_identical_estimate_until_missing_feedback(self):
        clear, occluded = KnowledgeEstimator(), KnowledgeEstimator()
        n, i = normal("Q", "S2"), appearance("Q")
        for est in (clear, occluded):
            est.record_pass_by("route", [n, i], normal_before=True, now=1)
            est.confirm_pass_by("route", normal_after=True, now=2)
        self.assertEqual(
            clear.snapshot("Q", "S2", now=3).as_policy_input(),
            occluded.snapshot("Q", "S2", now=3).as_policy_input(),
        )
        self.assert_belief(i, 0.72, occluded)
        occluded.observe_missing(i, now=4)
        self.assert_belief(i, 0, occluded)
        self.assert_belief(n, 0.72, occluded)
        occluded.record_send("crop", [i], now=5)
        occluded.observe_ack("crop", [i], memory_epoch=0, now=6)
        self.assert_belief(i, 0.95, occluded)
        self.assert_belief(i, 0.72, clear)

    def test_crop_and_verified_handling_combine_to_exact_point_995(self):
        self.send_ack("crop", [self.i])
        self.assert_belief(self.i, 0.95)
        self.verify_learned("e4/attempt1", [self.i])
        self.assert_belief(self.i, 0.995)

    def test_same_crop_version_in_new_message_is_not_independent_content(self):
        self.send_ack("crop-first", [self.i])
        self.send_ack("crop-again", [self.i])
        self.assert_belief(self.i, 0.95)
        self.estimator.record_send("crop-third", [self.i], now=self.now())
        self.estimator.observe_ack_timeout("crop-third", now=self.now())
        self.assert_belief(self.i, 0.95)

    def test_bad_task_source_revocation_preserves_other_evidence(self):
        est = self.estimator
        other_i = appearance("Q")
        est.record_pass_by("route", [self.i], now=self.now())
        self.verify_learned("wrong-task", [self.i, other_i])
        self.assert_belief(self.i, 0.972)
        self.assert_belief(other_i, 0.9)
        est.revoke_evidence("wrong-task", now=self.now())
        self.assert_belief(self.i, 0.72)
        self.assert_belief(other_i, 0)
        # Replaying the same verified task cannot resurrect a revoked source.
        est.verify_task("wrong-task", success=True, now=self.now())
        self.assert_belief(other_i, 0)

    def test_failed_verification_adds_no_learning(self):
        est = self.estimator
        est.record_result("wrong-object", [self.i], now=self.now())
        est.verify_task("wrong-object", success=False, now=self.now())
        self.assert_belief(self.i, 0)
        est.verify_task("wrong-object", success=True, now=self.now())
        self.assert_belief(self.i, 0)

    def test_not_found_invalidates_only_current_pose_then_versioned_update(self):
        est = self.estimator
        self.send_ack("task-v1", [self.n, self.c, self.i])
        before = est.snapshot("P", "S1", now=self.now())
        # An actual move unknown to A creates no estimator event.
        self.assert_belief(self.c, 1)
        est.record_result("search/attempt1", status="not_found", current_item=self.c, now=self.now())
        self.assert_belief(self.n, 1)
        self.assert_belief(self.c, 0)
        self.assert_belief(self.i, 0.95)
        c2 = current(version=2)
        est.register(c2, now=self.now())
        self.assert_belief(c2, 0)
        self.send_ack("pose-update-v2", [c2])
        self.assert_belief(c2, 1)
        self.assert_belief(self.c, 0)
        self.assertEqual(before.as_policy_input(), {"b_N": 1, "b_C": 1, "b_I": 0.95})

    def test_new_layout_and_appearance_versions_replace_old_evidence(self):
        est = self.estimator
        self.send_ack("old-items", [self.n, self.c, self.i])
        n2, i2 = normal(version=2), appearance(version=2)
        est.register(n2, now=self.now())
        est.register(i2, now=self.now())
        for item in (self.n, n2, self.i, i2):
            self.assert_belief(item, 0)
        self.assert_belief(self.c, 1)
        est.observe_ack("old-items", [self.n, self.i], memory_epoch=7, now=self.now())
        self.assert_belief(n2, 0)
        self.assert_belief(i2, 0)
        self.send_ack("new-items", [n2, i2])
        self.assert_belief(n2, 1)
        self.assert_belief(i2, 0.95)
        self.assertFalse(est.register(self.i, now=self.now()))
        self.assert_belief(i2, 0.95)

    def test_missing_feedback_blocks_late_old_ack_and_pending_learning(self):
        est = self.estimator
        est.record_send("old-crop", [self.i], now=self.now())
        est.record_result("old-task", [self.i], now=self.now())
        est.observe_missing(self.i, now=self.now())
        est.observe_ack("old-crop", [self.i], memory_epoch=7, now=self.now())
        est.verify_task("old-task", success=True, now=self.now())
        self.assert_belief(self.i, 0)
        self.send_ack("replacement-crop", [self.i])
        self.assert_belief(self.i, 0.95)

    def test_decision_snapshot_is_deeply_immutable_and_keeps_old_evidence(self):
        est = self.estimator
        self.verify_learned("learned", [self.i])
        snapshot = est.snapshot("P", "S1", now=self.now())
        exported = snapshot.as_policy_input()
        exported["b_I"] = 0
        with self.assertRaises(FrozenInstanceError):
            snapshot.appearance.probability = 0
        with self.assertRaises(FrozenInstanceError):
            snapshot.appearance.evidence[0].p_stored = 0
        est.observe_missing(self.i, now=self.now())
        self.assert_belief(self.i, 0)
        self.assertEqual(snapshot.appearance.probability, 0.9)
        self.assertEqual(snapshot.appearance.evidence[0].p_valid, 0.9)

    def test_dependency_groups_use_max_and_independent_groups_use_noisy_or(self):
        est = self.estimator
        est.record_pass_by("view-a", [self.i], dependency_group="same-source", now=self.now())
        self.verify_learned("handled", [self.i], dependency_group="same-source")
        self.assert_belief(self.i, 0.9)  # No false .972 from correlated evidence.
        est.record_pass_by("view-b", [self.i], dependency_group="different-source", now=self.now())
        self.assert_belief(self.i, 0.972)

    def test_conservative_config_never_assumes_source_independence(self):
        est = KnowledgeEstimator(config=EstimatorConfig(independent_sources=False))
        est.record_pass_by("view-a", [self.i], now=1)
        est.record_result("handled", [self.i], now=2)
        est.verify_task("handled", success=True, now=3)
        self.assert_belief(self.i, 0.9, est)
        est.record_send("crop", [self.i], now=4)
        est.observe_ack("crop", [self.i], memory_epoch=0, now=5)
        self.assert_belief(self.i, 0.95, est)


if __name__ == "__main__":
    unittest.main()
