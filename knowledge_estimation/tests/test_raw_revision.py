"""Regression for idempotent evidence vs monotonic expiry, without calibration."""

from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.calibrate import fit_sources
from knowledge_estimation.parameters import ParameterStore
from knowledge_estimation.probabilistic import KnowledgeEstimator, RawKnowledgeEstimatorV2
from knowledge_estimation.simulation.compare_raw import metrics
from knowledge_estimation.tests.fixtures import source_dataset, item


class RawRevisionTests(unittest.TestCase):
    def test_duplicate_paths_expire_C_without_erasing_storage_or_renewing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters = ParameterStore(fit_sources(source_dataset(Path(directory)/"source"), allow_fixture=True), allow_fixture=True)
            for event in ("ack", "send", "timeout", "result", "verification"):
                for factory, expected in ((KnowledgeEstimator, .5), (RawKnowledgeEstimatorV2, 0.)):
                    with self.subTest(event=event, estimator=factory.__name__):
                        est, current = factory(parameters=parameters), item("C")
                        send = dict(channel_condition="fixture-channel", content_condition="fixture-content")
                        est.record_send("m", [current], now=.1, **send)
                        est.observe_ack("m", [current], memory_epoch=0, received_at=.5, now=1.)
                        # ACK guarantees storage; measured content validity is .5.
                        self.assertEqual(est.estimate(current).stored_probability, 1.)
                        self.assertEqual(est.estimate(current).probability, .5)
                        est.record_result("task", now=1.)
                        est.verify_task("task", success=True, now=1.)
                        if event == "ack":
                            est.observe_ack("m", [current], memory_epoch=0, received_at=.5, now=8.)
                        elif event == "send":
                            est.record_send("m", [current], now=8., **send)
                        elif event == "timeout":
                            est.observe_ack_timeout("m", now=8.)
                        elif event == "result":
                            est.record_result("task", now=8.)
                        else:
                            est.verify_task("task", success=True, now=8.)
                        belief = est.estimate(current)
                        self.assertEqual(belief.probability, expected)
                        self.assertEqual(belief.stored_probability, 1.)
                        self.assertEqual(len(belief.evidence), 1)
                        self.assertEqual(belief.evidence[0].timestamp, 1.)
                        if factory is RawKnowledgeEstimatorV2:
                            self.assertTrue(belief.hard_invalid)
                            with self.assertRaises(ValueError):
                                est.observe_ack_timeout("m", now=7.)

    def test_revision_does_not_invent_quality_when_measurement_missing(self):
        est, current = RawKnowledgeEstimatorV2(), item("C")
        est.record_send("m", [current], now=.1)
        est.observe_ack("m", [current], memory_epoch=0, now=1.)
        self.assertIsNone(est.estimate(current).probability)
        self.assertEqual(est.estimate(current).stored_probability, 1.)
        self.assertNotEqual(est.fingerprint, KnowledgeEstimator().fingerprint)

    def test_metrics_report_worsening_and_missing_on_identical_pairs(self):
        rows = [dict(case="a", phase="b", kind="N", K=1, old=1., new=.5),
                dict(case="a", phase="b", kind="N", K=0, old=None, new=None)]
        result = metrics(rows)[0]
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["paired"], 1)
        self.assertEqual(result["change"], "worsened")
        self.assertEqual(result["delta"], .25)
        self.assertEqual(result["new_missing"], 1)
