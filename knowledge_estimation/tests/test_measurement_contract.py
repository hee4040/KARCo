"""Meaningful regressions for evidence validity and fail-closed measurements."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.calibrate import fit_sources, fit_scores
from knowledge_estimation.collect import replay
from knowledge_estimation.evaluate import evaluate, binomial_interval
from knowledge_estimation.measurement import label_before, load_dataset
from knowledge_estimation.parameters import MissingMeasurement, ParameterStore
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.probabilistic.projection import project
from knowledge_estimation.protocol import EstimatorAdapter, Observation, item_to_dict, item_from_dict
from knowledge_estimation.types import EstimatorConfig
from knowledge_estimation.tests.fixtures import item, snapshots, source_dataset, decision_dataset
from knowledge_estimation.tests.smoke import FIXTURE_EVALUATION_PROTOCOL, run
from knowledge_estimation.collect import MeasurementRecorder
from knowledge_estimation.annotate import attach
from knowledge_estimation.tests.fixtures import metadata, add_evidence


class MeasuredCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = source_dataset(Path(cls.temp.name) / "sources")
        cls.artifact = fit_sources(root, allow_fixture=True)
        cls.parameters = ParameterStore(cls.artifact, allow_fixture=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def estimator(self, *, independent=False):
        config = EstimatorConfig(independent_sources=independent,
            independence_evidence_id="fixture-only-explicit-independence-assumption" if independent else None)
        return KnowledgeEstimator(parameters=self.parameters, config=config)

    def stored(self, est, content, *, msg="m", now=0.):
        est.record_send(msg, [content], channel_condition="fixture-channel", content_condition="fixture-content", now=now)
        est.observe_ack(msg, [content], memory_epoch=0, now=now)

    def register_all(self, est):
        for kind in "NCI":
            est.register(item(kind), now=0.)

    def test_no_numeric_probability_defaults_or_fabricated_metadata(self):
        est = KnowledgeEstimator()
        n, appearance = item("N"), item("I")
        est.record_send("m", [n, appearance])
        self.assertIsNone(est.estimate(n).probability)
        est.observe_ack("m", [n, appearance], memory_epoch=0)
        self.assertEqual(est.estimate(n).probability, 1.)
        self.assertEqual(est.estimate(appearance).stored_probability, 1.)
        self.assertIsNone(est.estimate(appearance).probability)
        self.assertEqual(est.estimate(appearance).semantics, "unmeasured")
        with self.assertRaises(ValueError):
            replace(item("C"), ttl_seconds=None)
        with self.assertRaises(ValueError):
            replace(item("N"), frame_id=None)
        with self.assertRaises(ValueError):
            replace(item("N"), observed_at=None)

    def test_unmeasured_channel_timeout_is_rejected(self):
        est = KnowledgeEstimator()
        est.record_send("m", [item("N")])
        with self.assertRaises(MissingMeasurement):
            est.observe_ack_timeout("m", now=1.)

    def test_fixture_cannot_be_loaded_as_measurement(self):
        with self.assertRaises(MissingMeasurement):
            ParameterStore(self.artifact)
        store = ParameterStore.load("knowledge_estimation/params.yaml")
        self.assertEqual(store.origin, "unmeasured")
        with self.assertRaises(MissingMeasurement):
            store.value("p_acq", {"receiver_id": "B", "kind": "I", "site_id": "S2", "route_id": "route-S2", "condition_id": "unknown"})

    def test_no_pseudocount_and_conditional_ack_denominator(self):
        context = {"receiver_id": "B", "kind": "", "site_id": "", "route_id": "", "condition_id": "fixture-channel"}
        self.assertEqual(self.parameters.value("p_recv", context), 3/4)
        self.assertEqual(self.parameters.value("p_ack", context), 2/3)
        changed = deepcopy(self.artifact)
        next(e for e in changed["parameters"] if "value" in e)["value"] += .01
        with self.assertRaises(ValueError):
            ParameterStore(changed, allow_fixture=True)
        before = self.parameters.fingerprint
        returned = self.parameters.entry("p_recv", context)
        returned["value"] = 0.
        self.assertEqual(self.parameters.value("p_recv", context), 3/4)
        self.assertEqual(self.parameters.fingerprint, before)

    def test_crop_ack_does_not_assert_content_is_valid(self):
        est = self.estimator()
        self.stored(est, item("I"))
        belief = est.estimate(item("I"))
        self.assertEqual(belief.stored_probability, 1.)
        self.assertEqual(belief.probability, .5)
        self.assertTrue(belief.evidence[0].parameter_ids)

    def test_measured_delay_and_timeout_replace_one_attempt_prior(self):
        est = self.estimator()
        n = item("N")
        est.record_send("m", [n], channel_condition="fixture-channel", now=0.)
        self.assertEqual(est.estimate(n).probability, 0.)
        est.observe_delivery_progress("m", now=.3)
        self.assertEqual(est.estimate(n).probability, .25)
        with self.assertRaises(ValueError):
            est.observe_ack_timeout("m", now=.9)
        est.observe_ack_timeout("m", now=1.)
        self.assertAlmostEqual(est.estimate(n).probability, .5)
        est.record_send("m", [n], attempt_id=2, channel_condition="fixture-channel", now=2.)
        self.assertAlmostEqual(est.estimate(n).probability, .5)
        est.observe_ack_timeout("m", attempt_id=2, now=3.)
        self.assertAlmostEqual(est.estimate(n).probability, .5)  # No implicit independence.

    def test_independent_retry_union_needs_explicit_assumption(self):
        with self.assertRaises(ValueError):
            EstimatorConfig(independent_sources=True)
        est = self.estimator(independent=True)
        n = item("N")
        for attempt, time in ((1, 0.), (2, 2.)):
            est.record_send("m", [n], channel_condition="fixture-channel", attempt_id=attempt, now=time)
            est.observe_ack_timeout("m", attempt_id=attempt, now=time+1)
        self.assertAlmostEqual(est.estimate(n).probability, .75)

    def test_empirical_cdf_has_no_unmeasured_tail(self):
        context = {"receiver_id": "B", "kind": "", "site_id": "", "route_id": "", "condition_id": "fixture-task"}
        self.assertEqual(self.parameters.cdf("F_task", context, .9), .5)
        with self.assertRaises(MissingMeasurement):
            self.parameters.cdf("F_task", context, 2.)

    def test_one_pass_progress_is_replaced_and_normal_gate_checked(self):
        est = self.estimator()
        est.register(item("N"))
        i = item("I")
        est.record_pass_by("p", [i], route_id="route-S2", site_id="S2", condition_id="fixture-pass",
                           normal_before=True, elapsed=.5)
        self.assertEqual(est.estimate(i).probability, 0.)
        est.confirm_pass_by("p", normal_after=True)
        self.assertEqual(est.estimate(i).probability, .25)
        est.advance_pass("p", elapsed=.8, now=1.)
        self.assertEqual(est.estimate(i).probability, .5)
        est.advance_pass("p", elapsed=.8, now=1.)
        self.assertEqual(est.estimate(i).probability, .5)
        self.assertEqual(len(est.estimate(i).evidence), 1)

    def test_repeated_passes_with_common_occlusion_do_not_accumulate(self):
        est = self.estimator(independent=True)
        est.register(item("N"))
        for visit in ("p1", "p2"):
            est.record_pass_by(visit, [item("I")], route_id="route-S2", site_id="S2", condition_id="fixture-pass",
                normal_before=True, complete=True, dependency_group="same-obstruction")
            est.confirm_pass_by(visit, normal_after=True)
        self.assertEqual(est.estimate(item("I")).probability, .5)

    def test_unmeasured_route_stays_unknown_without_global_fallback(self):
        est = self.estimator()
        est.register(item("N"))
        est.record_pass_by("p", [item("I")], route_id="unmeasured-route", site_id="S2", condition_id="fixture-pass",
                           normal_before=True, complete=True)
        est.confirm_pass_by("p", normal_after=True)
        self.assertIsNone(est.estimate(item("I")).probability)

    def test_layout_change_between_pass_and_confirmation_excludes_the_opportunity(self):
        est = self.estimator()
        est.register(item("N"))
        est.record_pass_by("p", [item("I")], route_id="route-S2", site_id="S2", condition_id="fixture-pass",
                           normal_before=True, complete=True)
        est.register(item("N", version=2))
        est.confirm_pass_by("p", normal_after=True)
        self.assertEqual(est.estimate(item("I")).probability, 0.)

    def test_old_c_expiry_is_not_renewed_by_new_observation_or_late_ack(self):
        est = self.estimator()
        self.register_all(est)
        old = item("C")
        self.stored(est, old)
        self.assertEqual(est.estimate(old).probability, .5)
        fresh = replace(old, observed_at=4.)
        est.register(fresh, now=4.)
        belief = est.snapshot("Q", "S2", now=7.)
        self.assertEqual(belief.current.probability, 0.)
        self.assertEqual(belief.current.stored_probability, 1.)
        self.assertTrue(belief.current.hard_invalid)
        self.stored(est, fresh, msg="new", now=7.)
        self.assertEqual(est.estimate(fresh).probability, .5)
        late = self.estimator()
        late.record_send("m", [old], content_condition="fixture-content")
        late.observe_ack("m", [old], memory_epoch=0, now=8.)
        self.assertEqual(late.estimate(old).probability, 0.)

    def test_epoch_version_and_evidence_revocation_fence_old_updates(self):
        est = self.estimator()
        self.stored(est, item("N"))
        new = item("N", version=2)
        est.register(new)
        est.observe_ack("m", [item("N")], memory_epoch=0)
        self.assertEqual(est.estimate(new).probability, 0.)
        self.stored(est, new, msg="new")
        est.observe_status(1)
        est.observe_ack("new", [new], memory_epoch=0)
        self.assertEqual(est.estimate(new).probability, 0.)
        est.record_result("e/v2/a1", [new], memory_epoch=1)
        est.revoke_evidence("e/v2/a1")
        est.verify_task("e/v2/a1", success=True)
        self.assertEqual(est.estimate(new).probability, 0.)

    def test_failed_detection_is_not_missing_and_physical_failure_keeps_memory(self):
        for reason in ("not_found", "low_quality", "ambiguous_identity", "not_visible", "grasp_fail", "nav_fail", "tag_lost"):
            est = self.estimator()
            self.stored(est, item("I"))
            est.observe_missing(item("I"), reason=reason)
            self.assertEqual(est.estimate(item("I")).probability, .5)
            est.observe_missing(item("I"), reason="missing")
            self.assertEqual(est.estimate(item("I")).probability, 0.)

    def test_result_needs_external_verification_and_measured_q(self):
        est = self.estimator()
        est.register(item("N"))
        est.record_result("e/v1/a1", [item("N"), item("I")], learning_condition="fixture-content")
        self.assertEqual(est.estimate(item("I")).probability, 0.)
        est.verify_task("e/v1/a1", success=True)
        self.assertEqual(est.estimate(item("N")).probability, 1.)
        self.assertEqual(est.estimate(item("I")).probability, .5)
        unmeasured = KnowledgeEstimator()
        unmeasured.record_result("e/v1/a1", [item("I")])
        unmeasured.verify_task("e/v1/a1", success=True)
        self.assertIsNone(unmeasured.estimate(item("I")).probability)
        with self.assertRaisesRegex(ValueError, "approved layout"):
            est.record_result("e/v2/a1", [item("N", version=2)], learning_condition="fixture-content")
        self.assertEqual(est.estimate(item("N")).probability, 1.)

    def test_invalid_content_preserves_r_and_same_crop_cannot_repair_itself(self):
        est = self.estimator()
        content = item("I")
        self.stored(est, content)
        est.observe_missing(content, reason="invalid_content")
        self.assertEqual(est.estimate(content).stored_probability, 1.)
        self.assertEqual(est.estimate(content).probability, 0.)
        self.stored(est, content, msg="same-crop")
        self.assertEqual(est.estimate(content).probability, 0.)
        self.assertTrue(est.estimate(content).hard_invalid)
        fresh = replace(content, content_id="new-fixture-crop", value="new-fixture-template")
        self.stored(est, fresh, msg="recapture")
        self.assertEqual(est.estimate(fresh).probability, .5)

    def test_diagnosed_bad_frame_invalidates_pose_but_keeps_appearance_and_storage(self):
        est = self.estimator()
        self.register_all(est)
        for kind in "NCI":
            self.stored(est, item(kind), msg=kind)
        est.observe_frame_invalid("tag:S2", reason="verified-transform-mismatch")
        self.assertEqual(est.estimate(item("N")).probability, 0.)
        self.assertEqual(est.estimate(item("C")).probability, 0.)
        self.assertEqual(est.estimate(item("C")).stored_probability, 1.)
        self.assertEqual(est.estimate(item("I")).probability, .5)
        self.stored(est, item("N"), msg="bad-frame-again")
        self.assertEqual(est.estimate(item("N")).probability, 0.)
        fixed = replace(item("N"), frame_id="tag:S2/calibration-v2", observed_at=1.)
        self.stored(est, fixed, msg="corrected-frame", now=1.)
        self.assertEqual(est.estimate(fixed).probability, 1.)

    def test_strict_runtime_input_and_item_metadata_roundtrip(self):
        adapter = EstimatorAdapter()
        with self.assertRaises(ValueError):
            adapter.handle(Observation("version", 0., {"item": item_to_dict(item("N")), "actual_cache": {}}))
        with self.assertRaises(ValueError):
            adapter.handle(Observation("send", 0., {"msg_id": "m", "items": [], "p_delivery": .9}))
        content = replace(item("C"), value=(1., 2., 3.), uncertainty=(.01, .01, .02), association_confidence=.5)
        self.assertEqual(item_from_dict(json.loads(json.dumps(item_to_dict(content)))), content)


class OfflineContractTests(unittest.TestCase):
    def pending_capture(self, root):
        writer = MeasurementRecorder(root, protocol_id="fixture-contract-v1", episode_id="calibration-episode", origin="fixture")
        add_evidence(writer)
        for kind in "NCI":
            writer.observe(Observation("version", 0., {"item": item_to_dict(item(kind))}))
        before, _ = snapshots(0.)
        writer.record_decision(decision_id="pending", meta=metadata("calibration"), time=0., object_ref="Q", site_id="S2",
            event_id="e", task_version=1, attempt_id=1, selected_items=list("ENCI"), required_items=list("ENCI"),
            before_snapshot=before)
        writer.observe(Observation("send", .1, {"msg_id": "m", "items": [item_to_dict(item(k)) for k in "NCI"]}))
        writer.observe(Observation("timeout", 1., {"msg_id": "m"}))
        after, _ = snapshots(1., stored=True)
        for row in after["items"].values():
            row["physical_object_id"], row["valid_content"] = None, None
        writer.record_after_receipt("pending", snapshot=after)
        return writer.finalize()

    def test_unannotated_capture_and_unmeasured_timeout_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"raw"
            manifest = self.pending_capture(root)
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["annotations_complete"])
            self.assertIn("timeout", (root/"observations.jsonl").read_text())
            self.assertIn("no-ACK update requires", (root/"measurement_gaps.jsonl").read_text())
            with self.assertRaisesRegex(ValueError, "pending"):
                fit_scores(root, allow_fixture=True)

    def test_offline_annotation_cannot_change_captured_state_or_a_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/"raw", Path(directory)/"annotated"
            self.pending_capture(root)
            annotations = Path(directory)/"annotations.jsonl"
            rows = []
            for phase, time, stored in (("before", 0., False), ("after", 1., True)):
                _, world = snapshots(time, stored=stored)
                rows.append({"decision_id": "pending", "phase": phase, "world": world,
                    "item_assessments": {k: {"physical_object_id": "fixture-physical-Q", "valid_content": True}
                                         for k in ("NCI" if stored else "")}})
            annotations.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            manifest = attach(root, annotations, output, allow_fixture=True)
            self.assertTrue(manifest["annotations_complete"])
            self.assertEqual((root/"inputs.jsonl").read_bytes(), (output/"inputs.jsonl").read_bytes())
            self.assertEqual((root/"observations.jsonl").read_bytes(), (output/"observations.jsonl").read_bytes())
            fit_scores(output, allow_fixture=True)
            rows[1]["item_assessments"]["N"]["version"] = 99
            annotations.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            with self.assertRaisesRegex(ValueError, "rewrite"):
                attach(root, annotations, Path(directory)/"invalid", allow_fixture=True)
            self.assertFalse((Path(directory)/"invalid").exists())

    def test_csv_to_evaluation_pipeline_fixture_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory)/"smoke")
            self.assertTrue(result["passed"])
            self.assertTrue(result["not_robot_performance"])
            root = Path(directory)/"smoke"
            model = json.loads((root/"fixture_calibration.json").read_text())
            report = json.loads((root/"fixture_contract_report.json").read_text())
            self.assertEqual(report["origin"], "fixture")
            self.assertEqual(report["decisions"], 3)
            self.assertEqual(report["high_confidence_failure_causes"]["N"]["unannounced_epoch_change"], 1)
            with self.assertRaises(ValueError):
                load_dataset(root/"replayed_fixture")
            with self.assertRaises(ValueError):
                evaluate(root/"replayed_fixture", model, {}, allow_fixture=True)

    def test_before_snapshot_time_and_versions_determine_k_not_after(self):
        snapshot, world = snapshots(2., stored=True)
        self.assertEqual(label_before(snapshot, world, 2.)["I"]["K"], 1)
        world["versions"]["I"] = 2
        self.assertEqual(label_before(snapshot, world, 2.)["I"]["K"], 0)
        self.assertEqual(label_before(snapshot, world, 2.)["I"]["R"], 1)
        with self.assertRaises(ValueError):
            label_before(snapshot, world, 1.)

    def test_small_sample_zero_failures_does_not_mean_zero_risk(self):
        interval = binomial_interval(0, 10, .95)
        self.assertAlmostEqual(interval[1], 1-.025**.1)
        self.assertGreater(interval[1], .3)
        self.assertIsNone(binomial_interval(0, 0, .95))

    def test_independence_of_splits_and_integrity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = decision_dataset(Path(directory)/"data")
            manifest = json.loads((root/"manifest.json").read_text())
            (root/"trials.csv").write_text("damaged")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dataset(root, allow_fixture=True)
        from knowledge_estimation.measurement import check_split_groups
        first = {**metadata_for_test("calibration"), "split": "calibration"}
        second = {**metadata_for_test("test"), "split": "test", "session_id": first["session_id"]}
        with self.assertRaisesRegex(ValueError, "leaks"):
            check_split_groups([first, second])

    def test_calibration_does_not_fit_final_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_dataset(Path(directory)/"source")
            params = ParameterStore(fit_sources(source, allow_fixture=True), allow_fixture=True)
            raw = decision_dataset(Path(directory)/"raw", parameters=params)
            model = fit_scores(raw, allow_fixture=True)
            # Change only held-out truth; refresh its dataset checksum to model
            # an intentional, internally valid external-annotation correction.
            labels = (raw/"labels.jsonl").read_text().splitlines()
            changed = []
            for line in labels:
                row = json.loads(line)
                if row["decision_id"].startswith("test-"):
                    row["world"]["versions"]["I"] = 2
                changed.append(json.dumps(row))
            (raw/"labels.jsonl").write_text("\n".join(changed)+"\n")
            import hashlib
            manifest = json.loads((raw/"manifest.json").read_text())
            manifest["sha256"]["labels.jsonl"] = hashlib.sha256((raw/"labels.jsonl").read_bytes()).hexdigest()
            (raw/"manifest.json").write_text(json.dumps(manifest))
            other = fit_scores(raw, allow_fixture=True)
            self.assertEqual(model["maps"], other["maps"])
            self.assertEqual(model["estimator_fingerprint"], other["estimator_fingerprint"])


def metadata_for_test(split):
    return {"episode_id": split+"-ep", "session_id": split+"-session", "object_instance": split+"-obj", "route_group_id": split+"-route"}


if __name__ == "__main__":
    unittest.main()
