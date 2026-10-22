"""End-to-end frozen audit, coverage accounting and privileged-state boundary."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.measurement import load_dataset
from knowledge_estimation.parameters import ParameterStore
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.simulation.run import items
from knowledge_estimation.simulation.run import run as collect
from knowledge_estimation.simulation.repair_calibration import build_plan, run as repair
from knowledge_estimation.simulation.verify_estimator import default_plan, run as verify, metrics_for, sha


class VerificationMetricsTests(unittest.TestCase):
    def test_missing_predictions_are_not_zero_error_and_comparisons_are_paired(self):
        rows = [{"K": 1, "raw": 1., "cal": None}, {"K": 0, "raw": .5, "cal": .25},
                {"K": 1, "raw": None, "cal": None}]
        result = metrics_for("case", "phase", "I", rows, .9)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["raw_known"], 2)
        self.assertEqual(result["cal_known"], 1)
        self.assertEqual(result["cal_missing"], 2)
        self.assertEqual(result["raw_brier"], .125)
        self.assertEqual(result["raw_brier_on_paired"], .25)
        self.assertEqual(result["cal_brier_on_paired"], .0625)
        self.assertEqual(result["paired_known"], 1)
        empty = metrics_for("case", "phase", "I", [{"K": 1, "raw": None, "cal": None}], .9)
        self.assertIsNone(empty["cal_brier"])


class FrozenVerificationTests(unittest.TestCase):
    def test_all_interventions_use_frozen_model_and_report_coverage(self):
        config = json.loads((Path(__file__).parents[1]/"simulation"/"scenario.json").read_text(encoding="utf-8"))
        for sweep in config["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
        plan = build_plan(config)
        for sweep in plan["config"]["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
            sweep["speeds_mps"] = [1.]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect(root/"old", config)
            repair(root/"model", old_data=root/"old", plan=plan)
            audit_plan = default_plan(root/"model")
            audit_plan["config"]["sweeps"]["test"] = {
                "offsets_m": [.45], "headings_degrees": [68.], "speeds_mps": [1.]}
            before = sha(root/"model"/"selected_model.json")
            result = verify(root/"audit", model_root=root/"model", plan=audit_plan)
            self.assertEqual(result["episodes"], 24)
            self.assertEqual(result["unique_geometry_designs"], 2)
            self.assertEqual(result["cases"], 12)
            self.assertEqual(result["logic_failures"], [])
            self.assertGreater(result["logic_checks"], 0)
            self.assertEqual(result["model_sha256"], before)
            self.assertEqual(sha(root/"model"/"selected_model.json"), before)
            self.assertGreater(result["unsupported_predictions"], 0)
            self.assertEqual(result["calibrated_predictions"]+result["unsupported_predictions"], result["item_predictions"])
            with self.assertRaisesRegex(ValueError, "allow-simulation"):
                load_dataset(root/"audit"/"episodes")
            _, _, inputs = load_dataset(root/"audit"/"episodes", allow_simulation=True)
            for row in inputs:
                self.assertEqual(row["selected_items"], list("ENCI") if row["decision_id"].endswith("before_send") else [])
            with (root/"audit"/"predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            hidden = [r for r in rows if r["case"] == "hidden_reset" and r["phase"] == "after_reset"]
            self.assertTrue(all(r["K"] == "0" for r in hidden))
            for row in hidden:
                prior = next(r for r in rows if r["episode_id"] == row["episode_id"] and r["kind"] == row["kind"] and r["phase"] == "after_channel")
                self.assertEqual(row["raw"], prior["raw"])
                self.assertEqual(row["cal"], prior["cal"])
            unsupported_ack = [r for r in rows if r["case"] == "ack_without_pass" and r["phase"] == "after_channel" and r["kind"] in "NI"]
            self.assertTrue(all(r["cal"] == "" and "unseen calibration evidence" in r["calibration_reason"] for r in unsupported_ack))
            unknown = [r for r in rows if r["case"] == "unknown_route" and r["phase"] == "after_pass" and r["kind"] in "NI"]
            self.assertTrue(all(r["raw"] == "" and r["cal"] == "" for r in unknown))
            with (root/"audit"/"reliability_by_case_phase.csv").open(encoding="utf-8-sig", newline="") as stream:
                bins = list(csv.DictReader(stream))
            self.assertEqual(sum(int(b["samples"]) for b in bins if b["level"] == "cal"), result["calibrated_predictions"])
            # Also test previously UNSEEN late ACKs, not only duplicate replay.
            parameters = ParameterStore.load(root/"model"/"simulation_params.yaml", allow_simulation=True)
            payload = list(items(config).values())
            reset_est = KnowledgeEstimator(parameters=parameters)
            reset_est.record_send("pending", payload, now=0., channel_condition="los-channel", content_condition="ideal-content")
            reset_est.observe_status(1, now=1.)
            reset_est.observe_ack("pending", payload, memory_epoch=0, received_at=.1, now=1.1)
            self.assertTrue(all(v == 0. for v in reset_est.snapshot("Q", "S2", now=1.1).as_policy_input().values()))
            version_est = KnowledgeEstimator(parameters=parameters)
            version_est.record_send("pending", payload, now=0., channel_condition="los-channel", content_condition="ideal-content")
            for old in payload:
                version_est.register(replace(old, version=old.version+1, observed_at=1.), now=1.)
            version_est.observe_ack("pending", payload, memory_epoch=0, received_at=.1, now=1.1)
            self.assertTrue(all(v == 0. for v in version_est.snapshot("Q", "S2", now=1.1).as_policy_input().values()))
