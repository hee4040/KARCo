"""Raw retention preserves knowledge semantics, including unknown quality."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.calibrate import fit_sources, fit_scores
from knowledge_estimation.parameters import ParameterStore, MissingMeasurement
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.probabilistic.projection import (
    calibration_context, calibration_result, project, validate_model,
)
from knowledge_estimation.measurement import load_dataset
from knowledge_estimation.simulation.run import run as collect
from knowledge_estimation.simulation.repair_calibration import build_plan, run as repair
from knowledge_estimation.simulation.compare_selective import experiment_plan, run as compare
from knowledge_estimation.tests.fixtures import source_dataset, decision_dataset, item


def attach_policy(model, gates=None):
    result = deepcopy(model)
    result["application_policy"] = {"name": "validation_gated_raw_fallback_v1", "gates": gates or {k: {} for k in "NCI"}}
    result.pop("model_id")
    result["model_id"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    validate_model(result)
    return result


class SelectiveProjectionTests(unittest.TestCase):
    def test_ACK_confirms_R_without_forcing_K_or_filling_unknown_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = ParameterStore(fit_sources(source_dataset(root/"source"), allow_fixture=True), allow_fixture=True)
            data = decision_dataset(root/"data", parameters=params)
            base = fit_scores(data, allow_fixture=True, conditioning="evidence_sources_v1")
            model = attach_policy(base)
            est = KnowledgeEstimator(parameters=params)
            payload = [item(k) for k in "NCI"]
            est.record_send("m", payload, now=0., channel_condition="fixture-channel", content_condition="fixture-content")
            est.observe_ack("m", payload, memory_epoch=0, now=1.)
            snapshot = est.snapshot("Q", "S2", now=1.)
            self.assertEqual(snapshot.appearance.stored_probability, 1.)
            self.assertEqual(snapshot.appearance.probability, .5)  # measured fixture content quality, not ACK=>K=1
            result = project(snapshot, model)
            self.assertEqual(result.b_cal["b_I"], .5)
            self.assertIn("fresh_ACK", result.application_reasons["b_I"])
            # Removing content quality measurements preserves R=1 and K=None.
            artifact = params.artifact
            artifact["parameters"] = [p for p in artifact["parameters"] if p["name"] not in {"r_crop", "r_current"}]
            unknown = KnowledgeEstimator(parameters=ParameterStore(artifact, allow_fixture=True))
            unknown.record_send("m", payload, now=0., channel_condition="fixture-channel", content_condition="fixture-content")
            unknown.observe_ack("m", payload, memory_epoch=0, now=1.)
            snap = unknown.snapshot("Q", "S2", now=1.)
            self.assertEqual(snap.appearance.stored_probability, 1.)
            self.assertIsNone(snap.appearance.probability)
            value, reason = calibration_result(model, "I", None, hard_invalid=False, sources=["ack"], timestamps=[1.], now=1.)
            self.assertIsNone(value)
            self.assertEqual(reason, "unmeasured_raw_parameters")

    def test_fallback_is_raw_not_a_constant_and_calibration_can_raise_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = ParameterStore(fit_sources(source_dataset(root/"source"), allow_fixture=True), allow_fixture=True)
            base = fit_scores(decision_dataset(root/"data", parameters=params), allow_fixture=True, conditioning="evidence_sources_v1")
            model = attach_policy(base)
            for raw in (0., .2, .8, 1.):
                value, reason = calibration_result(model, "I", raw, hard_invalid=False, sources=["send"], timestamps=[1.], now=2.)
                self.assertEqual(value, raw)
                self.assertEqual(reason, "raw_no_validated_context")
            key = calibration_context(["ack"], [1.], 2.)
            gates = {k: {} for k in "NCI"}
            gates["I"][key] = {"use_calibration": True, "reason": "validation_improved", "train_rows": 2, "validation_rows": 2}
            enabled = attach_policy(base, gates)
            value, reason = calibration_result(enabled, "I", .5, hard_invalid=False, sources=["ack"], timestamps=[1.], now=2.)
            self.assertEqual(value, 1.)  # explicit fixture map raises; no blanket downward cap
            self.assertEqual(reason, "calibration_validation_improved")
            self.assertEqual(calibration_result(enabled, "I", .6, hard_invalid=False, sources=["ack"], timestamps=[1.], now=2.)[0], .6)
            self.assertEqual(calibration_result(enabled, "I", .8, hard_invalid=True, sources=["ack"], timestamps=[1.], now=2.)[0], 0.)
            with self.assertRaises(ValueError):
                calibration_result(enabled, "I", .5, hard_invalid=False, sources=["ack"])
            self.assertNotEqual(calibration_context(["ack"], [1.], 1.), key)


class SelectiveExperimentTests(unittest.TestCase):
    def test_validation_precedes_final_and_native_split_is_preserved(self):
        base = json.loads((Path(__file__).parents[1]/"simulation"/"scenario.json").read_text(encoding="utf-8"))
        for sweep in base["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
        small = build_plan(base)
        for sweep in small["config"]["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
            sweep["speeds_mps"] = [1.]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect(root/"old", base)
            repair(root/"model", old_data=root/"old", plan=small)
            plan = experiment_plan(root/"model")
            plan["base_audit_plan"]["cases"] = ["pass_only", "normal_channel", "ack_without_pass", "hidden_reset", "unknown_route"]
            for name in ("validation_sweep", "final_sweep"):
                plan[name]["speeds_mps"] = [1.]
                plan[name]["headings_degrees"] = plan[name]["headings_degrees"][:1]
            result = compare(root/"compare", model_root=root/"model", plan=plan)
            selection = json.loads((root/"compare"/"selection.json").read_text(encoding="utf-8"))
            self.assertFalse(selection["final_test_exists"])
            _, _, validation = load_dataset(root/"compare"/"validation"/"episodes", allow_simulation=True)
            _, _, final = load_dataset(root/"compare"/"final_test"/"episodes", allow_simulation=True)
            self.assertTrue(all(r["split"] == "validation" for r in validation))
            self.assertTrue(all(r["split"] == "test" for r in final))
            self.assertEqual(result["coverage"]["new"], result["coverage"]["raw"])
            self.assertGreater(result["coverage"]["new"], result["coverage"]["old"])
            self.assertEqual(result["final_episodes"], 20)
