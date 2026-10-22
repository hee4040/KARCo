"""Causal context calibration, deployment parity and validation/test separation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.calibrate import fit_sources, fit_scores
from knowledge_estimation.evaluate import evaluate
from knowledge_estimation.measurement import read_jsonl
from knowledge_estimation.parameters import ParameterStore, MissingMeasurement
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.probabilistic.projection import source_key, project, calibrated_value
from knowledge_estimation.simulation.repair_calibration import build_plan, run as repair, choose
from knowledge_estimation.simulation.run import run as collect_simulation
from knowledge_estimation.tests.fixtures import source_dataset, decision_dataset, item
from knowledge_estimation.tests.smoke import FIXTURE_EVALUATION_PROTOCOL


class ContextCalibrationTests(unittest.TestCase):
    def test_runtime_offline_parity_and_unknown_context_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = ParameterStore(fit_sources(source_dataset(root/"source"), allow_fixture=True), allow_fixture=True)
            data = decision_dataset(root/"data", parameters=params)
            model = fit_scores(data, allow_fixture=True, conditioning="evidence_sources_v1")
            report = evaluate(data, model, FIXTURE_EVALUATION_PROTOCOL, allow_fixture=True)
            est = KnowledgeEstimator(parameters=params)
            for k in "NCI":
                est.register(item(k))
            est.record_send("m", [item(k) for k in "NCI"], now=.1,
                            channel_condition="fixture-channel", content_condition="fixture-content")
            est.observe_ack("m", [item(k) for k in "NCI"], memory_epoch=0, received_at=.5, now=1.)
            runtime = project(est.snapshot("Q", "S2", now=2.), model)
            offline = next(r for r in report["paired_decision_rows"] if r["decision_id"] == "test-d1")
            self.assertEqual(runtime.b_cal, {"b_"+k: offline["cal"][k] for k in "NCI"})
            with self.assertRaisesRegex(MissingMeasurement, "unseen calibration evidence"):
                calibrated_value(model, "N", .5, hard_invalid=False, sources=["new_source"])
            self.assertEqual(calibrated_value(model, "C", .8, hard_invalid=True, sources=["new_source"]), 0.)
            self.assertEqual(source_key(["pass_by", "ack", "ack"]), '["ack","pass_by"]')
            self.assertIsNone(calibrated_value(model, "N", None, hard_invalid=False, sources=[]))

    def test_fit_is_invariant_to_final_labels_and_hidden_condition_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = ParameterStore(fit_sources(source_dataset(root/"source"), allow_fixture=True), allow_fixture=True)
            data = decision_dataset(root/"data", parameters=params)
            before = fit_scores(data, allow_fixture=True, conditioning="evidence_sources_v1")
            labels = list(read_jsonl(data/"labels.jsonl"))
            for row in labels:
                if row["decision_id"].startswith("test-"):
                    row["world"]["versions"]["I"] = 2
            (data/"labels.jsonl").write_text("".join(json.dumps(r)+"\n" for r in labels), encoding="utf-8")
            manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
            manifest["sha256"]["labels.jsonl"] = hashlib.sha256((data/"labels.jsonl").read_bytes()).hexdigest()
            (data/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            after = fit_scores(data, allow_fixture=True, conditioning="evidence_sources_v1")
            self.assertEqual(before["context_maps"], after["context_maps"])
            # The context contains source types only; no phase, reset or hidden wall.
            keys = {key for group in after["context_maps"].values() for key in group}
            self.assertEqual(keys, {"[]", '["ack"]'})

    def test_selection_is_validation_only(self):
        stats = {"pooled": {"primary": .2, "secondary": .1},
                 "evidence_sources_v1": {"primary": .1, "secondary": .2}}
        self.assertEqual(choose(stats), "evidence_sources_v1")
        stats["pooled"]["primary"] = .05
        self.assertEqual(choose(stats), "pooled")

    def test_fresh_experiment_freezes_selection_before_test(self):
        base = json.loads((Path(__file__).parents[1]/"simulation"/"scenario.json").read_text(encoding="utf-8"))
        for sweep in base["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
        plan = build_plan(base)
        for sweep in plan["config"]["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
            sweep["speeds_mps"] = [1.]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect_simulation(root/"old", base)
            result = repair(root/"new", old_data=root/"old", plan=plan)
            selection = json.loads((root/"new"/"selection.json").read_text(encoding="utf-8"))
            self.assertFalse(selection["final_test_exists_when_selecting"])
            self.assertEqual(selection["selected"], choose(result["validation"]))
            selected = json.loads((root/"new"/"selected_model.json").read_text(encoding="utf-8"))
            self.assertEqual(selected["model_id"], selection["selected_model_id"])
            self.assertEqual(result["episodes_total"], 56)
            self.assertEqual(result["counts"]["validation"]["episodes"], 16)
            self.assertEqual(result["counts"]["test"]["episodes"], 16)
            self.assertEqual(set(result["test"]), {"pooled", result["selected"]})
            sources = selected.get("context_maps", {})
            self.assertFalse(any("reset" in key or "phase" in key for group in sources.values() for key in group))
