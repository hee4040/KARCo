"""Use the teammate's exact local Git package as an explicit policy fixture."""

from dataclasses import asdict
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from knowledge_estimation.calibrate import fit_scores, fit_sources
from knowledge_estimation.integration.rule_based import RuleBasedSelector
from knowledge_estimation.parameters import MissingMeasurement, ParameterStore
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.tests.fixtures import source_dataset, decision_dataset, item


class ActualRuleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        repo = Path(__file__).resolve().parents[2]
        cls.source = root / "rule-source"
        completed = subprocess.run([sys.executable, "-m", "knowledge_estimation.integration.prepare_rule_snapshot",
                                   "--output", str(cls.source)], cwd=repo, capture_output=True, text=True)
        if completed.returncode:
            cls.temp.cleanup()
            raise unittest.SkipTest("local teammate ref is unavailable; merge/fetch it before the actual-package contract test")
        sys.path.insert(0, str(cls.source))
        cls.schemas = importlib.import_module("rule_based_comm.schemas")
        cls.config = importlib.import_module("rule_based_comm.config").PolicyConfig()
        # These are teammate example settings used ONLY as explicit test input.
        # They are not copied to params.yaml or described as measured values.
        cls.adapter = importlib.import_module("rule_based_comm.policy_adapter").RulePolicyAdapter(
            importlib.import_module("rule_based_comm.rule_policy").RuleBasedCommunicationPolicy(cls.config))
        cls.manifest = {"origin": "fixture", "config": asdict(cls.config),
                        "evidence_ids": ["teammate-config-fixture"], "input_sha256": {
                            "config": hashlib.sha256(json.dumps(asdict(cls.config), sort_keys=True).encode()).hexdigest()}}
        sources = source_dataset(root / "source-measurement-fixture")
        cls.parameters = ParameterStore(fit_sources(sources, allow_fixture=True), allow_fixture=True)
        data = decision_dataset(root / "decision-fixture", parameters=cls.parameters)
        cls.model = fit_scores(data, allow_fixture=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def snapshot(self):
        est = KnowledgeEstimator(parameters=self.parameters)
        for kind in "NCI":
            est.register(item(kind))
        return est.snapshot("Q", "S2")

    def selector(self):
        return RuleBasedSelector(adapter=self.adapter, model=self.model, policy_manifest=self.manifest, allow_fixture=True)

    def test_actual_policy_consumes_named_calibrated_values(self):
        result = self.selector().select({"event_id": "event-1", "object_id": "Q", "spot_id": "S2",
                                         "target_confirmed": True}, self.snapshot())
        self.assertEqual(result["action"], "SEND")
        self.assertEqual(result["selected_items"], ["E", "N"])
        self.assertEqual(result["belief_level"], "cal")
        self.assertEqual(result["observation"]["b_N"], 0.)

    def test_common_no_comm_and_reobserve_are_preserved(self):
        selector = self.selector()
        first = selector.select({"event_id": "event-1", "already_processing": True}, self.snapshot())
        self.assertEqual(first["action"], "NO_COMM")
        self.assertIsNone(first["action_id"])
        second = selector.select({"event_id": "event-1", "target_confirmed": False}, self.snapshot())
        self.assertEqual(second["action"], "REOBSERVE")

    def test_no_default_policy_values_or_implicit_fixture_use(self):
        with self.assertRaises(MissingMeasurement):
            RuleBasedSelector(adapter=self.adapter, model=self.model, allow_fixture=True)
        with self.assertRaises(ValueError):
            RuleBasedSelector(adapter=self.adapter, model=self.model, policy_manifest=self.manifest)
        changed = {**self.manifest, "config": {}}
        with self.assertRaises(ValueError):
            RuleBasedSelector(adapter=self.adapter, model=self.model, policy_manifest=changed, allow_fixture=True)

    def test_privileged_and_mismatched_object_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.selector().select({"event_id": "e", "actual_B_cache": {}}, self.snapshot())
        with self.assertRaises(ValueError):
            self.selector().select({"event_id": "e", "object_id": "P"}, self.snapshot())

    def test_calibration_requires_same_parameter_fingerprint(self):
        wrong = KnowledgeEstimator()
        for kind in "NCI":
            wrong.register(item(kind))
        with self.assertRaises(ValueError):
            self.selector().select({"event_id": "e"}, wrong.snapshot("Q", "S2"))
        from copy import deepcopy
        tampered = deepcopy(self.model)
        tampered["maps"]["N"]["blocks"][0]["value"] = .12345
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            RuleBasedSelector(adapter=self.adapter, model=tampered, policy_manifest=self.manifest, allow_fixture=True)


if __name__ == "__main__":
    unittest.main()
