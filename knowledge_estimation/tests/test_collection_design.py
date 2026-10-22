"""Balanced collection, split isolation, phase and train-only fitting."""

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.calibrate import fit_sources
from knowledge_estimation.measurement import read_jsonl
from knowledge_estimation.parameters import MissingMeasurement
from knowledge_estimation.simulation.collection_design import build_plan, allocations, check_allocations, validate_plan, REFERENCE_SPLITS
from knowledge_estimation.simulation.collect_designed import run, validate_collection, A_input, validate_A_input, sample_pass
from knowledge_estimation.simulation.analyze_designed import analyze
from knowledge_estimation.tests.fixtures import source_dataset


class CollectionDesignTests(unittest.TestCase):
    def test_matched_strata_fresh_geometry_and_explicit_precision(self):
        plan = build_plan()
        rows = list(allocations(plan))
        counts = check_allocations(plan, rows)
        self.assertEqual({counts[s] for s in REFERENCE_SPLITS}, {2160})
        balance = [Counter(r["stratum_id"] for r in rows if r["split"] == s) for s in REFERENCE_SPLITS]
        self.assertTrue(all(c == balance[0] for c in balance))
        self.assertEqual(len(balance[0]), 108)
        self.assertEqual(plan["precision_target"]["minimum_draws_per_marginal_group"], 352)
        self.assertEqual(plan["precision_target"]["actual_smallest_group_draws"], 360)
        self.assertEqual(plan["precision_target"]["actual_draws_per_speed"], 720)
        self.assertEqual(rows, list(allocations(plan)))
        corrupt = deepcopy(rows)
        corrupt[1] = corrupt[0]
        with self.assertRaises(ValueError):
            check_allocations(plan, corrupt)
        plan["seed"] += 1
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_plan(plan)

    def test_A_roundtrip_cannot_include_truth_or_semantic_trial_ids(self):
        plan = build_plan()
        draw = next(allocations(plan))
        config = plan["config"]
        A = json.loads(json.dumps(A_input(config, draw)))
        validate_A_input(plan, config, draw, A)
        self.assertNotIn("-h", A["trial_id"])
        A["walls"] = draw["walls"]
        with self.assertRaisesRegex(ValueError, "hidden"):
            validate_A_input(plan, config, draw, A)

    def test_sensor_phase_is_retained_and_refinement_is_aligned(self):
        plan = build_plan()
        draw = next(allocations(plan))
        frames, _ = sample_pass(plan["config"], draw)
        fine, _ = sample_pass(plan["config"], draw, 4)
        self.assertGreater(frames[0]["time"], 0.)
        self.assertAlmostEqual(frames[0]["time"], draw["frame_phase_seconds"], places=9)
        fine_times = {f["time"] for f in fine}
        self.assertTrue(all(f["time"] in fine_times for f in frames))

    def test_source_fit_can_use_train_and_cannot_use_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = source_dataset(Path(directory)/"source")
            path = root/"trials.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row["split"] = "train"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = root/"manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"]["trials.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(fit_sources(root, allow_fixture=True, fit_split="train")["fit_split"], "train")
            with self.assertRaises(MissingMeasurement):
                fit_sources(root, allow_fixture=True)
            with self.assertRaises(ValueError):
                fit_sources(root, allow_fixture=True, fit_split="test")

    def test_end_to_end_reserved_calibration_and_unsupported_OOD(self):
        # Relaxed precision is ONLY for a fast contract test, not collected data.
        plan = build_plan(seed=42, epsilon=.49, alpha=.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"collection"
            report = run(root, plan)
            self.assertTrue(report["quality_passed"])
            self.assertTrue(validate_collection(root)["passed"])
            self.assertGreater(analyze(root, root.parent/"analysis")["rows"], 0)
            self.assertFalse((root/"calibration"/"raw_predictions.jsonl").exists())
            self.assertTrue(all(all(v is None for v in r["raw"].values()) for r in read_jsonl(root/"ood_speed"/"raw_predictions.jsonl")))
            artifact = json.loads((root/"raw_parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["fit_split"], "train")
            self.assertTrue(all("/train/" in e for p in artifact["parameters"] for e in p["episodes"]))
            self.assertEqual(artifact["groups"]["object_instance"], ["physical-Q"])
            with (root/"test"/"B_labels.private.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_collection(root)
