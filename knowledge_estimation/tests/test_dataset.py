"""Dataset-specific leakage, label correctness, reproducibility and split tests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.learning.dataset import FEATURE_NAMES, extract_features
from knowledge_estimation.learning.dataset.audit import audit_dataset, read_jsonl
from knowledge_estimation.learning.dataset.collection import CollectionConfig, collect, run_episode
from knowledge_estimation.simulator.engine import Simulator


def feature_sample(sim, target="Q"):
    record = sim.record_sample(target)
    view = record["a_view"]
    return extract_features(sim.export_observations()[:view["history_end"]], view), record


class CausalInputTests(unittest.TestCase):
    def test_hidden_occlusion_changes_label_without_changing_inputs_before_feedback(self):
        left, right = Simulator("pass_clear", seed=7), Simulator("pass_occluded", seed=7)
        left.step(6)
        right.step(6)
        a, lrecord = feature_sample(left)
        b, rrecord = feature_sample(right)
        self.assertEqual(left.export_observations(), right.export_observations())
        self.assertEqual(a, b)
        self.assertEqual(lrecord["a_view"], rrecord["a_view"])
        self.assertEqual(lrecord["labels"]["I"], 1)
        self.assertEqual(rrecord["labels"]["I"], 0)

    def test_unannounced_reset_changes_only_labels_at_the_sample_boundary(self):
        sim = Simulator("first_event", config={"p_delivery": 1, "p_ack": 1})
        sim.step(25)
        before, first = feature_sample(sim, "P")
        self.assertEqual(first["labels"]["N"], 1)
        sim.reset_receiver_memory(notify=False)
        after, second = feature_sample(sim, "P")
        self.assertEqual(before, after)
        self.assertEqual(first["a_view"], second["a_view"])
        self.assertEqual(second["labels"], {"N": 0, "C": 0, "I": 0})

    def test_hidden_pose_and_appearance_replacements_do_not_reveal_versions(self):
        sim = Simulator()
        view = sim.a_view("Q")
        sim.move_object("Q")
        sim.change_appearance("Q")
        self.assertEqual(sim.a_view("Q"), view)
        self.assertEqual(sim.label_audit("Q")["items"]["I"]["actual_version"], 2)
        self.assertEqual(sim.a_view("Q")["known_versions"]["I"], 1)

    def test_samples_before_send_and_after_send_have_different_causal_prefixes(self):
        sim = Simulator()
        sim.send_info("N", target="Q")
        record = sim.export_records()[0]
        history = sim.export_observations()
        self.assertEqual(record["a_view"]["history_end"], 6)
        features = extract_features(history[:6], record["a_view"])
        self.assertEqual(features["N_send_count"], 0)
        self.assertEqual(feature_sample(sim)[0]["N_send_count"], 1)
        with self.assertRaises(ValueError):
            extract_features(history, record["a_view"])

    def test_later_ack_does_not_change_a_frozen_sample(self):
        sim = Simulator(config={"p_delivery": 1, "p_ack": 1})
        sim.send_info("N", target="Q")
        features, record = feature_sample(sim)
        self.assertEqual(features["N_ack_count"], 0)
        sim.step(1)
        history = sim.export_observations()
        self.assertEqual(extract_features(history[:record["a_view"]["history_end"]], record["a_view"]), features)
        self.assertEqual(feature_sample(sim)[0]["N_ack_count"], 1)
        self.assertEqual(record["labels"]["N"], 0)

    def test_history_rejects_future_and_out_of_order_timestamps(self):
        sim = Simulator()
        view, history = sim.a_view("Q"), sim.export_observations()
        corrupt = deepcopy(history)
        corrupt[-1]["timestamp"] = 1
        with self.assertRaises(ValueError):
            extract_features(corrupt, view)
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))

    def test_full_journal_does_not_drop_early_events_at_gui_deque_limit(self):
        sim = Simulator()
        for _ in range(1001):
            sim._observe("status", memory_epoch=0, boot_id="unchanged")
        self.assertEqual(len(sim.observations), 1000)
        self.assertEqual(len(sim.export_observations()), 1007)
        self.assertEqual(sim.a_view("Q")["history_end"], 1007)

    def test_varied_scene_changes_both_planned_geometry_and_actual_pass(self):
        sim = Simulator(scene={"q_x": 6.3, "q_y": 3.7, "pass_x": 4.2, "b_camera_range": 1.4})
        sim.step(6)
        features, record = feature_sample(sim)
        self.assertEqual(features["planned_in_range"], 0)
        self.assertEqual(record["labels"]["I"], 0)
        self.assertEqual(record["a_view"]["baseline"]["b_I"], 0)
        self.assertEqual(sim.snapshot()["objects"][1]["normal_x"], 6.3)


class CollectionTests(unittest.TestCase):
    def test_collection_replays_and_families_never_cross_splits(self):
        options = CollectionConfig(families=5, seconds=12, sample_interval=1, seed=17)
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/"left", Path(directory)/"right"
            manifest = collect(left, options)
            replay = collect(right, options)
            self.assertEqual(manifest, replay)
            self.assertEqual(manifest["sha256"], replay["sha256"])
            report = audit_dataset(left)
            self.assertTrue(report["passed"])
            self.assertEqual(report["episodes"], 20)
            family_splits = {}
            for entry in manifest["episodes"]:
                family_splits.setdefault(entry["family_id"], set()).add(entry["split"])
            self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))
            self.assertEqual({entry["split"] for entry in manifest["episodes"]}, {"train", "validation", "test"})
            for row in read_jsonl(left/"inputs"/"train.jsonl"):
                self.assertEqual(set(row["features"]), set(FEATURE_NAMES))
                self.assertNotIn("labels", row)
                self.assertNotIn("audit", row)
                self.assertNotIn("scenario", row)
            with self.assertRaises(FileExistsError):
                collect(left, options)

    def test_auditor_detects_tampering_and_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"data"
            collect(root, CollectionConfig(families=5, seconds=1))
            labels = root/"labels"/"test.jsonl"
            labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit_dataset(root)
            (root/"manifest.json").write_text(json.dumps({"complete": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit_dataset(root)

    def test_invalid_collection_options_fail_before_writing(self):
        for kwargs in ({"families": 4}, {"families": True}, {"seconds": float("nan")},
                       {"seconds": .03}, {"sample_interval": 0}, {"seconds": 1, "sample_interval": 2}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CollectionConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
