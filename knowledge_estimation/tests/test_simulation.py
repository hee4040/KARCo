"""Geometry, information boundary and real simulation collection checks."""

import json
from math import pi
from pathlib import Path
import tempfile
import unittest

from knowledge_estimation.measurement import load_dataset, decision_pairs, read_jsonl
from knowledge_estimation.parameters import ParameterStore, MissingMeasurement
from knowledge_estimation.simulation.run import run
from knowledge_estimation.simulation.world import blocked, visible, run_pass


CONFIG = Path(__file__).parents[1]/"simulation"/"scenario.json"


class GeometryTests(unittest.TestCase):
    def test_gate_uses_frame_intervals_without_accumulated_roundoff(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["route_x"] = [-.5, .5]
        result = run_pass(config, 0., 90., False)
        self.assertEqual(result.acquired_at, {"N": .3, "I": .6})

    def test_sector_range_and_occlusion(self):
        args = dict(z_min=.6, z_max=3., fov=pi/2, walls=[])
        self.assertTrue(visible((0, 0, 0), (2, 0), **args))
        self.assertFalse(visible((0, 0, pi), (2, 0), **args))
        self.assertFalse(visible((0, 0, 0), (4, 0), **args))
        self.assertFalse(visible((0, 0, 0), (.5, 0), **args))
        self.assertTrue(visible((0, 0, 0), (3, 1), **args))  # axial depth, not Euclidean radius
        self.assertFalse(visible((0, 0, 0), (2, 0), **{**args, "walls": [(1, -1, 1.5, 1)]}))
        self.assertTrue(blocked((0, 0), (2, 0), [(1, 0, 1.5, 1)]))
        self.assertFalse(blocked((0, 0), (2, 0), [(1, 1, 1.5, 2)]))

    def test_dwell_controls_storage_without_success_draws(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        observed = run_pass(config, 1., 90., False)
        self.assertEqual(set(observed.acquired_at), set("NI"))
        config["storage_dwell_seconds"] = {"N": 100., "I": 100.}
        self.assertEqual(run_pass(config, 1., 90., False).acquired_at, {})
        self.assertEqual(observed, run_pass(json.loads(CONFIG.read_text(encoding="utf-8")), 1., 90., False))

    def test_timestep_sensitivity_of_acquisition(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        coarse = run_pass(config, 1., 90., False)
        config["dt_seconds"] /= 2
        fine = run_pass(config, 1., 90., False)
        self.assertEqual(set(coarse.acquired_at), set(fine.acquired_at))
        for k in coarse.acquired_at:
            self.assertLessEqual(abs(coarse.acquired_at[k]-fine.acquired_at[k]), 2*config["dt_seconds"]+1e-8)


class SimulationPipelineTests(unittest.TestCase):
    def test_collection_calibration_heldout_truth_and_origin_boundary(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        # A smaller declared sweep keeps this integration test inexpensive.
        for sweep in config["sweeps"].values():
            sweep["offsets_m"] = sweep["offsets_m"][:2]
            sweep["headings_degrees"] = sweep["headings_degrees"][:2]
        config["evaluation_protocol"]["bootstrap_repetitions"] = 5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"simulation"
            summary = run(root, config)
            report = json.loads((root/"report.json").read_text(encoding="utf-8"))
            self.assertIsNone(report["metrics"]["raw"]["N"]["brier_bootstrap"]["interval"])
            self.assertEqual(summary["origin"], "simulation")
            self.assertEqual(summary["test_episodes"], 16)
            with self.assertRaisesRegex(ValueError, "allow-simulation"):
                load_dataset(root/"replayed")
            with self.assertRaises(MissingMeasurement):
                ParameterStore.load(root/"simulation_params.yaml")
            store = ParameterStore.load(root/"simulation_params.yaml", allow_simulation=True)
            source_ids = set(store.artifact["groups"]["episode_id"])
            paired = list(decision_pairs(root/"replayed", split="test"))
            self.assertFalse(source_ids.intersection(r[0]["episode_id"] for r in paired))
            # Hidden reset changes privileged labels but not A's belief or history.
            journals = list(read_jsonl(root/"replayed"/"observations.jsonl"))
            self.assertFalse(any("snapshot" in str(r.get("event", {}).get("data", {})) for r in journals))
            for row, before, _, truth in paired:
                if row["decision_id"].endswith("after_possible_reset") and before["snapshot"]["memory_epoch"] == 1:
                    self.assertTrue(all(truth[k]["K"] == 0 for k in "NCI"))
                    prior = next(p[0] for p in paired if p[0]["episode_id"] == row["episode_id"] and p[0]["decision_id"].endswith("after_channel"))
                    self.assertEqual(row["b_raw"], prior["b_raw"])
            traces = list(read_jsonl(root/"source_runs"/"world_trace.jsonl"))
            for entry in store.artifact["parameters"]:
                if entry["name"] == "p_acq":
                    k = entry["context"]["kind"]
                    self.assertEqual(entry["successes"], sum(k in t["acquired_at"] for t in traces))
                    self.assertEqual(entry["trials"], len(traces))
            self.assertTrue(all(v is not None for level in summary["brier"].values() for v in level.values()))
            analysis = json.loads((root/"condition_report.json").read_text(encoding="utf-8"))
            acquisition = next(a for a in analysis["acquisition"] if a["stage"] == "test" and a["factor"] == "all" and a["kind"] == "N")
            self.assertEqual(acquisition["unique_pass_designs"], 8)  # reset variants not double-counted
            group = next(e for e in analysis["estimation"] if e["phase"] == "after_pass" and e["factor"] == "all" and e["kind"] == "N")
            self.assertEqual(group["samples"], 16)

    def test_planned_speed_has_separate_source_context_and_fixed_horizon(self):
        from knowledge_estimation.simulation.run import capture
        from knowledge_estimation.calibrate import fit_sources
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        for sweep in config["sweeps"].values():
            sweep["offsets_m"] = [1.]
            sweep["headings_degrees"] = [90.]
            sweep["speeds_mps"] = [.5, 2.]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"sources"
            capture(root, config, ("source",), sources_only=True)
            params = fit_sources(root, allow_simulation=True)
            entries = [e for e in params["parameters"] if e["name"] == "p_acq"]
            self.assertEqual({e["context"]["condition_id"] for e in entries},
                             {"unreported-view:speed=0.5mps", "unreported-view:speed=2mps"})
            cdfs = [e for e in params["parameters"] if e["name"] == "F_pass"]
            for e in cdfs:
                self.assertEqual(e["observation_horizon_seconds"], 12. if "0.5mps" in e["context"]["condition_id"] else 3.)
