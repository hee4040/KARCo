"""Headless integration checks for native movement, sensors and messages."""

from math import hypot
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from knowledge_estimation.simulator.bridge import EstimatorAdapter, Observation
from knowledge_estimation.simulator.engine import SCENARIOS, Simulator


DT = 0.05


def advance_until(simulator, predicate, *, seconds=120):
    snapshot = simulator.snapshot()
    if predicate(snapshot):
        return snapshot
    for _ in range(int(seconds / DT)):
        snapshot = simulator.step(DT)
        if predicate(snapshot):
            return snapshot
    raise AssertionError(f"condition not reached within {seconds}s: {snapshot['event']}")


def complete(simulator, *, seconds=120):
    return advance_until(simulator, lambda snap: snap["event"]["state"] == "complete", seconds=seconds)


class NativeSimulatorTests(unittest.TestCase):
    def test_fixed_clock_accumulates_dt_and_moves_in_world_units(self):
        sim = Simulator("first_event", seed=7)
        initial = sim.snapshot()
        unchanged = sim.step(0)
        self.assertEqual(initial, unchanged)
        self.assertEqual(sim.step(.024)["time"], 0)
        moved = sim.step(.026)
        self.assertAlmostEqual(moved["time"], .05)
        a0, a1 = initial["robots"]["A"], moved["robots"]["A"]
        self.assertAlmostEqual(hypot(a1["x"] - a0["x"], a1["y"] - a0["y"]), 1.15 * .05)
        self.assertEqual(initial["robots"]["B"]["x"], moved["robots"]["B"]["x"])
        for value in (-.01, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(dt=value):
                with self.assertRaises(ValueError):
                    sim.step(value)

    def test_same_seed_and_fixed_ticks_are_independent_of_dt_chunking(self):
        large = Simulator("pass_occluded", seed=7)
        small = Simulator("pass_occluded", seed=7)
        large.step(30)
        for _ in range(1800):
            small.step(1 / 60)
        self.assertEqual(large.snapshot(), small.snapshot())
        self.assertEqual(large.export_records(), small.export_records())
        replay = Simulator("pass_occluded", seed=7)
        replay.step(30)
        self.assertEqual(large.snapshot(), replay.snapshot())

    def test_all_four_seeded_scenarios_finish_with_actual_normal_objects(self):
        self.assertEqual(set(SCENARIOS), {"first_event", "ack_loss", "pass_clear", "pass_occluded"})
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                sim = Simulator(name, seed=7)
                snapshot = complete(sim)
                self.assertLess(snapshot["time"], 120)
                target = snapshot["event"]["target"]
                obj = next(item for item in snapshot["objects"] if item["id"] == target)
                self.assertLess(hypot(obj["x"] - obj["normal_x"], obj["y"] - obj["normal_y"]), .12)
                self.assertEqual(snapshot["beliefs"][target]["b_N"], 1)
                self.assertTrue(any(event["kind"] == "verification" for event in sim.observations))
                self.assertTrue(sim.export_records())

    def test_ack_loss_has_real_receipt_timeout_and_same_message_retry(self):
        sim = Simulator("ack_loss", seed=7)
        receipt = advance_until(
            sim, lambda snap: snap["receiver"]["knowledge"]["P"]["N"] == 1
            and snap["beliefs"]["P"]["b_N"] == 0,
        )
        self.assertGreater(receipt["time"], 0)
        timed_out = advance_until(
            sim, lambda snap: 0 < snap["beliefs"]["P"]["b_N"] < 1,
        )
        self.assertEqual(timed_out["receiver"]["knowledge"]["P"]["N"], 1)
        self.assertAlmostEqual(timed_out["beliefs"]["P"]["b_N"], .9 * .1 / (1 - .9 * .9))
        result = complete(sim)
        self.assertEqual(result["beliefs"]["P"]["b_N"], 1)
        self.assertGreaterEqual(result["stats"]["retries"], 1)
        self.assertGreaterEqual(result["stats"]["packets_lost"], 1)
        sends = [event["data"] for event in sim.observations if event["kind"] == "send"]
        self.assertTrue(any(item["attempt_id"] >= 2 for item in sends))
        self.assertEqual(len({item["msg_id"] for item in sends}), 1)

    def test_hidden_occlusion_is_not_a_knowledge_estimator_input(self):
        clear = Simulator("pass_clear", seed=7)
        occluded = Simulator("pass_occluded", seed=7)
        saw_false_positive = False
        for _ in range(int(60 / DT)):
            left, right = clear.step(DT), occluded.step(DT)
            if any(event["kind"] == "missing" for event in occluded.observations):
                break
            self.assertEqual(left["beliefs"]["Q"], right["beliefs"]["Q"])
            if right["beliefs"]["Q"]["b_I"] > 0:
                saw_false_positive = True
                self.assertAlmostEqual(right["beliefs"]["Q"]["b_I"], .72)
                self.assertEqual(left["receiver"]["knowledge"]["Q"]["I"], 1)
                self.assertEqual(right["receiver"]["knowledge"]["Q"]["I"], 0)
        else:
            self.fail("occluded B never requested missing knowledge")
        self.assertTrue(saw_false_positive)
        self.assertEqual(right["beliefs"]["Q"]["b_I"], 0)
        self.assertEqual(right["beliefs"]["Q"]["b_N"], 1)
        self.assertGreaterEqual(right["stats"]["requests"], 1)
        recovered = complete(occluded)
        self.assertEqual(recovered["receiver"]["knowledge"]["Q"]["I"], 1)
        self.assertAlmostEqual(recovered["beliefs"]["Q"]["b_I"], .995)
        for observation in occluded.observations:
            self.assertTrue({"truth", "actual_cache", "occluded", "occlusion"}.isdisjoint(observation["data"]))

    def test_unknown_memory_reset_waits_for_epoch_bearing_ack(self):
        sim = Simulator("first_event", seed=7)
        old = complete(sim)
        reset = sim.reset_receiver_memory(notify=False)
        self.assertEqual(reset["receiver"]["memory_epoch"], old["receiver"]["memory_epoch"] + 1)
        self.assertEqual(reset["receiver"]["known_epoch"], old["receiver"]["known_epoch"])
        self.assertEqual(reset["beliefs"]["P"], old["beliefs"]["P"])
        self.assertEqual(reset["receiver"]["knowledge"]["P"], {"N": 0, "C": 0, "I": 0})
        sim.step(.5)
        self.assertEqual(sim.snapshot()["beliefs"]["P"], old["beliefs"]["P"])
        sim.send_info(("N",))
        fresh = advance_until(sim, lambda snap: snap["receiver"]["known_epoch"] == snap["receiver"]["memory_epoch"])
        self.assertEqual(fresh["beliefs"]["P"]["b_N"], 1)
        self.assertEqual(fresh["beliefs"]["P"]["b_I"], 0)

    def test_status_reset_propagates_after_network_delivery(self):
        sim = Simulator("first_event", seed=7)
        old = complete(sim)
        reset = sim.reset_receiver_memory(notify=True)
        self.assertEqual(reset["receiver"]["known_epoch"], old["receiver"]["known_epoch"])
        self.assertEqual(reset["beliefs"]["P"]["b_N"], 1)
        fresh = advance_until(sim, lambda snap: snap["receiver"]["known_epoch"] == snap["receiver"]["memory_epoch"])
        self.assertGreater(fresh["time"], old["time"])
        self.assertEqual(fresh["beliefs"]["P"], {"b_N": 0, "b_C": 0, "b_I": 0})

    def test_physical_target_move_does_not_reveal_itself_before_a_observes(self):
        sim = Simulator("pass_clear", seed=7)
        before = advance_until(
            sim, lambda snap: snap["event"]["target"] == "Q"
            and snap["beliefs"]["Q"]["b_C"] == 1
            and snap["robots"]["A"]["x"] > 9,
        )
        observations_before = len(sim.observations)
        moved = sim.move_target()
        self.assertEqual(moved["beliefs"]["Q"], before["beliefs"]["Q"])
        self.assertEqual(moved["receiver"]["knowledge"]["Q"]["C"], 0)
        self.assertEqual(len(sim.observations), observations_before)
        physical = next(item for item in moved["objects"] if item["id"] == "Q")
        # The move is physical truth; A's version stays at its last observation.
        self.assertGreater(physical["version"], sim.export_records()[-1]["versions"]["C"])

    def test_learning_attempt_does_not_disclose_actual_retention_success(self):
        sim = Simulator("first_event", seed=7, config={"q_learn": .1})
        result = complete(sim)
        self.assertEqual(result["receiver"]["knowledge"]["P"]["I"], 0)
        self.assertAlmostEqual(result["beliefs"]["P"]["b_I"], .1)

    def test_not_found_reobserves_and_recovers_with_a_distinct_task_attempt(self):
        sim = Simulator("pass_clear", seed=7)
        advance_until(
            sim, lambda snap: snap["event"]["target"] == "Q"
            and snap["beliefs"]["Q"]["b_C"] == 1
            and snap["robots"]["A"]["x"] > 9,
        )
        sim.move_target()
        sim.move_target()  # Move beyond the old pose's matching radius.
        missing = advance_until(
            sim, lambda snap: any(event["kind"] == "result" and event["data"].get("status") == "not_found"
                                  for event in sim.observations),
        )
        self.assertEqual(missing["beliefs"]["Q"]["b_C"], 0)
        self.assertEqual(missing["beliefs"]["Q"]["b_N"], 1)
        self.assertAlmostEqual(missing["beliefs"]["Q"]["b_I"], .72)
        self.assertEqual(complete(sim)["event"]["target"], "Q")
        results = [event["data"] for event in sim.observations if event["kind"] == "result"]
        failed = next(event for event in results if event.get("status") == "not_found")
        self.assertTrue(any(event.get("status") == "ok" and event["task_id"] != failed["task_id"]
                            and event["task_id"].split("/")[0] == failed["task_id"].split("/")[0]
                            for event in results))

    def test_old_epoch_result_is_not_scheduled_for_verification(self):
        sim = Simulator("first_event", seed=7, config={"p_delivery": 1, "p_ack": 1})
        pending = advance_until(
            sim, lambda snap: any(packet["label"].startswith("RESULT") and packet["progress"] < 1
                                  for packet in snap["network"]),
        )
        reset = sim.reset_receiver_memory(notify=False)
        # A future middleware callback can deliver STATUS before the older,
        # delayed RESULT. Exercise that public observation boundary directly.
        sim.adapter.handle(Observation("status", pending["time"], {
            "memory_epoch": reset["receiver"]["memory_epoch"],
        }))
        after = sim.step(2)
        self.assertEqual(after["receiver"]["known_epoch"], reset["receiver"]["memory_epoch"])
        self.assertEqual(after["beliefs"]["P"]["b_I"], 0)
        self.assertFalse(any(event["kind"] == "verification" for event in sim.observations))

    def test_restored_object_has_no_falsely_valid_old_current_pose_label(self):
        sim = Simulator("pass_clear", seed=7)
        snapshot = complete(sim)
        obj = next(item for item in snapshot["objects"] if item["id"] == "Q")
        if snapshot["receiver"]["knowledge"]["Q"]["C"]:
            # Inspect cache geometry only to audit the evaluation label; it never
            # becomes an A-side observation or a learned feature.
            cached_pose = sim._cache["Q"]["C"]["pose"]
            self.assertLess(hypot(cached_pose[0] - obj["x"], cached_pose[1] - obj["y"]), .12)

    def test_decision_records_are_frozen_before_ack_and_exported_by_copy(self):
        sim = Simulator("first_event", seed=7)
        complete(sim)
        records = sim.export_records()
        first = records[0]
        self.assertEqual(first["features"]["b_N"], 0)
        self.assertEqual(first["labels"]["N"], 0)
        self.assertEqual(sim.snapshot()["beliefs"]["P"]["b_N"], 1)
        first["features"]["b_N"] = -1
        self.assertEqual(sim.export_records()[0]["features"]["b_N"], 0)
        self.assertEqual(set(sim.export_records()[0]["features"]), {"b_N", "b_C", "b_I"})

    def test_reset_replays_same_seed_from_initial_physical_state(self):
        sim = Simulator("pass_occluded", seed=7)
        initial = sim.snapshot()
        sim.step(30)
        self.assertTrue(sim.export_records())
        self.assertEqual(sim.reset(seed=7), initial)
        self.assertEqual(sim.export_records(), [])

    def test_headless_cli_exports_valid_json_without_opening_a_window(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "episode.json"
            environment = dict(os.environ, PYTHONIOENCODING="utf-8")
            process = subprocess.run(
                [sys.executable, "-m", "knowledge_estimation.simulator", "--headless",
                 "--scenario", "pass_occluded", "--seed", "7", "--seconds", "30",
                 "--output", str(destination)],
                capture_output=True, text=True, encoding="utf-8", env=environment,
                timeout=30, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            summary = json.loads(process.stdout)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(summary["event"]["state"], "complete")
            self.assertAlmostEqual(summary["time"], 30)
            self.assertEqual(payload["snapshot"]["scenario"], "pass_occluded")
            self.assertGreater(len(payload["decision_records"]), 0)


class NativeBridgeTests(unittest.TestCase):
    def test_json_roundtrip_is_immutable_and_rejects_hidden_truth_fields(self):
        data = {"memory_epoch": 0}
        event = Observation("status", 0, data)
        data["memory_epoch"] = 9
        self.assertEqual(event.data["memory_epoch"], 0)
        self.assertEqual(Observation.from_dict(event.to_dict()), event)
        with self.assertRaises(TypeError):
            event.data["memory_epoch"] = 1
        adapter = EstimatorAdapter()
        for extra in ("truth", "actual_cache", "occluded"):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    adapter.handle(Observation("status", 0, {"memory_epoch": 0, extra: 1}))

    def test_observation_boolean_fields_cannot_accept_truthy_strings(self):
        adapter = EstimatorAdapter()
        for kind, data in (
            ("verification", {"task_id": "unknown", "success": "false"}),
            ("pass_confirmation", {"pass_id": "unknown", "normal_after": "false"}),
            ("pass_by", {"pass_id": "p", "items": [], "normal_before": "false"}),
        ):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    adapter.handle(Observation(kind, 0, data))


if __name__ == "__main__":
    unittest.main()
