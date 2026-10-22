"""Transport contracts, including rejection of simulator truth leakage."""

import json
import unittest

from knowledge_estimation.simulator.bridge import EstimatorAdapter, Observation


N = {"kind": "N", "object_id": "P", "site_id": "S1", "version": 1}
I = {"kind": "I", "object_id": "P", "version": 1}


class ObservationAdapterTests(unittest.TestCase):
    def test_json_messages_drive_same_estimator_lifecycle(self):
        adapter = EstimatorAdapter()
        events = [
            Observation("send", 1, {"msg_id": "m1", "items": [N]}),
            Observation("timeout", 2, {"msg_id": "m1"}),
            Observation("ack", 3, {"msg_id": "m1", "stored_items": [N], "memory_epoch": 0}),
            Observation("result", 4, {"task_id": "e1/a1", "learned_items": [I]}),
        ]
        for event in events:
            adapter.handle(Observation.from_dict(json.loads(json.dumps(event.to_dict()))))
        self.assertEqual(adapter.belief("P", "S1").as_policy_input(), {"b_N": 1, "b_C": 0, "b_I": 0})
        adapter.handle(Observation("verification", 5, {"task_id": "e1/a1", "success": True}))
        self.assertAlmostEqual(adapter.belief("P", "S1").appearance.probability, .9)

    def test_payload_is_copied_and_deeply_frozen(self):
        data = {"msg_id": "m1", "items": [dict(N)]}
        event = Observation("send", 1, data)
        data["items"][0]["version"] = 99
        self.assertEqual(event.to_dict()["data"]["items"][0]["version"], 1)
        with self.assertRaises(TypeError):
            event.data["items"][0]["version"] = 99

    def test_hidden_truth_and_unknown_fields_are_not_an_observation(self):
        adapter = EstimatorAdapter()
        for extra in ("truth", "actual_cache", "occluded", "unknown_epoch"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                adapter.handle(Observation("pass_by", 1, {"pass_id": "r1", "items": [N], extra: True}))
        self.assertEqual(adapter.estimator.history, ())

    def test_malformed_and_nonfinite_events_rejected(self):
        for timestamp in (-1, True, float("nan"), float("inf")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                Observation("status", timestamp, {"memory_epoch": 1})
        with self.assertRaises(ValueError):
            Observation.from_dict({"kind": "status", "timestamp": 1, "data": {}, "truth": {}})
        with self.assertRaises(ValueError):
            EstimatorAdapter().handle(Observation("send", 1, {"msg_id": "m1", "items": [{**N, "truth": 1}]}))
        for event in (
            Observation("verification", 1, {"task_id": "e1", "success": "false"}),
            Observation("pass_confirmation", 1, {"pass_id": "r1", "normal_after": "false"}),
            Observation("send", 1, {"msg_id": "m1", "items": [N], "p_delivery": True}),
        ):
            with self.subTest(kind=event.kind), self.assertRaises(ValueError):
                EstimatorAdapter().handle(event)


if __name__ == "__main__":
    unittest.main()
