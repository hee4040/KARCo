"""Record instrumented A events and separate evaluation-only B DB snapshots.

This collector has no simulated success rate, random event generator or RGB
recognizer. ROS/robot callbacks provide observations and atomic database dumps.
"""

import csv
import hashlib
import json
from pathlib import Path
import argparse
import shutil
from dataclasses import asdict

from .measurement import TRIAL_FIELDS, META_FIELDS, check_meta, label_before, validate_raw_snapshot, load_dataset, decision_pairs, write_json, time_value
from .parameters import ParameterStore, MissingMeasurement
from .probabilistic import KnowledgeEstimator
from .protocol import EstimatorAdapter, Observation
from .types import identifier


class MeasurementRecorder:
    def __init__(self, output, *, protocol_id, episode_id, origin="measured", estimator=None):
        if origin not in {"measured", "fixture", "simulation"}:
            raise ValueError("unknown origin")
        identifier(protocol_id, "protocol_id")
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema_version": 1, "complete": False, "origin": origin,
                         "protocol_id": protocol_id, "evidence": {}, "sha256": {}}
        self.adapter = EstimatorAdapter(estimator)
        store = self.adapter.estimator.parameters
        if store.origin != "unmeasured" and (store.origin != origin or store.artifact.get("protocol_id") != protocol_id):
            raise ValueError("parameter origin/protocol differs from the capture protocol")
        self._events = 0
        self._decisions, self._after = {}, set()
        self._closed = False
        for name in ("observations.jsonl", "inputs.jsonl", "labels.jsonl"):
            (self.root / name).touch()
        (self.root / "transport.jsonl").touch()
        (self.root / "measurement_gaps.jsonl").touch()
        with (self.root / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(TRIAL_FIELDS)
        write_json(self.root / "manifest.json", self.manifest)
        self.begin_episode(episode_id, estimator=estimator)

    def begin_episode(self, episode_id, *, estimator=None):
        identifier(episode_id, "episode_id")
        if self._decisions and set(self._decisions) != self._after:
            raise ValueError("finish after snapshots before starting another episode")
        if estimator is None:
            previous = self.adapter.estimator
            estimator = KnowledgeEstimator(receiver_id=previous.receiver_id, config=previous.config,
                                           parameters=previous.parameters)
        self.adapter = EstimatorAdapter(estimator)
        store = estimator.parameters
        if store.origin != "unmeasured" and (store.origin != self.manifest["origin"] or store.artifact.get("protocol_id") != self.manifest["protocol_id"]):
            raise ValueError("episode parameters differ from capture protocol")
        self._episode = episode_id
        self._append("observations.jsonl", {"episode_id": episode_id, "reset": True,
            "receiver_id": estimator.receiver_id, "memory_epoch": estimator.memory_epoch})
        self._events += 1

    def _append(self, name, row):
        if self._closed:
            raise ValueError("dataset is finalized")
        with (self.root / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def add_evidence(self, evidence_id, relative_path, *, description):
        identifier(evidence_id, "evidence_id")
        identifier(description, "evidence description")
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise ValueError("copy raw evidence into this dataset before registering it")
        if evidence_id in self.manifest["evidence"]:
            raise ValueError("evidence id already exists")
        self.manifest["evidence"][evidence_id] = {"path": path.relative_to(self.root.resolve()).as_posix(),
                                                  "description": description}

    def observe(self, observation: Observation):
        try:
            self.adapter.handle(observation)
        except MissingMeasurement as error:
            # Valid events remain collectable before calibration. Unknown
            # state is explicit; replay after fitting supplies the measurement.
            self._append("measurement_gaps.jsonl", {"episode_id": self._episode,
                "timestamp": observation.timestamp, "reason": str(error)})
        self._append("observations.jsonl", {"episode_id": self._episode, "event": observation.to_dict()})
        self._events += 1

    def record_decision(self, *, decision_id, meta, time, object_ref, site_id,
                        event_id, task_version, attempt_id,
                        selected_items, required_items, before_snapshot, world_before=None):
        """Call before any transmission; B truth is written to labels only."""
        identifier(decision_id, "decision_id")
        identifier(event_id, "event_id")
        if type(task_version) is not int or type(attempt_id) is not int or min(task_version, attempt_id) < 0:
            raise ValueError("task version and attempt id must be non-negative integers")
        if decision_id in self._decisions:
            raise ValueError("decision id already exists")
        if set(meta) != META_FIELDS:
            raise ValueError("metadata requires exactly declared grouping fields; no privileged extras")
        check_meta(meta, self.manifest)
        if meta["episode_id"] != self._episode:
            raise ValueError("decision is not in the active A observation episode")
        if before_snapshot["time"] != time:
            raise ValueError("capture the B DB at the decision time")
        validate_raw_snapshot(before_snapshot)
        if world_before is not None:
            label_before(before_snapshot, world_before, time)
        if world_before is not None and self.manifest["origin"] == "measured" and world_before["assessment_method"] not in {"external_camera", "manual_annotation"}:
            raise ValueError("synthetic truth cannot be logged as measured")
        snapshot = self.adapter.belief(object_ref, site_id, now=time)
        for name, items in (("selected_items", selected_items), ("required_items", required_items)):
            if len(set(items)) != len(items) or set(items) - set("ENCI"):
                raise ValueError(f"invalid {name}")
        row = {**meta, "decision_id": decision_id, "time": time, "object_ref": object_ref,
               "event_id": event_id, "task_version": task_version, "attempt_id": attempt_id,
               "site_id": site_id, "receiver_id": snapshot.receiver_id,
               "known_epoch": snapshot.memory_epoch, "history_end": self._events,
               "b_raw": snapshot.as_policy_input(), "estimator_fingerprint": snapshot.estimator_fingerprint,
               "parameter_fit_groups": self.adapter.estimator.parameters.artifact.get("groups", {}),
               "known_versions": {b.item.key.kind.value: b.item.version for b in (snapshot.normal, snapshot.current, snapshot.appearance)},
               "semantics": {b.item.key.kind.value: b.semantics for b in (snapshot.normal, snapshot.current, snapshot.appearance)},
               "hard_invalid": {b.item.key.kind.value: b.hard_invalid for b in (snapshot.normal, snapshot.current, snapshot.appearance)},
               "selected_items": list(selected_items), "required_items": list(required_items)}
        beliefs = (snapshot.normal, snapshot.current, snapshot.appearance)
        row["evidence"] = {b.item.key.kind.value: [asdict(e) for e in b.evidence] for b in beliefs}
        table = {e["parameter_id"]: e for e in self.adapter.estimator.parameters.artifact["parameters"]}
        row["source_contexts"] = {b.item.key.kind.value: [
            {"name": table[p]["name"], "context": table[p]["context"]}
            for p in sorted({p for e in b.evidence for p in e.parameter_ids}) if p in table] for b in beliefs}
        self._append("inputs.jsonl", row)
        self._append("labels.jsonl", {"decision_id": decision_id, "phase": "before",
            "receiver_id": snapshot.receiver_id, "object_ref": object_ref,
            "snapshot": before_snapshot, "world": world_before})
        self._decisions[decision_id] = row

    def record_after_receipt(self, decision_id, *, snapshot, world=None, outcome=None):
        row = self._decisions[decision_id]
        validate_raw_snapshot(snapshot)
        if decision_id in self._after or snapshot["time"] < row["time"]:
            raise ValueError("duplicate or out-of-order after snapshot")
        if world is not None:
            label_before(snapshot, world, snapshot["time"])
        if world is not None and self.manifest["origin"] == "measured" and world["assessment_method"] not in {"external_camera", "manual_annotation"}:
            raise ValueError("synthetic truth cannot be logged as measured")
        self._append("labels.jsonl", {"decision_id": decision_id, "phase": "after",
            "receiver_id": row["receiver_id"], "object_ref": row["object_ref"],
            "snapshot": snapshot, "world": world, "outcome": outcome})
        self._after.add(decision_id)

    def record_transport(self, *, direction, message_kind, encoded_payload: bytes,
                         decision_id, category, time, is_retry):
        """Instrument the actual serializer's bytes, without sending anything."""
        if not isinstance(encoded_payload, bytes):
            raise ValueError("record the exact serialized bytes, not an assumed size")
        identifier(message_kind, "message_kind")
        time_value(time, "transport timestamp")
        if direction not in {"A_to_B", "B_to_A"} or category not in {"initial", "roundtrip", "probe"}:
            raise ValueError("unknown direction/category")
        if type(is_retry) is not bool:
            raise ValueError("is_retry must be boolean")
        self._append("transport.jsonl", {"episode_id": self._episode, "decision_id": decision_id,
            "direction": direction, "message_kind": message_kind, "time": time,
            "category": category, "is_retry": is_retry, "bytes": len(encoded_payload),
            "payload_sha256": hashlib.sha256(encoded_payload).hexdigest()})

    def record_trial(self, row):
        if set(row) != set(TRIAL_FIELDS):
            raise ValueError("trial requires all CSV columns; unused cells must be empty strings")
        check_meta(row, self.manifest)
        if row["evidence_id"] not in self.manifest["evidence"]:
            raise ValueError("register raw evidence first")
        if self._closed:
            raise ValueError("dataset is finalized")
        with (self.root / "trials.csv").open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
            writer.writerow(row)

    def finalize(self):
        if self._closed or set(self._decisions) != self._after:
            raise ValueError("dataset already finalized or missing post-receipt snapshots")
        # Structural validation before declaring complete.
        captured = list(decision_pairs(self.root, require_labels=False))
        self.manifest["annotations_complete"] = all(before["world"] is not None and after["world"] is not None
                                                     for _, before, after, _ in captured)
        self.manifest["sha256"] = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(self.root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        self.manifest["complete"] = True
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            load_dataset(self.root, allow_fixture=self.manifest["origin"] == "fixture",
                         allow_simulation=self.manifest["origin"] == "simulation")
        except Exception:
            self.manifest["complete"] = False
            (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
            raise
        self._closed = True
        return self.manifest


def replay(root, output, parameters, *, allow_fixture=False, allow_simulation=False):
    """Recompute raw scores with frozen measured rates, using A prefixes only.

    Actual selected items/outcomes are retained, not simulated counterfactuals.
    Privileged labels are copied after prediction and never passed to adapter.
    """
    root = Path(root)
    manifest, trials, inputs = load_dataset(root, allow_fixture=allow_fixture, allow_simulation=allow_simulation)
    if parameters.origin != manifest["origin"]:
        raise ValueError("parameter and log origins differ")
    if not inputs:
        raise ValueError("no decision logs to replay")
    # Group independence between source calibration and final test is audited
    # by the recorded source split and the common dataset manifest.
    source_groups = parameters.artifact.get("groups", {})
    for row in inputs:
        if row["split"] in {"validation", "test"}:
            for field, values in source_groups.items():
                if row[field] in values:
                    raise ValueError(f"source fitting leaks {field} into evaluation")
    pairs = {r["decision_id"]: (before, after) for r, before, after, _ in decision_pairs(root, require_labels=False)}
    from .measurement import read_jsonl
    journal = list(read_jsonl(root / "observations.jsonl"))
    first = journal[0]
    if not first.get("reset"):
        raise ValueError("A journal must begin with an explicit episode reset")
    estimator = KnowledgeEstimator(first["receiver_id"], memory_epoch=first["memory_epoch"], parameters=parameters)
    writer = MeasurementRecorder(output, protocol_id=manifest["protocol_id"], episode_id=first["episode_id"],
                                 origin=manifest["origin"], estimator=estimator)
    for evidence_id, reference in manifest["evidence"].items():
        destination = writer.root / reference["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / reference["path"], destination)
        writer.add_evidence(evidence_id, reference["path"], description=reference["description"])
    for trial in trials:
        writer.record_trial(trial)
    cursor = 1
    for row in inputs:
        end = row["history_end"]
        if type(end) is not int or not cursor <= end <= len(journal):
            raise ValueError("invalid/non-causal A journal prefix")
        for wrapper in journal[cursor:end]:
            if wrapper.get("reset"):
                writer.begin_episode(wrapper["episode_id"], estimator=KnowledgeEstimator(wrapper["receiver_id"],
                    memory_epoch=wrapper["memory_epoch"], parameters=parameters))
            else:
                if wrapper["episode_id"] != writer._episode:
                    raise ValueError("A journal episode mismatch")
                writer.observe(Observation.from_dict(wrapper["event"]))
        cursor = end
        before, after = pairs[row["decision_id"]]
        meta = {name: row[name] for name in ("split", "episode_id", "session_id", "object_instance", "route_group_id", "origin", "protocol_id")}
        writer.record_decision(decision_id=row["decision_id"], meta=meta, time=row["time"],
            object_ref=row["object_ref"], site_id=row["site_id"], selected_items=row["selected_items"],
            event_id=row["event_id"], task_version=row["task_version"], attempt_id=row["attempt_id"],
            required_items=row["required_items"], before_snapshot=before["snapshot"], world_before=before["world"])
        writer.record_after_receipt(row["decision_id"], snapshot=after["snapshot"], world=after["world"], outcome=after.get("outcome"))
    if "transport.jsonl" in manifest["sha256"]:
        shutil.copyfile(root / "transport.jsonl", writer.root / "transport.jsonl")
    writer.manifest["replayed_from_manifest_sha256"] = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    return writer.finalize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--allow-simulation", action="store_true")
    args = parser.parse_args(argv)
    try:
        store = ParameterStore.load(args.params, allow_fixture=args.allow_fixture, allow_simulation=args.allow_simulation)
        result = replay(args.dataset, args.output, store, allow_fixture=args.allow_fixture, allow_simulation=args.allow_simulation)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Replay stopped: {error}\n")
    print(json.dumps({"output": str(args.output), "origin": result["origin"], "complete": result["complete"]}))


if __name__ == "__main__":
    main()
