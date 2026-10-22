"""Shared measured-log contracts; privileged DB truth is offline-only."""

import csv
import hashlib
import json
from pathlib import Path

from .types import identifier, counter


GROUP_FIELDS = ("session_id", "object_instance", "route_group_id")
META_FIELDS = {"split", "episode_id", *GROUP_FIELDS, "origin", "protocol_id"}
SPLITS = {"train", "calibration", "validation", "test"}
TRIAL_FIELDS = ("split", "episode_id", *GROUP_FIELDS, "origin", "protocol_id", "evidence_id",
                "opportunity_id", "source", "receiver_id", "kind", "site_id", "route_id",
                "condition_id", "success", "duration_seconds", "deadline_seconds")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def time_value(value, name):
    if type(value) not in (int, float) or not 0 <= value < float("inf"):
        raise ValueError(f"{name} must be finite and non-negative")


def check_meta(row, manifest):
    if row["split"] not in SPLITS:
        raise ValueError("unknown data split")
    for name in ("episode_id", *GROUP_FIELDS):
        identifier(row[name], name)
    if row["origin"] != manifest["origin"] or row["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("mixed protocol or data origin")


def check_split_groups(rows):
    seen = {key: {} for key in (*GROUP_FIELDS, "episode_id")}
    for row in rows:
        for field, mapping in seen.items():
            prior = mapping.setdefault(row[field], row["split"])
            if prior != row["split"]:
                raise ValueError(f"{field} leaks across splits: {row[field]}")


def load_dataset(root, *, allow_fixture=False, allow_simulation=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("complete"):
        raise ValueError("dataset must be complete measured-log schema 1")
    if manifest["origin"] not in {"measured", "fixture", "simulation"}:
        raise ValueError("unknown data origin")
    if manifest["origin"] == "simulation" and not allow_simulation:
        raise ValueError("simulation data requires --allow-simulation; it is not hardware measurement")
    if manifest["origin"] == "fixture" and not allow_fixture:
        raise ValueError("test fixtures are not measured data; use --allow-fixture only for a smoke test")
    for relative, digest in manifest["sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"dataset checksum/path mismatch: {relative}")
    for name in ("observations.jsonl", "inputs.jsonl", "labels.jsonl", "trials.csv"):
        if name not in manifest["sha256"]:
            raise ValueError(f"missing checksum: {name}")
    with (root / "trials.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError("unexpected trials.csv schema")
        trials = list(reader)
    inputs = list(read_jsonl(root / "inputs.jsonl"))
    for row in trials + inputs:
        check_meta(row, manifest)
        if row.get("evidence_id") and row["evidence_id"] not in manifest["evidence"]:
            raise ValueError("trial has no registered raw measurement evidence")
    check_split_groups(trials + inputs)
    for row in read_jsonl(root / "labels.jsonl"):
        world = row["world"]
        if world is None:
            continue  # Captured raw DB; external annotation is still pending.
        if world["assessment_id"] not in manifest["evidence"]:
            raise ValueError("external truth must link to registered raw evidence")
        if manifest["origin"] == "measured" and world["assessment_method"] not in {"external_camera", "manual_annotation"}:
            raise ValueError("synthetic truth cannot be declared measured")
        if manifest["origin"] == "simulation" and world["assessment_method"] != "simulator_ground_truth":
            raise ValueError("simulation labels must identify simulator ground truth")
        for content in row["snapshot"]["items"].values():
            if not content["evidence_ids"] or any(e not in manifest["evidence"] for e in content["evidence_ids"]):
                raise ValueError("DB validity assessment lacks raw evidence")
    for reference in manifest["evidence"].values():
        if reference["path"] not in manifest["sha256"]:
            raise ValueError("raw evidence must be inside dataset with a checksum")
    return manifest, trials, inputs


def label_before(snapshot, world, expected_time):
    """R and K at the decision time; never inferred from request-only labels.

    world is external identity/version assessment, not A/B's own recognition.
    Absence of an entry means absence from an atomic B database snapshot.
    """
    if snapshot["time"] != expected_time or world["time"] != expected_time:
        raise ValueError("B snapshot and external world assessment must align with the decision time")
    time_value(expected_time, "decision time")
    identifier(world["assessment_id"], "external assessment_id")
    if world["assessment_method"] not in {"external_camera", "manual_annotation", "fixture", "simulator_ground_truth"}:
        raise ValueError("final truth requires external_camera/manual_annotation")
    identifier(world["physical_object_id"], "external physical object identity")
    identifier(world["frame_id"], "external frame_id")
    for kind in "NCI":
        counter(world["versions"][kind], "external world version")
    result = {}
    for kind in "NCI":
        row = snapshot["items"].get(kind)
        stored = row is not None
        valid, causes = False, []
        if row is not None:
            for name in ("version", "physical_object_id", "valid_content", "frame_id", "observed_at", "received_at", "evidence_ids"):
                if name not in row:
                    raise ValueError(f"B DB entry lacks {name}")
            if type(row["valid_content"]) is not bool:
                raise ValueError("valid_content requires an externally assessed boolean")
            time_value(row["observed_at"], "observed_at")
            time_value(row["received_at"], "received_at")
            if row["observed_at"] > expected_time or row["received_at"] > expected_time:
                raise ValueError("future B DB entry")
            if row["physical_object_id"] != world["physical_object_id"]:
                causes.append("wrong_object_association")
            if row["version"] != world["versions"][kind]:
                causes.append("stale_version")
            if not row["valid_content"]:
                causes.append("invalid_content")
            if row["frame_id"] != world["frame_id"]:
                causes.append("frame_mismatch")
            if kind == "C":
                if row.get("ttl_seconds") is None:
                    raise ValueError("C label needs protocol TTL; there is no default")
                time_value(row["ttl_seconds"], "C TTL")
                if expected_time >= row["observed_at"] + row["ttl_seconds"]:
                    causes.append("expired")
            valid = not causes
        else:
            causes.append("missing")
        result[kind] = {"R": int(stored), "K": int(valid), "causes": causes}
    return result


def validate_raw_snapshot(snapshot):
    time_value(snapshot["time"], "snapshot time")
    counter(snapshot["memory_epoch"], "B snapshot memory_epoch")
    if set(snapshot["items"]) - set("NCI"):
        raise ValueError("unknown B DB item kind")
    for kind, row in snapshot["items"].items():
        counter(row["version"], "B DB version")
        identifier(row["frame_id"], "B DB frame_id")
        for name in ("observed_at", "received_at"):
            time_value(row[name], name)
            if row[name] > snapshot["time"]:
                raise ValueError("future entry in B DB snapshot")
        if not isinstance(row["evidence_ids"], list) or not row["evidence_ids"]:
            raise ValueError("raw DB entries need source/capture evidence ids")
        if kind == "C":
            time_value(row["ttl_seconds"], "C task-protocol TTL")


def decision_pairs(root, *, split=None, require_labels=True):
    root = Path(root)
    inputs = [r for r in read_jsonl(root / "inputs.jsonl") if split is None or r["split"] == split]
    selected_ids = {r["decision_id"] for r in inputs}
    labels = {}
    for row in read_jsonl(root / "labels.jsonl"):
        if split is not None and row["decision_id"] not in selected_ids:
            continue
        key = (row["decision_id"], row["phase"])
        if key in labels:
            raise ValueError("duplicate B snapshot phase")
        labels[key] = row
    seen = set()
    for row in inputs:
        key = row["decision_id"]
        if key in seen:
            raise ValueError("duplicate decision id")
        seen.add(key)
        before = labels.pop((key, "before"), None)
        after = labels.pop((key, "after"), None)
        if before is None or after is None:
            raise ValueError("every decision needs before and after B snapshots")
        if before["receiver_id"] != row["receiver_id"] or after["receiver_id"] != row["receiver_id"]:
            raise ValueError("B snapshot receiver mismatch")
        if before["object_ref"] != row["object_ref"] or after["object_ref"] != row["object_ref"]:
            raise ValueError("B snapshot object mismatch")
        if after["snapshot"]["time"] < row["time"]:
            raise ValueError("post-receipt snapshot precedes decision")
        validate_raw_snapshot(before["snapshot"])
        validate_raw_snapshot(after["snapshot"])
        if before["snapshot"]["time"] != row["time"]:
            raise ValueError("pre-decision B DB timestamp mismatch")
        if require_labels and (before["world"] is None or after["world"] is None):
            raise ValueError("external annotations are pending; cannot fit/evaluate guessed labels")
        truth = label_before(before["snapshot"], before["world"], row["time"]) if before["world"] is not None else None
        # Verify the post snapshot contract, but never use it as pre-decision K.
        if after["world"] is not None:
            label_before(after["snapshot"], after["world"], after["snapshot"]["time"])
        yield row, before, after, truth
    if labels:
        raise ValueError("unpaired B snapshots")
