"""Read-only audit of raw acquisition errors and geometry dataset quality.

python -m knowledge_estimation.simulation.audit_acquisition --output knowledge_estimation/data/acquisition_audit_v1
This audits existing data; it neither fits an estimator nor creates new trials.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from math import cos, sin, tan, radians, isclose
from pathlib import Path

from ..calibrate import fit_sources
from ..measurement import load_dataset, read_jsonl, decision_pairs, write_json
from ..parameters import ParameterStore, MissingMeasurement
from ..probabilistic import RawKnowledgeEstimatorV2
from ..protocol import EstimatorAdapter, Observation
from .analyze import write_csv
from .world import run_pass


def crosses(a, b, c, d):
    """Closed 2D segments, orientation test independent of slab clipping."""
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def on(p, q, r):
        return abs(orient(p, q, r)) < 1e-10 and all(min(p[i], q[i])-1e-10 <= r[i] <= max(p[i], q[i])+1e-10 for i in (0, 1))
    values = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (values[0]*values[1] < 0 and values[2]*values[3] < 0 or
            on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b))


def occluded(a, b, walls):
    for x0, y0, x1, y1 in walls:
        if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (a, b)):
            return True
        vertices = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        if any(crosses(a, b, vertices[i], vertices[(i+1) % 4]) for i in range(4)):
            return True
    return False


def seen(pose, target, config, walls):
    # Camera-frame x/z formulation; production uses angular atan2 comparisons.
    dx, dy = target[0]-pose[0], target[1]-pose[1]
    z = dx*cos(pose[2])+dy*sin(pose[2])
    lateral = -dx*sin(pose[2])+dy*cos(pose[2])
    return (config["camera_min_depth_m"] <= z <= config["camera_range_m"] and
            abs(lateral) <= z*tan(radians(config["camera_fov_degrees"])/2)+1e-10 and
            not occluded(pose[:2], target, walls))


def audit_trace(trace, config):
    issues, acquisition = [], {}
    previous_seen, dwell, maximum = False, 0., 0.
    speed = trace["speed_mps"]
    for index, frame in enumerate(trace["frames"]):
        expected_time = round(index*config["dt_seconds"], 10)
        expected_pose = (config["route_x"][0]+speed*expected_time, trace["offset"], radians(trace["heading"]))
        if not isclose(frame["time"], expected_time, abs_tol=1e-9) or any(not isclose(a, b, abs_tol=1e-9) for a, b in zip(frame["pose"], expected_pose)):
            issues.append("time_or_pose")
        visible = seen(expected_pose, config["object_xy"], config, trace["walls"])
        dwell = dwell+config["dt_seconds"] if index and visible and previous_seen else 0.
        maximum = max(maximum, dwell)
        for kind in "NI":
            if visible and dwell+1e-9 >= config["storage_dwell_seconds"][kind] and kind not in acquisition:
                acquisition[kind] = expected_time
        if visible != frame["visible"] or abs(dwell-frame["contiguous_visible_seconds"]) > 1e-8 or sorted(acquisition) != frame["stored"]:
            issues.append("visibility_dwell_or_stored")
        previous_seen = visible
    duration = (config["route_x"][1]-config["route_x"][0])/speed
    if len(trace["frames"]) != round(duration/config["dt_seconds"])+1:
        issues.append("frame_count")
    if acquisition != trace["acquired_at"]:
        issues.append("acquisition_time")
    delivered = not occluded(config["sender_xy"], trace["frames"][-1]["pose"][:2], trace["walls"])
    if delivered != trace["delivered"]:
        issues.append("channel")
    return acquisition, maximum, sorted(set(issues))


def geometry_key(trace):
    return (trace["speed_mps"], trace["offset"], trace["heading"], json.dumps(trace["walls"], sort_keys=True))


def independent_label(snapshot, world):
    result = {}
    for kind in "NCI":
        item = snapshot["items"].get(kind)
        valid = item is not None
        if item is not None:
            valid = (item["version"] == world["versions"][kind] and item["valid_content"] and
                     item["physical_object_id"] == world["physical_object_id"] and item["frame_id"] == world["frame_id"])
            if kind == "C":
                valid = valid and snapshot["time"] < item["observed_at"]+item["ttl_seconds"]
        result[kind] = int(valid)
    return result


def condition_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        for factor in ("all", "speed_mps", "route_standoff_m", "heading_degrees", "walls"):
            groups[row["stage"], row["kind"], factor, "all" if factor == "all" else row[factor]].append(row)
    table = []
    for (stage, kind, factor, value), group in groups.items():
        table.append(dict(stage=stage, kind=kind, factor=factor, value=value, unique_designs=len(group),
            acquired=sum(r["K"] for r in group), actual_rate=sum(r["K"] for r in group)/len(group),
            mean_prediction=sum(r["prediction"] for r in group)/len(group),
            signed_gap=sum(r["prediction"]-r["K"] for r in group)/len(group),
            brier=sum((r["prediction"]-r["K"])**2 for r in group)/len(group)))
    return table


def run(output, *, source_root, evaluation_root):
    source_root, evaluation_root, root = Path(source_root), Path(evaluation_root), Path(output)
    parameters = ParameterStore.load(source_root/"simulation_params.yaml", allow_simulation=True)
    roots = {"source": source_root/"source_runs", **{s: evaluation_root/s/"new"/"episodes" for s in ("validation", "test")}}
    root.mkdir(parents=True, exist_ok=False)
    issues, checks, observations, identities = [], {}, {}, {}
    records, sensitivity, parameter_checks, manifest_hashes = [], [], [], {}
    metadata_sets = {}
    for stage, data in roots.items():
        manifest, trials, inputs = load_dataset(data, allow_simulation=True)
        manifest_hashes[stage] = hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest()
        config = json.loads((data/"scenario.json").read_text(encoding="utf-8"))
        traces = list(read_jsonl(data/"world_trace.jsonl"))
        trace_by_episode = {t["episode_id"]: t for t in traces}
        if len(trace_by_episode) != len(traces):
            issues.append(dict(stage=stage, issue="duplicate_episode_trace"))
        metadata_sets[stage] = {field: {r[field] for r in trials+inputs} for field in ("episode_id", "session_id", "object_instance", "route_group_id")}
        unique = {}
        # Select normal_channel before any interventions; do not multiply the
        # acquisition sample size by eleven copies of the same physical pass.
        canonical = traces if stage == "source" else [t for t in traces if t["case"] == "normal_channel"]
        for trace in canonical:
            key = geometry_key(trace)
            if key in unique:
                issues.append(dict(stage=stage, issue="duplicate_canonical_geometry", episode_id=trace["episode_id"]))
            acquired, maximum, failures = audit_trace(trace, config)
            unique[key] = trace
            for failure in failures:
                issues.append(dict(stage=stage, issue=failure, episode_id=trace["episode_id"]))
            for kind in "NI":
                context = {"receiver_id": "B", "kind": kind, "site_id": "S2", "route_id": "table-route",
                           "condition_id": trace.get("pass_condition", f"unreported-view:speed={trace['speed_mps']:g}mps")}
                p = parameters.entry("p_acq", context)["value"]
                records.append(dict(stage=stage, episode_id=trace["episode_id"], kind=kind,
                    speed_mps=trace["speed_mps"], route_standoff_m=round(abs(config["object_xy"][1]-trace["offset"]), 10),
                    heading_degrees=trace["heading"], walls=bool(trace["walls"]), K=int(kind in acquired), prediction=p,
                    max_contiguous_seconds=maximum, threshold_seconds=config["storage_dwell_seconds"][kind]))
            # Numerical sensitivity, not new measured trials or replacement labels.
            for divisor in (2, 4):
                fine_config = {**config, "speed_mps": trace["speed_mps"], "dt_seconds": config["dt_seconds"]/divisor}
                fine = run_pass(fine_config, trace["offset"], trace["heading"], bool(trace["walls"]))
                for kind in "NI":
                    sensitivity.append(dict(stage=stage, episode_id=trace["episode_id"], kind=kind, dt_divisor=divisor,
                        coarse_K=int(kind in acquired), finer_K=int(kind in fine.acquired_at),
                        label_changed=(kind in acquired) != (kind in fine.acquired_at)))
        identities[stage] = set(unique)
        if stage == "source":
            refit = fit_sources(data, allow_simulation=True)
            def entry_key(entry):
                return entry["name"], json.dumps(entry["context"], sort_keys=True)
            frozen = {entry_key(e): e for e in parameters.artifact["parameters"]}
            for entry in refit["parameters"]:
                old = frozen[entry_key(entry)]
                # Includes empirical CDF samples_seconds/horizon, evidence and
                # provenance, not just scalar success rates.
                ok = entry == old
                parameter_checks.append(dict(name=entry["name"], context=json.dumps(entry["context"], sort_keys=True), matched=ok))
                if not ok:
                    issues.append(dict(stage=stage, issue="source_parameter_count_mismatch"))
            if {entry_key(e) for e in refit["parameters"]} != set(frozen):
                issues.append(dict(stage=stage, issue="source_parameter_set_mismatch"))
            for row in trials:
                trace = trace_by_episode[row["episode_id"]]
                if row["source"] in {"p_acq", "F_pass"}:
                    expected = row["kind"] in trace["acquired_at"]
                    if int(row["success"]) != int(expected) or (row["source"] == "F_pass" and float(row["duration_seconds"]) != trace["acquired_at"][row["kind"]]):
                        issues.append(dict(stage=stage, issue="trial_trace_mismatch"))
                elif row["source"] == "p_recv":
                    if int(row["success"]) != int(trace["delivered"]):
                        issues.append(dict(stage=stage, issue="delivery_trial_trace_mismatch"))
                elif row["source"] in {"p_ack", "F_recv", "r_crop", "r_current"}:
                    # These successes are explicitly ideal-model outcomes,
                    # never independent measurements of hardware quality.
                    if not trace["delivered"] or row["success"] != "1":
                        issues.append(dict(stage=stage, issue="ideal_channel_content_trial_mismatch"))
                    if row["source"] == "F_recv" and float(row["duration_seconds"]) != config["one_way_delay_seconds"]:
                        issues.append(dict(stage=stage, issue="delivery_duration_mismatch"))
        else:
            pair_truth, label_count, after_pass_count = {}, 0, 0
            physical_ids, frames = set(), set()
            for row, before, after, truth in decision_pairs(data):
                for label in (before, after):
                    independent = independent_label(label["snapshot"], label["world"])
                    physical_ids.add(label["world"]["physical_object_id"])
                    frames.add(label["world"]["frame_id"])
                    label_count += 3
                    if label is before and any(independent[k] != truth[k]["K"] for k in "NCI"):
                        issues.append(dict(stage=stage, issue="label_contract_mismatch", decision_id=row["decision_id"]))
                pair_truth[row["decision_id"]] = {k: truth[k]["K"] for k in "NCI"}
                if row["decision_id"].endswith("/after_pass"):
                    t = trace_by_episode[row["episode_id"]]
                    for kind in "NI":
                        after_pass_count += 1
                        if truth[kind]["K"] != int(kind in t["acquired_at"]):
                            issues.append(dict(stage=stage, issue="pass_DB_label_mismatch"))
                        if t["case"] == "normal_channel":
                            expected = next(r["prediction"] for r in records if r["stage"] == stage and r["episode_id"] == t["episode_id"] and r["kind"] == kind)
                            if row["b_raw"]["b_"+kind] != expected:
                                issues.append(dict(stage=stage, issue="pass_prior_logged_prediction_mismatch"))
            # Replay only A events. Privileged geometry/labels are never passed.
            decisions = defaultdict(list)
            for row in inputs:
                decisions[row["history_end"]].append(row)
            adapter, replay_count = None, 0
            log = list(read_jsonl(data/"observations.jsonl"))
            for index, event in enumerate(log, 1):
                if event.get("reset"):
                    adapter = EstimatorAdapter(RawKnowledgeEstimatorV2(parameters=parameters))
                else:
                    try:
                        adapter.handle(Observation.from_dict(event["event"]))
                    except MissingMeasurement:
                        # Collection explicitly logs unsupported contexts too;
                        # their state contains None, never an invented value.
                        pass
                for row in sorted(decisions.get(index, []), key=lambda r: r["time"]):
                    belief = adapter.belief("Q", "S2", now=row["time"])
                    actual = belief.as_policy_input()
                    if actual != row["b_raw"]:
                        issues.append(dict(stage=stage, issue="A_only_replay_mismatch", decision_id=row["decision_id"]))
                    replay_count += 1
            if replay_count != len(inputs):
                issues.append(dict(stage=stage, issue="missing_replayed_decision"))
            # Audit recorded score/label joins too, not only snapshots.
            import csv
            with (data.parent/"predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            for p in predictions:
                key = p["episode_id"]+"/"+p["phase"]
                if int(p["K"]) != pair_truth[key][p["kind"]]:
                    issues.append(dict(stage=stage, issue="prediction_label_join_mismatch"))
            observations[stage] = {"label_items_checked": label_count, "after_pass_NI_labels_checked": after_pass_count,
                "A_only_decisions_replayed": replay_count, "prediction_label_joins": len(predictions),
                "physical_object_ids": sorted(physical_ids), "frame_ids": sorted(frames)}
        checks[stage] = {"traces": len(traces), "unique_canonical_passes": len(unique),
            "canonical_frames_checked": sum(len(t["frames"]) for t in unique.values()),
            "nominal_object_instances": len(metadata_sets[stage]["object_instance"])}
    overlaps = []
    for a, b in (("source", "validation"), ("source", "test"), ("validation", "test")):
        overlaps.append(dict(first=a, second=b, exact_geometry_overlap=len(identities[a] & identities[b]),
            **{field+"_overlap": len(metadata_sets[a][field] & metadata_sets[b][field]) for field in metadata_sets[a]}))
    write_csv(root/"acquisition_designs.csv", records)
    write_csv(root/"conditions.csv", condition_metrics(records))
    write_csv(root/"temporal_resolution.csv", sensitivity)
    write_csv(root/"parameter_recount.csv", parameter_checks)
    write_csv(root/"split_overlap.csv", overlaps)
    summary = {"scope": "Read-only exploratory audit of existing synthetic data; test is now diagnostic and not untouched for future changes",
        "data_roots": {s: str(p) for s, p in roots.items()}, "input_manifest_sha256": manifest_hashes,
        "checks": checks, "label_and_replay_checks": observations,
        "parameter_entries_recounted": len(parameter_checks), "integrity_issues": issues,
        "split_overlap": overlaps,
        "resolution_label_changes": {stage: {str(d): sum(r["label_changed"] for r in sensitivity if r["stage"] == stage and r["dt_divisor"] == d) for d in (2, 4)} for stage in roots},
        "same_world_code_across_datasets": len({(p/"model_world.py").read_bytes() for p in roots.values()}) == 1,
        "findings_not_integrity_errors": ["Different source/evaluation heading and offset grids change outcome mixtures",
            "source=72 unique passes, validation/test=24 each; intervention copies are not independent acquisition trials",
            "Synthetic object_instance/session/route IDs do not represent different physical objects/maps",
            "Deterministic ideal detector and arbitrary experimental dwell gates cannot validate physical D455 recognition",
            "No probability fitting, score calibration, relabeling, or source file edits in this audit"]}
    write_json(root/"summary.json", summary)
    if issues:
        raise AssertionError(f"{len(issues)} audit issues; inspect summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("knowledge_estimation/data/calibration_repair_v1"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("knowledge_estimation/data/raw_revision_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, source_root=args.source_root, evaluation_root=args.evaluation_root), indent=2))
