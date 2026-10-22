"""Balanced acquisition collection, reserved calibration, frozen raw evaluation.

python -m knowledge_estimation.simulation.collect_designed --output knowledge_estimation/data/designed_acquisition_v1
All numbers concern a declared ideal 2D experiment, never measured D455 rates.
"""

import argparse
from collections import defaultdict, Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
from math import floor, radians
from pathlib import Path
import shutil

from ..calibrate import fit_sources
from ..collect import MeasurementRecorder
from ..measurement import TRIAL_FIELDS, write_json, read_jsonl, load_dataset
from ..parameters import ParameterStore
from ..probabilistic import RawKnowledgeEstimatorV2
from ..protocol import item_from_dict, item_to_dict
from .analyze import write_csv
from .audit_acquisition import seen, independent_label
from .collection_design import (build_plan, validate_plan, allocations, check_allocations,
                                scene_id, cells, REFERENCE_SPLITS, OOD_SPLITS, digest)
from .run import items, store, snapshot, world
from .world import visible, validate_config


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def put(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n")


def sample_pass(config, draw, divisor=1):
    """Fixed-rate sensor with phase; finer grid contains original sample times."""
    dt = config["dt_seconds"]/divisor
    phase = draw["frame_phase_seconds"] % dt
    speed = draw["speed_mps"]
    duration, _ = validate_config({**config, "speed_mps": speed})
    steps = floor((duration-phase)/dt+1e-9)
    walls = config["wall_rectangles"] if draw["walls"] else []
    frames, acquired, intervals = [], {}, 0
    for index in range(steps+1):
        t = round(phase+index*dt, 10)
        pose = (config["route_x"][0]+speed*t, draw["offset"], radians(draw["heading_degrees"]))
        is_seen = visible(pose, config["object_xy"], z_min=config["camera_min_depth_m"],
                          z_max=config["camera_range_m"], fov=radians(config["camera_fov_degrees"]), walls=walls)
        intervals = intervals+1 if is_seen and frames and frames[-1]["visible"] else 0
        dwell = round(intervals*dt, 10)
        for kind in "NI":
            if is_seen and dwell >= config["storage_dwell_seconds"][kind] and kind not in acquired:
                acquired[kind] = t
        frames.append(dict(time=t, pose=list(pose), visible=is_seen,
                           contiguous_visible_seconds=dwell, stored=sorted(acquired)))
    return frames, acquired


def config_for(plan, split):
    return plan["ood_environment_config"] if split == "ood_environment" else plan["config"]


def A_input(config, draw):
    duration = (config["route_x"][1]-config["route_x"][0])/draw["speed_mps"]
    return dict(trial_id=draw["trial_id"], split=draw["split"], planned_speed_mps=draw["speed_mps"],
        route_id="table-route", site_id="S2", normal_before=True, normal_after=True,
        complete=True, completed_at=duration, approved_items=[item_to_dict(items(config)[k]) for k in "NI"])


def predict(A, parameters):
    """Only this allowlisted object enters the estimator, never a draw/label."""
    est = RawKnowledgeEstimatorV2(parameters=parameters)
    payload = [item_from_dict(i) for i in A["approved_items"]]
    for item in payload:
        est.register(item, now=0.)
    est.record_pass_by("p", payload, route_id=A["route_id"], site_id=A["site_id"],
        condition_id=f"unreported-view:speed={A['planned_speed_mps']:g}mps",
        normal_before=A["normal_before"], complete=A["complete"], now=A["completed_at"])
    est.confirm_pass_by("p", normal_after=A["normal_after"], now=A["completed_at"])
    return {k: est.estimate(item).probability for k, item in zip("NI", payload)}


def validate_A_input(plan, config, draw, A):
    # JSON turns tuple-valued approved poses into lists. Compare their canonical
    # serialized meaning rather than rejecting a valid round trip.
    if set(A) != set(plan["A_input_allowlist"]) or digest(A) != digest(A_input(config, draw)):
        raise ValueError("A input includes hidden data or differs from planned contract")


def source_row(plan, draw, source, kind, acquired):
    config = plan["config"]
    row = dict.fromkeys(TRIAL_FIELDS, "")
    row.update(split="train", episode_id=draw["trial_id"], session_id=f"synthetic-seed-{plan['seed']}",
        object_instance="physical-Q", route_group_id=draw["scene_id"], origin="simulation",
        protocol_id=config["protocol_id"], evidence_id="sim-evidence", opportunity_id=draw["trial_id"],
        source=source, receiver_id="B", kind=kind, site_id="S2", route_id="table-route",
        condition_id=f"unreported-view:speed={draw['speed_mps']:g}mps", success=str(int(kind in acquired)))
    if source == "F_pass":
        row.update(duration_seconds=str(acquired[kind]),
                   deadline_seconds=str((config["route_x"][1]-config["route_x"][0])/draw["speed_mps"]))
    return row


def collect_split(root, plan, split, draws, source_writer=None):
    folder = root/split
    folder.mkdir()
    config = config_for(plan, split)
    with ExitStack() as stack:
        A_file = stack.enter_context((folder/"A_inputs.jsonl").open("x", encoding="utf-8"))
        truth_file = stack.enter_context((folder/"B_labels.private.jsonl").open("x", encoding="utf-8"))
        trace_file = stack.enter_context(gzip.open(folder/"frames.private.jsonl.gz", "wt", encoding="utf-8"))
        for draw in draws:
            frames, acquired = sample_pass(config, draw)
            fine = {str(d): sample_pass(config, draw, d)[1] for d in (2, 4)}
            A = A_input(config, draw)
            memory, content = {}, items(config)
            for kind, at in acquired.items():
                store(memory, content[kind], at)
            before = snapshot({}, 0., 0)
            after = snapshot(memory, A["completed_at"], 0)
            W = world(content, A["completed_at"])
            truth = independent_label(after, W)
            put(A_file, A)
            put(truth_file, dict(trial_id=draw["trial_id"], split=split, K={k: truth[k] for k in "NI"},
                B_before=before, B_after=after, world_after=W, acquired_at=acquired,
                finer_acquired_at=fine, scene_id=draw["scene_id"], physical_object_id="physical-Q"))
            put(trace_file, dict(trial_id=draw["trial_id"], frames=frames, acquired_at=acquired))
            if source_writer is not None:
                for kind in "NI":
                    source_writer.record_trial(source_row(plan, draw, "p_acq", kind, acquired))
                    if kind in acquired:
                        source_writer.record_trial(source_row(plan, draw, "F_pass", kind, acquired))
    write_json(folder/"manifest.json", {"schema_version": 2, "origin": "simulation", "split": split,
        "plan_id": plan["plan_id"], "draws": len(draws), "scene_id": scene_id(config),
        "physical_object_id": "physical-Q", "physical_session": f"synthetic-seed-{plan['seed']}",
        "sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file()}})
    print(json.dumps({"collected": split, "unique_draws": len(draws)}), flush=True)


def evaluate_split(root, plan, split, parameters):
    folder = root/split
    # Persist predictions from A inputs BEFORE opening any B labels.
    predicted = {}
    fingerprint = RawKnowledgeEstimatorV2(parameters=parameters).fingerprint
    with (folder/"raw_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for A in read_jsonl(folder/"A_inputs.jsonl"):
            p = predict(A, parameters)
            predicted[A["trial_id"]] = p
            put(stream, dict(trial_id=A["trial_id"], raw=p, fingerprint=fingerprint))
    draws = {r["trial_id"]: r for r in read_jsonl(root/"allocation.private.jsonl") if r["split"] == split}
    groups = defaultdict(list)
    for truth in read_jsonl(folder/"B_labels.private.jsonl"):
        draw, p = draws[truth["trial_id"]], predicted[truth["trial_id"]]
        factors = [("all", "all"), ("speed_mps", draw["speed_mps"]),
            ("stratum", draw["stratum_id"]), ("walls", draw["walls"])]
        for factor, value in factors:
            for kind in "NI":
                groups[kind, factor, value].append((p[kind], truth["K"][kind],
                    any((kind in truth["finer_acquired_at"][d]) != bool(truth["K"][kind]) for d in ("2", "4"))))
    rows = []
    for (kind, factor, value), values in groups.items():
        known = [(p, y) for p, y, _ in values if p is not None]
        rows.append(dict(split=split, kind=kind, factor=factor, value=value, unique_draws=len(values),
            known=len(known), missing=len(values)-len(known), resolution_sensitive=sum(b for _, _, b in values),
            actual_K_rate=sum(y for _, y, _ in values)/len(values),
            mean_prediction=sum(p for p, _ in known)/len(known) if known else None,
            actual_rate_on_known=sum(y for _, y in known)/len(known) if known else None,
            brier=sum((p-y)**2 for p, y in known)/len(known) if known else None))
    write_csv(folder/"metrics.csv", rows)
    return [r for r in rows if r["factor"] in {"all", "speed_mps"}]


def validate_collection(root):
    """Fail closed on omissions, labels, physical duplicates and A-input leaks."""
    root = Path(root)
    if (root/"manifest.json").exists():
        full = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        if not full["complete"]:
            raise ValueError("collection is incomplete")
        for name, expected in full["sha256"].items():
            path = (root/name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != expected:
                raise ValueError(f"collection checksum/path mismatch: {name}")
    plan = json.loads((root/"collection_plan.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    allocation = list(read_jsonl(root/"allocation.private.jsonl"))
    counts = check_allocations(plan, allocation)
    draws = {r["trial_id"]: r for r in allocation}
    checks, issues = {}, []
    for split in (*REFERENCE_SPLITS, *OOD_SPLITS):
        folder = root/split
        manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if (set(manifest["sha256"]) != {"A_inputs.jsonl", "B_labels.private.jsonl", "frames.private.jsonl.gz"}
                or manifest["plan_id"] != plan["plan_id"] or manifest["split"] != split
                or manifest["scene_id"] != scene_id(config_for(plan, split))):
            raise ValueError("split manifest contract mismatch")
        for name, expected in manifest["sha256"].items():
            if sha(folder/name) != expected:
                raise ValueError(f"checksum mismatch: {split}/{name}")
        config = config_for(plan, split)
        inputs = list(read_jsonl(folder/"A_inputs.jsonl"))
        labels_list = list(read_jsonl(folder/"B_labels.private.jsonl"))
        labels = {r["trial_id"]: r for r in labels_list}
        ids = {r["trial_id"] for r in inputs}
        expected_ids = {r["trial_id"] for r in allocation if r["split"] == split}
        if ids != expected_ids or len(ids) != len(inputs) or set(labels) != ids or len(labels_list) != len(ids):
            raise ValueError("missing/duplicate/unpaired draws")
        for A in inputs:
            validate_A_input(plan, config, draws[A["trial_id"]], A)
        frame_count, trace_ids, sensitive = 0, set(), Counter()
        with gzip.open(folder/"frames.private.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                trace = json.loads(line)
                key = trace["trial_id"]
                if key in trace_ids or key not in ids:
                    raise ValueError("duplicate/unallocated trace")
                trace_ids.add(key)
                draw, truth = draws[key], labels[key]
                if truth["scene_id"] != draw["scene_id"] or truth["physical_object_id"] != "physical-Q" or truth["split"] != split:
                    raise ValueError("false physical grouping metadata")
                acquired, dwell, previous = {}, 0., False
                walls = config["wall_rectangles"] if draw["walls"] else []
                frames = trace["frames"]
                duration = (config["route_x"][1]-config["route_x"][0])/draw["speed_mps"]
                expected_n = floor((duration-draw["frame_phase_seconds"])/config["dt_seconds"]+1e-9)+1
                if len(frames) != expected_n:
                    raise ValueError("missing sensor frame")
                for index, f in enumerate(frames):
                    t = round(draw["frame_phase_seconds"]+index*config["dt_seconds"], 10)
                    pose = (config["route_x"][0]+draw["speed_mps"]*t, draw["offset"], radians(draw["heading_degrees"]))
                    visible_now = seen(pose, config["object_xy"], config, walls)
                    dwell = dwell+config["dt_seconds"] if index and visible_now and previous else 0.
                    for kind in "NI":
                        if visible_now and dwell+1e-9 >= config["storage_dwell_seconds"][kind] and kind not in acquired:
                            acquired[kind] = t
                    if abs(f["time"]-t) > 1e-9 or any(abs(a-b) > 1e-9 for a, b in zip(f["pose"], pose)) or f["visible"] != visible_now or abs(f["contiguous_visible_seconds"]-dwell) > 1e-8 or f["stored"] != sorted(acquired):
                        raise ValueError("independent frame oracle mismatch")
                    previous = visible_now
                frame_count += len(frames)
                if acquired != trace["acquired_at"] or acquired != truth["acquired_at"]:
                    raise ValueError("acquisition trace/label mismatch")
                if truth["B_before"]["items"] or any(truth["K"][k] != int(k in acquired) for k in "NI"):
                    raise ValueError("initial memory or label mismatch")
                recomputed = independent_label(truth["B_after"], truth["world_after"])
                if any(recomputed[k] != truth["K"][k] for k in "NI"):
                    raise ValueError("B storage/validity contract mismatch")
                for kind in "NI":
                    if any((kind in truth["finer_acquired_at"][d]) != bool(truth["K"][kind]) for d in ("2", "4")):
                        sensitive[kind] += 1
        if trace_ids != ids:
            raise ValueError("missing traces")
        checks[split] = dict(unique_draws=len(ids), strata=len(cells(plan, split)),
            per_cell=plan["per_cell"], frames_checked=frame_count,
            resolution_sensitive_N=sensitive["N"], resolution_sensitive_I=sensitive["I"],
            scene_id=scene_id(config), physical_object_id="physical-Q")
    _, trials, _ = load_dataset(root/"source_trials", allow_simulation=True)
    expected_trials = []
    train_labels = {r["trial_id"]: r for r in read_jsonl(root/"train"/"B_labels.private.jsonl")}
    for draw in allocation:
        if draw["split"] == "train":
            acquired = train_labels[draw["trial_id"]]["acquired_at"]
            for kind in "NI":
                expected_trials.append(source_row(plan, draw, "p_acq", kind, acquired))
                if kind in acquired:
                    expected_trials.append(source_row(plan, draw, "F_pass", kind, acquired))
    if trials != expected_trials:
        raise ValueError("source trials differ from actual train opportunities/labels")
    freeze = json.loads((root/"freeze_before_evaluation.json").read_text(encoding="utf-8"))
    if sha(root/"raw_parameters.json") != freeze["parameters_sha256"] or freeze["plan_id"] != plan["plan_id"]:
        raise ValueError("frozen estimator parameters changed")
    artifact = json.loads((root/"raw_parameters.json").read_text(encoding="utf-8"))
    if fit_sources(root/"source_trials", allow_simulation=True, fit_split="train") != artifact:
        raise ValueError("parameters are not train-only empirical rates/CDFs")
    if freeze["allocation_sha256"] != sha(root/"allocation.private.jsonl"):
        raise ValueError("allocation changed after fitting")
    for name, expected in freeze["code_sha256"].items():
        if sha(Path(__file__).parents[1]/name) != expected or sha(root/"implementation"/name) != expected:
            raise ValueError("collection implementation changed during experiment")
    if RawKnowledgeEstimatorV2(parameters=ParameterStore(artifact, allow_simulation=True)).fingerprint != freeze["estimator_fingerprint"]:
        raise ValueError("raw estimator changed after freeze")
    parameters = ParameterStore(artifact, allow_simulation=True)
    for split in ("validation", "test", *OOD_SPLITS):
        A_records = {r["trial_id"]: r for r in read_jsonl(root/split/"A_inputs.jsonl")}
        predictions = list(read_jsonl(root/split/"raw_predictions.jsonl"))
        if len(predictions) != len(A_records) or {r["trial_id"] for r in predictions} != set(A_records):
            raise ValueError("prediction coverage mismatch")
        for prediction in predictions:
            if prediction["fingerprint"] != freeze["estimator_fingerprint"] or prediction["raw"] != predict(A_records[prediction["trial_id"]], parameters):
                raise ValueError("raw prediction is not reproducible from A inputs only")
    if (root/"calibration"/"raw_predictions.jsonl").exists():
        raise ValueError("reserved calibration split was used")
    lock = json.loads((root/"freeze_before_test.json").read_text(encoding="utf-8"))
    if lock["parameters_sha256"] != freeze["parameters_sha256"] or lock["validation_metrics_sha256"] != sha(root/"validation"/"metrics.csv"):
        raise ValueError("test freeze mismatch")
    return dict(passed=True, issues=issues, split_counts=counts, checks=checks,
        reference_balance="identical cell counts and continuous within-cell sampling laws in all four splits",
        independence="independent geometry/phase draws; same physical-Q, map and model, not independent real environments")


def run(output, plan=None):
    plan = plan or build_plan()
    validate_plan(plan)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"collection_plan.json", plan)
    allocation = list(allocations(plan))
    counts = check_allocations(plan, allocation)
    with (root/"allocation.private.jsonl").open("x", encoding="utf-8") as stream:
        for row in allocation:
            put(stream, row)
    package = Path(__file__).parents[1]
    code = {}
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        if "data" in relative.parts or "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        destination = root/"implementation"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        code[relative.as_posix()] = sha(path)
    write_json(root/"allocation_frozen_before_labels.json", dict(plan_id=plan["plan_id"],
        allocation_sha256=sha(root/"allocation.private.jsonl"), counts=counts, code_sha256=code))
    source = MeasurementRecorder(root/"source_trials", protocol_id=plan["config"]["protocol_id"],
        episode_id="source-start", origin="simulation")
    evidence = source.root/"train_frames.private.jsonl.gz"
    evidence.touch()
    source.add_evidence("sim-evidence", evidence.name, description="Independent train geometry/phase draws; ideal 2D acquisition only")
    grouped = {s: [d for d in allocation if d["split"] == s] for s in (*REFERENCE_SPLITS, *OOD_SPLITS)}
    collect_split(root, plan, "train", grouped["train"], source)
    shutil.copyfile(root/"train"/"frames.private.jsonl.gz", evidence)
    write_json(source.root/"collection_plan.json", plan)
    source.finalize()
    artifact = fit_sources(source.root, allow_simulation=True, fit_split="train")
    write_json(root/"raw_parameters.json", artifact)
    parameters = ParameterStore(artifact, allow_simulation=True)
    frozen = dict(plan_id=plan["plan_id"], parameters_sha256=sha(root/"raw_parameters.json"),
        estimator_fingerprint=RawKnowledgeEstimatorV2(parameters=parameters).fingerprint,
        allocation_sha256=sha(root/"allocation.private.jsonl"), code_sha256=code,
        training_split="train", calibration="reserved, no score fitting", source="synthetic geometry counts only")
    write_json(root/"freeze_before_evaluation.json", frozen)
    collect_split(root, plan, "calibration", grouped["calibration"])
    collect_split(root, plan, "validation", grouped["validation"])
    results = {"validation": evaluate_split(root, plan, "validation", parameters)}
    write_json(root/"freeze_before_test.json", {**frozen,
        "validation_metrics_sha256": sha(root/"validation"/"metrics.csv"),
        "selection": "none; no parameter or code changes after validation"})
    for split in ("test", *OOD_SPLITS):
        collect_split(root, plan, split, grouped[split])
        results[split] = evaluate_split(root, plan, split, parameters)
    quality = validate_collection(root)
    write_json(root/"quality_report.json", quality)
    write_json(root/"summary.json", dict(plan_id=plan["plan_id"], calibration="OFF and reserved",
        quality=quality, precision_target=plan["precision_target"], metrics=results,
        limits=plan["hardware_status"], policy="No rule/PPO or communication; acquisition data design only"))
    write_json(root/"manifest.json", dict(schema_version=2, origin="simulation", complete=True,
        plan_id=plan["plan_id"], sha256={p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}))
    return dict(draws=counts, quality_passed=quality["passed"], precision_target=plan["precision_target"],
                test=[r for r in results["test"] if r["factor"] == "all"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--epsilon", type=float, default=.10, help="Declared synthetic mean precision target, not sensor success rate")
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate_collection(args.output) if args.validate_only else run(args.output, build_plan(seed=args.seed, epsilon=args.epsilon, alpha=args.alpha))
    print(json.dumps(result, indent=2))
