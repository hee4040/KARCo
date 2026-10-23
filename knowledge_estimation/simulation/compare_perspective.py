"""Information-tier comparison of the perspective-taking acquisition model.

python -m knowledge_estimation.simulation.compare_perspective knowledge_estimation/data/stochastic_medium_v1 --output knowledge_estimation/data/perspective_medium_v1

Tiers of what A knows about B's pass (all include planned speed/route):
  L0      speed-only frequency table (existing raw estimator, read from the dataset)
  L1      + planned route line and A's map; camera heading unknown
          (with a forward mount this is an ablation: A ignores the known mount)
  L2      + planned camera heading (forward mount: route direction)
  L3@hz   + B pose reports at hz (executed pose sent by B, assumed self-localised exactly)
Raw scores are written for every split before any label is read. Each tier gets
one monotone (PAV) map fitted on the calibration split only; no tier, sample
count or assumption is selected on validation/test. p* is used only to score.
"""

import argparse
from collections import defaultdict
import gzip
import json
from math import cos, degrees, sin
from multiprocessing import Pool
from pathlib import Path

from ..calibrate import monotone_map
from ..measurement import read_jsonl, write_json
from ..probabilistic.perspective import ASSUMPTIONS, acquisition_probability
from .analyze import write_csv
from .collect_designed import config_for, put, sha
from .collect_stochastic import factors, metrics, validate_collection
from .collection_design import OOD_SPLITS, digest, route_of


TIERS = {"L1": (False, None), "L2": (True, None), "L3@0.5hz": (True, .5),
         "L3@1hz": (True, 1.), "L3@2hz": (True, 2.), "L3@5hz": (True, 5.)}
SPLITS = ("calibration", "validation", "test", *OOD_SPLITS)
REPORT_BYTES = 32  # declared: float64 time, x, y, yaw; serializer not measured


def reports_from(frames, hz):
    """Executed pose of the first sensor frame at/after each report time."""
    if hz is None:
        return []
    out, k = [], 0
    for frame in frames:
        if frame["time"]+1e-9 >= k/hz:
            out.append((frame["time"], *frame["pose"]))
            while k/hz <= frame["time"]+1e-9:
                k += 1
    return out


def view_for(config, A, draw, heading_known, reports):
    """Only plan, map, nominal camera and declared dwell; never noise state or labels."""
    N = next(i for i in A["approved_items"] if i["kind"] == "N")
    origin, psi, camera_rel, length = route_of(config, draw)
    return dict(route_origin=list(origin), route_direction_deg=degrees(psi), route_length_m=length,
        planned_speed_mps=A["planned_speed_mps"],
        object_xy=N["value"][:2], walls=config["wall_rectangles"] if draw["walls"] else [],
        camera=dict(fov_deg=config["camera_fov_degrees"], z_min_m=config["camera_min_depth_m"],
                    z_max_m=config["camera_range_m"], dt_s=config["dt_seconds"]),
        dwell_seconds=config["storage_dwell_seconds"],
        planned_heading_deg=degrees(psi+camera_rel) if heading_known else None, pose_reports=reports)


def _predict(job):
    view, samples, seed, assumption = job
    return acquisition_probability(view, samples=samples, seed=seed, assumption=assumption)


def apply_map(mapping, score):
    if not mapping["support"][0] <= score <= mapping["support"][1]:
        return None
    return next(b["value"] for b in mapping["blocks"] if score <= b["upper"])


def run(dataset, output, *, samples=64, workers=None, validate=True, assumption="constant", tiers=None):
    dataset, root = Path(dataset), Path(output)
    TIERS_RUN = {t: TIERS[t] for t in (tiers or TIERS)}
    A = ASSUMPTIONS[assumption]
    quality = validate_collection(dataset, workers) if validate else {"passed": "not re-run"}
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads((dataset/"collection_plan.json").read_text(encoding="utf-8"))
    stochastic = json.loads((dataset/"stochastic_plan.json").read_text(encoding="utf-8"))
    draws = {r["trial_id"]: r for r in read_jsonl(dataset/"allocation.private.jsonl")}
    write_json(root/"design.json", dict(dataset=str(dataset), dataset_manifest_sha256=sha(dataset/"manifest.json"),
        stochastic_plan_id=stochastic["stochastic_plan_id"], profile_hidden_from_model=stochastic["profile"]["name"],
        tiers={k: {"planned_heading": h, "report_hz": hz} for k, (h, hz) in TIERS_RUN.items()},
        samples=samples, A_assumption_name=assumption, A_assumption=A,
        world_override_hidden_from_model=stochastic.get("world_override"), report_bytes_declared=REPORT_BYTES,
        calibration="one PAV map per tier and item, calibration split rollout-0 labels only",
        selection="none; every tier reported"))
    # 1. Raw scores from A views, persisted before labels are opened.
    reports_per_pass = defaultdict(list)
    with Pool(workers) as pool:
        for split in SPLITS:
            config = config_for(base, split)
            inputs = list(read_jsonl(dataset/split/"A_inputs.jsonl"))
            with gzip.open(dataset/split/"frames.private.jsonl.gz", "rt", encoding="utf-8") as stream:
                frames = {r["trial_id"]: r["frames"] for r in map(json.loads, stream)}
            for tier, (heading_known, hz) in TIERS_RUN.items():
                jobs = []
                for record in inputs:
                    reports = reports_from(frames[record["trial_id"]], hz)
                    reports_per_pass[tier].append(len(reports))
                    # Seeds depend only on tier/trial: paired comparison across assumptions.
                    jobs.append((view_for(config, record, draws[record["trial_id"]], heading_known, reports), samples,
                                 int(digest([tier, record["trial_id"]]), 16) % 2**63, A))
                folder = root/tier
                folder.mkdir(exist_ok=True)
                with (folder/f"{split}_raw.jsonl").open("x", encoding="utf-8") as stream:
                    for record, p in zip(inputs, pool.imap(_predict, jobs, chunksize=8)):
                        put(stream, dict(trial_id=record["trial_id"], raw=p))
                print(json.dumps({"scored": split, "tier": tier, "n": len(jobs)}), flush=True)
    write_json(root/"freeze_before_labels.json", {f"{t}/{s}": sha(root/t/f"{s}_raw.jsonl") for t in TIERS_RUN for s in SPLITS})
    communication = {t: dict(mean_reports_per_pass=sum(v)/len(v), declared_bytes_per_pass=REPORT_BYTES*sum(v)/len(v))
                     for t, v in reports_per_pass.items()}
    headline = evaluate(dataset, root, list(TIERS_RUN), root)
    write_json(root/"summary.json", dict(dataset_quality=quality, communication=communication, headline=headline))
    return dict(communication=communication, test=[r for r in headline if r["split"] == "test"])


def planned_context(config, A, draw):
    """What A can compute from the plan: closest route-object distance and speed."""
    origin, psi, _, _ = route_of(config, draw)
    N = next(i for i in A["approved_items"] if i["kind"] == "N")
    dx, dy = N["value"][0]-origin[0], N["value"][1]-origin[1]
    return dict(distance_m=abs(-dx*sin(psi)+dy*cos(psi)), speed_mps=A["planned_speed_mps"])


def evaluate(dataset, root, tiers, out):
    """Fit calibration on the calibration split, then score raw/calibrated/selective.

    selective (rule fixed before its first use): calibrated value when A's planned
    context (closest distance, speed) lies inside the range seen in the calibration
    split, else the raw score. The range comes from plans only, never labels.
    """
    base = json.loads((dataset/"collection_plan.json").read_text(encoding="utf-8"))
    draws = {r["trial_id"]: r for r in read_jsonl(dataset/"allocation.private.jsonl")}
    labels = {s: {r["trial_id"]: r for r in read_jsonl(dataset/s/"B_labels.private.jsonl")} for s in SPLITS}
    context = {}
    for split in SPLITS:
        config = config_for(base, split)
        for A in read_jsonl(dataset/split/"A_inputs.jsonl"):
            context[A["trial_id"]] = planned_context(config, A, draws[A["trial_id"]])
    calibration_context = [context[i] for i in labels["calibration"]]
    support = {k: [min(c[k] for c in calibration_context), max(c[k] for c in calibration_context)] for k in ("distance_m", "speed_mps")}

    def inside(trial_id):
        return all(lo-1e-9 <= context[trial_id][k] <= hi+1e-9 for k, (lo, hi) in support.items())

    maps = {}
    for tier in tiers:
        raw = {r["trial_id"]: r["raw"] for r in read_jsonl(root/tier/"calibration_raw.jsonl")}
        maps[tier] = {k: monotone_map([(raw[i][k], labels["calibration"][i]["K"][k]) for i in raw]) for k in "NI"}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"calibration_maps.json", dict(maps=maps, selective_support=support))
    rows = []
    for split in SPLITS[1:]:
        predictions = {"L0": {r["trial_id"]: {"raw": r["raw"]} for r in read_jsonl(dataset/split/"raw_predictions.jsonl")}}
        for tier in tiers:
            predictions[tier] = {}
            for r in read_jsonl(root/tier/f"{split}_raw.jsonl"):
                calibrated = {k: apply_map(maps[tier][k], r["raw"][k]) for k in "NI"}
                predictions[tier][r["trial_id"]] = dict(raw=r["raw"], calibrated=calibrated,
                    selective=calibrated if inside(r["trial_id"]) else r["raw"])
        for tier, by_id in predictions.items():
            for stage in ("raw", "calibrated", "selective"):
                if tier == "L0" and stage != "raw":
                    continue
                groups = defaultdict(list)
                for trial_id, truth in labels[split].items():
                    b = by_id[trial_id][stage]
                    for factor, value in factors(base, draws[trial_id]):
                        for kind in "NI":
                            groups[kind, factor, json.dumps(value)].append(dict(b=b[kind], K=truth["K"][kind], p_star=truth["p_star"][kind]))
                for (kind, factor, value), group in groups.items():
                    rows.append(dict(tier=tier, stage=stage, **metrics(group, split, kind, factor, json.loads(value))))
        rows.append(dict(tier="all", stage="selective_coverage", split=split, kind="", factor="inside_calibration_support",
                         value=sum(inside(i) for i in labels[split])/len(labels[split])))
    write_csv(out/"metrics.csv", rows)
    return [r for r in rows if r["factor"] in ("all", "inside_calibration_support")]


def rescore(dataset, root, out):
    """Re-evaluate frozen raw scores of an earlier run without recomputing them."""
    root = Path(root)
    design = json.loads((root/"design.json").read_text(encoding="utf-8"))
    frozen = json.loads((root/"freeze_before_labels.json").read_text(encoding="utf-8"))
    for key, expected in frozen.items():
        tier, split = key.split("/")
        if sha(root/tier/f"{split}_raw.jsonl") != expected:
            raise ValueError(f"frozen raw scores changed: {key}")
    headline = evaluate(Path(dataset), root, list(design["tiers"]), Path(out))
    write_json(Path(out)/"summary.json", dict(rescored_from=str(root), headline=headline))
    return headline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--assumption", choices=sorted(ASSUMPTIONS), default="constant")
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=None)
    parser.add_argument("--rescore-from", type=Path, default=None,
                        help="earlier output folder; re-evaluate its frozen raw scores into --output")
    args = parser.parse_args()
    if args.rescore_from:
        result = rescore(args.dataset, args.rescore_from, args.output)
    else:
        result = run(args.dataset, args.output, samples=args.samples, workers=args.workers,
                     validate=not args.skip_validate, assumption=args.assumption, tiers=args.tiers)
    print(json.dumps(result, indent=2, default=str))
