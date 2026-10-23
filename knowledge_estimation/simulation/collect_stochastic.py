"""Balanced collection in the stochastic world, with per-scene true probabilities.

python -m knowledge_estimation.simulation.collect_stochastic --output knowledge_estimation/data/stochastic_medium_v1 --profile medium

Same label-blind allocation as collect_designed. Each scene is one physical
pass (rollout 0) for labels and training; evaluation splits additionally get
R independent re-rollouts of the same planned scene to estimate p*, the true
acquisition probability under the declared noise profile. p* never enters the
estimator or the train fit. All numbers concern a declared synthetic world.
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import gzip
import json
from multiprocessing import Pool
from pathlib import Path
import shutil

from ..calibrate import fit_sources
from ..collect import MeasurementRecorder
from ..measurement import write_json, read_jsonl, load_dataset
from ..parameters import ParameterStore
from ..probabilistic import RawKnowledgeEstimatorV2
from .analyze import write_csv
from .audit_acquisition import independent_label
from .collect_designed import sha, put, A_input, predict, validate_A_input, source_row, config_for, sample_pass
from .collection_design import (build_plan, validate_plan, allocations, check_allocations,
                                scene_id, REFERENCE_SPLITS, OOD_SPLITS, digest)
from .run import items, store, snapshot, world
from .stochastic_world import rollout, profile as load_profile


PSTAR_SPLITS = ("validation", "test", *OOD_SPLITS)


WORLD_OVERRIDABLE = ("camera_fov_degrees", "camera_range_m", "camera_min_depth_m", "storage_dwell_seconds")


def build_stochastic_plan(base, profile_name, rollouts, world_override=None):
    if type(rollouts) is not int or rollouts < 2:
        raise ValueError("at least two p* rollouts are required")
    prof = load_profile(profile_name)
    plan = dict(schema_version=1, origin="simulation", base_plan_id=base["plan_id"], profile=prof,
        pstar_rollouts=rollouts, pstar_splits=list(PSTAR_SPLITS), label_rollout=0,
        seed_rule="int(sha256([stochastic_plan_id, trial_id, rollout]))%2**63",
        p_star="mean acquisition over rollouts 1..R of the same planned scene; label rollout 0 excluded so p* and K are independent",
        estimator_inputs="unchanged A_input allowlist; p*, noise state and executed pose are truth-only")
    if world_override:
        # True sensor differs from the nominal values A keeps (misspecification test).
        if set(world_override)-set(WORLD_OVERRIDABLE):
            raise ValueError(f"world override may only change {WORLD_OVERRIDABLE}")
        plan["world_override"] = world_override
        plan["world_override_scope"] = "applied to the simulated world only; A inputs and A's nominal camera stay unchanged"
        # Same noise draws as the nominal world: differences come from the sensor only.
        plan["seed_plan_id"] = build_stochastic_plan(base, profile_name, rollouts)["stochastic_plan_id"]
    plan["stochastic_plan_id"] = digest(plan)
    return plan


def world_config(config, plan):
    return {**config, **plan.get("world_override", {})}


def validate_stochastic_plan(base, plan):
    validate_plan(base)
    if plan["base_plan_id"] != base["plan_id"] or plan["stochastic_plan_id"] != digest({k: v for k, v in plan.items() if k != "stochastic_plan_id"}):
        raise ValueError("stochastic plan changed after freezing")


def seed_for(plan, trial_id, index):
    return int(digest([plan.get("seed_plan_id", plan["stochastic_plan_id"]), trial_id, index]), 16) % 2**63


def simulate(job):
    """Pool worker: label rollout with frames, plus p* counts when requested."""
    config, draw, prof, seeds = job
    frames, acquired = rollout(config, draw, prof, seeds[0])
    counts = {"N": 0, "I": 0}
    for seed in seeds[1:]:
        for kind in rollout(config, draw, prof, seed, keep_frames=False)[1]:
            counts[kind] += 1
    return frames, acquired, counts


def collect_split(root, base, plan, split, draws, pool, source_writer=None):
    folder = root/split
    folder.mkdir()
    config = config_for(base, split)
    R = plan["pstar_rollouts"] if split in PSTAR_SPLITS else 0
    true_sensor = world_config(config, plan)
    jobs = [(true_sensor, d, plan["profile"], [seed_for(plan, d["trial_id"], i) for i in range(R+1)]) for d in draws]
    with ExitStack() as stack:
        A_file = stack.enter_context((folder/"A_inputs.jsonl").open("x", encoding="utf-8"))
        truth_file = stack.enter_context((folder/"B_labels.private.jsonl").open("x", encoding="utf-8"))
        trace_file = stack.enter_context(gzip.open(folder/"frames.private.jsonl.gz", "wt", encoding="utf-8"))
        for draw, (frames, acquired, counts) in zip(draws, pool.imap(simulate, jobs, chunksize=16)):
            A = A_input(config, draw)
            memory, content = {}, items(config)
            for kind, at in acquired.items():
                store(memory, content[kind], at)
            after = snapshot(memory, A["completed_at"], 0)
            W = world(content, A["completed_at"])
            truth = independent_label(after, W)
            put(A_file, A)
            put(truth_file, dict(trial_id=draw["trial_id"], split=split, K={k: truth[k] for k in "NI"},
                B_before=snapshot({}, 0., 0), B_after=after, world_after=W, acquired_at=acquired,
                p_star={k: counts[k]/R for k in "NI"} if R else None, p_star_rollouts=R,
                scene_id=draw["scene_id"], physical_object_id="physical-Q"))
            put(trace_file, dict(trial_id=draw["trial_id"], frames=frames, acquired_at=acquired))
            if source_writer is not None:
                for kind in "NI":
                    source_writer.record_trial(source_row(base, draw, "p_acq", kind, acquired))
                    if kind in acquired:
                        source_writer.record_trial(source_row(base, draw, "F_pass", kind, acquired))
    write_json(folder/"manifest.json", {"schema_version": 1, "origin": "simulation", "split": split,
        "stochastic_plan_id": plan["stochastic_plan_id"], "profile": plan["profile"]["name"], "draws": len(draws),
        "scene_id": scene_id(config), "sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file()}})
    print(json.dumps({"collected": split, "unique_draws": len(draws), "pstar_rollouts": R}), flush=True)


def auroc(pairs):
    """Rank AUROC with ties counted half; None without both classes."""
    positives = sorted(p for p, y in pairs if y)
    negatives = sorted(p for p, y in pairs if not y)
    if not positives or not negatives:
        return None
    wins, i, j = 0., 0, 0
    for p in positives:
        while i < len(negatives) and negatives[i] < p:
            i += 1
        j = i
        while j < len(negatives) and negatives[j] == p:
            j += 1
        wins += i+(j-i)/2
    return wins/(len(positives)*len(negatives))


def factors(base, draw):
    _, d, h, _ = draw["stratum_id"].split("-")
    bins = base["ood_distance_bins_m"] if draw["split"] == "ood_distance" else base["distance_bins_m"]
    return [("all", "all"), ("speed_mps", draw["speed_mps"]), ("distance_bin_m", bins[int(d[1:])]),
            ("heading_bin_degrees", base["heading_bins_degrees"][int(h[1:])]), ("walls", draw["walls"])]


def metrics(rows, split, kind, factor, value):
    known = [r for r in rows if r["b"] is not None]
    with_star = [r for r in rows if r["p_star"] is not None]
    mean = lambda xs: sum(xs)/len(xs) if xs else None
    return dict(split=split, kind=kind, factor=factor, value=value, unique_draws=len(rows),
        known=len(known), missing=len(rows)-len(known),
        actual_K_rate=mean([r["K"] for r in rows]),
        mean_prediction=mean([r["b"] for r in known]),
        brier=mean([(r["b"]-r["K"])**2 for r in known]),
        auroc=auroc([(r["b"], r["K"]) for r in known]),
        mean_p_star=mean([r["p_star"] for r in with_star]),
        # p* is the best achievable forecast; its Brier is the irreducible floor.
        brier_of_p_star=mean([(r["p_star"]-r["K"])**2 for r in with_star]),
        aleatoric_floor=mean([r["p_star"]*(1-r["p_star"]) for r in with_star]),
        mse_to_p_star=mean([(r["b"]-r["p_star"])**2 for r in known if r["p_star"] is not None]),
        uncertain_scene_fraction=mean([.05 < r["p_star"] < .95 for r in with_star]))


def evaluate_split(root, base, split, parameters):
    folder = root/split
    predicted = {}
    fingerprint = RawKnowledgeEstimatorV2(parameters=parameters).fingerprint
    with (folder/"raw_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for A in read_jsonl(folder/"A_inputs.jsonl"):
            predicted[A["trial_id"]] = p = predict(A, parameters)
            put(stream, dict(trial_id=A["trial_id"], raw=p, fingerprint=fingerprint))
    draws = {r["trial_id"]: r for r in read_jsonl(root/"allocation.private.jsonl") if r["split"] == split}
    groups = defaultdict(list)
    for truth in read_jsonl(folder/"B_labels.private.jsonl"):
        draw, b = draws[truth["trial_id"]], predicted[truth["trial_id"]]
        for factor, value in factors(base, draw):
            for kind in "NI":
                groups[kind, factor, json.dumps(value)].append(dict(b=b[kind], K=truth["K"][kind],
                    p_star=truth["p_star"][kind] if truth["p_star"] else None))
    rows = [metrics(v, split, kind, factor, json.loads(value)) for (kind, factor, value), v in groups.items()]
    write_csv(folder/"metrics.csv", rows)
    return [r for r in rows if r["factor"] == "all"]


def validate_collection(root, workers=None):
    """Fail closed: checksums, A-input contract, re-simulated labels and p*."""
    root = Path(root)
    full = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in full["sha256"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"collection checksum/path mismatch: {name}")
    base = json.loads((root/"collection_plan.json").read_text(encoding="utf-8"))
    plan = json.loads((root/"stochastic_plan.json").read_text(encoding="utf-8"))
    validate_stochastic_plan(base, plan)
    allocation = list(read_jsonl(root/"allocation.private.jsonl"))
    counts = check_allocations(base, allocation)
    draws = {r["trial_id"]: r for r in allocation}
    checks = {}
    with Pool(workers) as pool:
        for split in (*REFERENCE_SPLITS, *OOD_SPLITS):
            config = config_for(base, split)
            inputs = list(read_jsonl(root/split/"A_inputs.jsonl"))
            labels = {r["trial_id"]: r for r in read_jsonl(root/split/"B_labels.private.jsonl")}
            expected = [d for d in allocation if d["split"] == split]
            if [A["trial_id"] for A in inputs] != [d["trial_id"] for d in expected] or set(labels) != {d["trial_id"] for d in expected}:
                raise ValueError("missing/duplicate/unpaired draws")
            for A in inputs:
                validate_A_input(base, config, draws[A["trial_id"]], A)
            # Re-simulate every label rollout and p* for a fixed prefix of scenes.
            R = plan["pstar_rollouts"] if split in PSTAR_SPLITS else 0
            true_sensor = world_config(config, plan)
            jobs = [(true_sensor, d, plan["profile"], [seed_for(plan, d["trial_id"], i) for i in range(R+1 if n < 50 else 1)])
                    for n, d in enumerate(expected)]
            for n, (d, (_, acquired, c)) in enumerate(zip(expected, pool.imap(simulate, jobs, chunksize=16))):
                truth = labels[d["trial_id"]]
                if acquired != truth["acquired_at"] or any(truth["K"][k] != int(k in acquired) for k in "NI"):
                    raise ValueError("label is not the declared rollout 0")
                if any(independent_label(truth["B_after"], truth["world_after"])[k] != truth["K"][k] for k in "NI"):
                    raise ValueError("B storage/validity contract mismatch")
                if R and n < 50 and truth["p_star"] != {k: c[k]/R for k in "NI"}:
                    raise ValueError("p* not reproducible")
            # The world code path must still collapse to the deterministic one.
            # The legacy reference only exists for the free-pointing design.
            equivalence = 0
            if base.get("camera_mount", "free") == "free":
                none = load_profile("none")
                for d in expected[:100]:
                    if rollout(true_sensor, d, none, 0)[1] != sample_pass(true_sensor, d)[1]:
                        raise ValueError("zero-noise world differs from deterministic reference")
                equivalence = min(len(expected), 100)
            checks[split] = dict(unique_draws=len(expected), resimulated_labels=len(expected),
                                 resimulated_p_star=min(len(expected), 50) if R else 0, zero_noise_equivalence=equivalence)
    _, trials, _ = load_dataset(root/"source_trials", allow_simulation=True)
    train_labels = {r["trial_id"]: r for r in read_jsonl(root/"train"/"B_labels.private.jsonl")}
    expected_trials = []
    for draw in allocation:
        if draw["split"] == "train":
            acquired = train_labels[draw["trial_id"]]["acquired_at"]
            for kind in "NI":
                expected_trials.append(source_row(base, draw, "p_acq", kind, acquired))
                if kind in acquired:
                    expected_trials.append(source_row(base, draw, "F_pass", kind, acquired))
    if trials != expected_trials:
        raise ValueError("source trials differ from train label rollouts")
    artifact = json.loads((root/"raw_parameters.json").read_text(encoding="utf-8"))
    freeze = json.loads((root/"freeze_before_evaluation.json").read_text(encoding="utf-8"))
    if fit_sources(root/"source_trials", allow_simulation=True, fit_split="train") != artifact or sha(root/"raw_parameters.json") != freeze["parameters_sha256"]:
        raise ValueError("parameters are not frozen train-only rates")
    parameters = ParameterStore(artifact, allow_simulation=True)
    for split in ("validation", "test", *OOD_SPLITS):
        A_records = {r["trial_id"]: r for r in read_jsonl(root/split/"A_inputs.jsonl")}
        for prediction in read_jsonl(root/split/"raw_predictions.jsonl"):
            if prediction["raw"] != predict(A_records[prediction["trial_id"]], parameters):
                raise ValueError("raw prediction is not reproducible from A inputs only")
    if (root/"calibration"/"raw_predictions.jsonl").exists():
        raise ValueError("reserved calibration split was used")
    return dict(passed=True, split_counts=counts, checks=checks)


def copy_implementation(root):
    package = Path(__file__).parents[1]
    code = {}
    for path in [*package.rglob("*.py"), Path(__file__).with_name("noise_profiles.json")]:
        relative = path.relative_to(package)
        if {"data", "tests", "__pycache__"} & set(relative.parts):
            continue
        destination = root/"implementation"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        code[relative.as_posix()] = sha(path)
    return code


def run(output, base=None, *, profile_name="medium", rollouts=64, workers=None, world_override=None):
    base = base or build_plan()
    plan = build_stochastic_plan(base, profile_name, rollouts, world_override)
    validate_stochastic_plan(base, plan)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"collection_plan.json", base)
    write_json(root/"stochastic_plan.json", plan)
    allocation = list(allocations(base))
    counts = check_allocations(base, allocation)
    with (root/"allocation.private.jsonl").open("x", encoding="utf-8") as stream:
        for row in allocation:
            put(stream, row)
    code = copy_implementation(root)
    write_json(root/"allocation_frozen_before_labels.json", dict(stochastic_plan_id=plan["stochastic_plan_id"],
        allocation_sha256=sha(root/"allocation.private.jsonl"), counts=counts, code_sha256=code))
    grouped = {s: [d for d in allocation if d["split"] == s] for s in (*REFERENCE_SPLITS, *OOD_SPLITS)}
    with Pool(workers) as pool:
        source = MeasurementRecorder(root/"source_trials", protocol_id=base["config"]["protocol_id"],
            episode_id="source-start", origin="simulation")
        evidence = source.root/"train_frames.private.jsonl.gz"
        evidence.touch()
        source.add_evidence("sim-evidence", evidence.name,
            description=f"Train label rollouts in stochastic profile {profile_name}; synthetic 2D only")
        collect_split(root, base, plan, "train", grouped["train"], pool, source)
        shutil.copyfile(root/"train"/"frames.private.jsonl.gz", evidence)
        write_json(source.root/"collection_plan.json", base)
        source.finalize()
        artifact = fit_sources(source.root, allow_simulation=True, fit_split="train")
        write_json(root/"raw_parameters.json", artifact)
        parameters = ParameterStore(artifact, allow_simulation=True)
        frozen = dict(stochastic_plan_id=plan["stochastic_plan_id"], parameters_sha256=sha(root/"raw_parameters.json"),
            estimator_fingerprint=RawKnowledgeEstimatorV2(parameters=parameters).fingerprint,
            allocation_sha256=sha(root/"allocation.private.jsonl"), code_sha256=code,
            training_split="train", calibration="reserved, no score fitting")
        write_json(root/"freeze_before_evaluation.json", frozen)
        collect_split(root, base, plan, "calibration", grouped["calibration"], pool)
        results = {}
        for split in ("validation", "test", *OOD_SPLITS):
            collect_split(root, base, plan, split, grouped[split], pool)
            results[split] = evaluate_split(root, base, split, parameters)
    write_json(root/"manifest.json", dict(schema_version=1, origin="simulation", complete=True,
        sha256={p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}))
    quality = validate_collection(root, workers)
    write_json(root/"quality_report.json", quality)
    write_json(root/"summary.json", dict(stochastic_plan_id=plan["stochastic_plan_id"], profile=profile_name,
        pstar_rollouts=rollouts, calibration="OFF and reserved", estimator="unchanged L0 raw (speed-only context)",
        quality=quality, metrics=results, limits=base["hardware_status"] | {"execution_noise": f"declared profile {profile_name}"}))
    return dict(draws=counts, quality_passed=quality["passed"], test=results["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026102301)
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--camera-mount", choices=("free", "forward"), default="free",
                        help="free: legacy camera heading sweep; forward: camera fixed to travel direction")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ood-distance", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                        help="OOD closest-distance bin in metres (default: plan default [2, 2.5])")
    parser.add_argument("--world-override", type=json.loads, default=None,
                        help='JSON, e.g. {"camera_fov_degrees": 80}; true sensor only, A keeps nominal values')
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (validate_collection(args.output, args.workers) if args.validate_only else
              run(args.output, build_plan(seed=args.seed, camera_mount=args.camera_mount,
                                          ood_distance_bins_m=[args.ood_distance] if args.ood_distance else None),
                  profile_name=args.profile, rollouts=args.rollouts, workers=args.workers,
                  world_override=args.world_override))
    print(json.dumps(result, indent=2, default=str))
