"""Label-blind balanced reference design; no success-rate generation."""

from copy import deepcopy
import hashlib
from itertools import product
import json
from math import ceil, log
from pathlib import Path
import random


REFERENCE_SPLITS = ("train", "calibration", "validation", "test")
OOD_SPLITS = ("ood_environment", "ood_distance", "ood_speed")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scene_id(config):
    # Shared real model identity is retained across reference splits.
    keys = ("object_xy", "sender_xy", "route_x", "wall_rectangles", "camera_fov_degrees",
            "camera_min_depth_m", "camera_range_m", "storage_dwell_seconds", "dt_seconds")
    return digest({k: config[k] for k in keys})


def build_plan(*, seed=20261023, epsilon=.10, alpha=.05):
    if type(seed) is not int or not 0 < epsilon < 1 or not 0 < alpha < 1:
        raise ValueError("explicit integer seed, epsilon and alpha in (0,1) required")
    config = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    config["protocol_id"] = "d455-balanced-acquisition-v2"
    config["sweeps"] = {}  # Existing legacy grids are deliberately unused.
    heading_bins = [[-180.+60*i, -120.+60*i] for i in range(6)]
    distance_bins = [[.5, 1.], [1., 1.5], [1.5, 2.]]
    speeds = [.5, 1., 2.]
    cells_per_speed = len(heading_bins)*len(distance_bins)*2
    total_cells = cells_per_speed*len(speeds)
    smallest_group_cells = total_cells//max(len(speeds), len(distance_bins), len(heading_bins), 2)
    # All one-factor speed/distance/heading/wall x kind means per split.
    # Independent bounded geometry draws
    # within equal-size strata satisfy Hoeffding even though not identically
    # distributed. This controls deviation from the declared simulator mean,
    # NOT error relative to physical robots, nor every cell's uncertainty.
    means = 2*(len(speeds)+len(distance_bins)+len(heading_bins)+2)
    minimum = ceil(log(2*means/alpha)/(2*epsilon**2))
    per_cell = ceil(minimum/smallest_group_cells)
    heldout = deepcopy(config)
    heldout.update(object_xy=[-.4, 2.3], sender_xy=[-.4, 4.3],
                   wall_rectangles=[[-.9, 1., -.2, 1.4], [.8, -4., 1.4, 4.]])
    plan = {"schema_version": 2, "origin": "simulation", "seed": seed, "config": config,
        "reference_splits": list(REFERENCE_SPLITS), "ood_splits": list(OOD_SPLITS),
        "speeds_mps": speeds, "distance_bins_m": distance_bins, "heading_bins_degrees": heading_bins,
        "wall_states": [False, True], "per_cell": per_cell, "ood_per_cell": per_cell,
        "ood_environment_config": heldout, "ood_distance_bins_m": [[2., 2.5]], "ood_speeds_mps": [1.5],
        "precision_target": {"absolute_error": epsilon, "failure_probability_per_split": alpha,
            "simultaneous_means": means, "minimum_draws_per_marginal_group": minimum,
            "actual_draws_per_speed": cells_per_speed*per_cell,
            "actual_smallest_group_draws": smallest_group_cells*per_cell,
            "method": "Hoeffding + union bound for 28 one-factor speed/distance/heading/wall x N/I means per reference split",
            "source_url": "https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830",
            "scope": "Independent randomized geometry/phase draws in fixed synthetic reference distribution only; not interaction-cell or hardware accuracy"},
        "reference_distribution": "Equal mass per speed x distance-bin x heading-bin x wall-state; independent uniform distance/heading/frame phase inside each cell. Experimental design, NOT measured operation frequencies.",
        "frame_phase": "Uniform [0, dt); sampled once per physical pass; not reset to zero per episode",
        "normal_interval": "Static approved layout throughout every pass; initial B cache empty",
        "identity_scope": "Same physical-Q and map assets in reference splits; only draw IDs are disjoint. No different-object generalization claim.",
        "fit_policy": "Only train p_acq/F_pass. Calibration reserved and unused; validation descriptive; freeze before test; no posthoc calibration or test selection.",
        "A_input_allowlist": ["trial_id", "split", "planned_speed_mps", "route_id", "site_id", "normal_before", "normal_after", "complete", "completed_at", "approved_items"],
        "truth_only": ["stratum_id", "distance_m", "heading_degrees", "walls", "frame_phase_seconds", "frames", "acquired_at", "B_snapshot", "scene_id"],
        "numerical_audit": "Retain every draw, including boundary cases; check aligned half/quarter timestep; never relabel or filter to improve scores",
        "hardware_status": {"camera_intrinsics": "nominal, not measured", "N_I_dwell": "experimental .3/.6 seconds, not measured",
                            "identity_pose": "ideal", "execution_noise": "not modeled", "validity": "simulation only"}}
    plan["plan_id"] = digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"}):
        raise ValueError("plan changed after freezing")
    if plan["reference_splits"] != list(REFERENCE_SPLITS) or plan["ood_splits"] != list(OOD_SPLITS):
        raise ValueError("split roles must be explicit")
    if type(plan["per_cell"]) is not int or plan["per_cell"] <= 0 or plan["ood_per_cell"] != plan["per_cell"]:
        raise ValueError("positive balanced cell allocation required")
    if scene_id(plan["config"]) == scene_id(plan["ood_environment_config"]):
        raise ValueError("OOD environment must have a different actual scene")
    target = plan["precision_target"]
    sizes = [len(plan["speeds_mps"]), len(plan["distance_bins_m"]), len(plan["heading_bins_degrees"]), len(plan["wall_states"])]
    means = 2*sum(sizes)
    minimum = ceil(log(2*means/target["failure_probability_per_split"])/(2*target["absolute_error"]**2))
    actual = len(plan["heading_bins_degrees"])*len(plan["distance_bins_m"])*2*plan["per_cell"]
    smallest = actual*len(plan["speeds_mps"])//max(sizes)
    if minimum != target["minimum_draws_per_marginal_group"] or actual != target["actual_draws_per_speed"] or smallest != target["actual_smallest_group_draws"] or smallest < minimum or means != target["simultaneous_means"]:
        raise ValueError("allocation does not meet declared synthetic precision target")
    if set(plan["speeds_mps"]) & set(plan["ood_speeds_mps"]):
        raise ValueError("OOD speeds overlap measured contexts")
    if plan["ood_distance_bins_m"][0][0] < plan["distance_bins_m"][-1][1]:
        raise ValueError("OOD distance overlaps reference range")


def cells(plan, split):
    speeds = plan["ood_speeds_mps"] if split == "ood_speed" else plan["speeds_mps"]
    distances = plan["ood_distance_bins_m"] if split == "ood_distance" else plan["distance_bins_m"]
    return list(product(speeds, range(len(distances)), range(len(plan["heading_bins_degrees"])), plan["wall_states"]))


def allocations(plan):
    validate_plan(plan)
    for split in (*REFERENCE_SPLITS, *OOD_SPLITS):
        config = plan["ood_environment_config"] if split == "ood_environment" else plan["config"]
        rng = random.Random(int(digest([plan["seed"], split]), 16))
        distances = plan["ood_distance_bins_m"] if split == "ood_distance" else plan["distance_bins_m"]
        for speed, di, hi, walls in cells(plan, split):
            stratum = f"s{speed:g}-d{di}-h{hi}-w{int(walls)}"
            for repeat in range(plan["per_cell"]):
                # Opaque identity: never encode hidden heading/walls into A IDs.
                opaque = digest([plan["plan_id"], split, stratum, repeat])[:24]
                trial_id = f"{plan['plan_id'][:12]}/{split}/{opaque}"
                distance = rng.uniform(*distances[di])
                yield {"trial_id": trial_id, "split": split, "stratum_id": stratum,
                    "scene_id": scene_id(config), "physical_object_id": "physical-Q",
                    "route_family_id": "table-route", "speed_mps": speed,
                    "distance_m": distance, "offset": config["object_xy"][1]-distance,
                    "heading_degrees": rng.uniform(*plan["heading_bins_degrees"][hi]),
                    "walls": walls, "frame_phase_seconds": rng.random()*config["dt_seconds"]}


def check_allocations(plan, rows):
    from collections import Counter
    rows = list(rows)
    if rows != list(allocations(plan)):
        raise ValueError("allocation differs from frozen label-blind draws")
    ids, geometries = set(), set()
    for row in rows:
        geometry = tuple(row[k] for k in ("scene_id", "speed_mps", "distance_m", "heading_degrees", "walls"))
        if row["trial_id"] in ids or geometry in geometries:
            raise ValueError("duplicate trial or physical geometry; repeats are not extra acquisition evidence")
        ids.add(row["trial_id"])
        geometries.add(geometry)
    counts = Counter((r["split"], r["stratum_id"]) for r in rows)
    if any(n != plan["per_cell"] for n in counts.values()):
        raise ValueError("unbalanced stratum allocation")
    return {s: sum(r["split"] == s for r in rows) for s in (*REFERENCE_SPLITS, *OOD_SPLITS)}
