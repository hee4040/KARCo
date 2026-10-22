"""Fit measured source rates or a frozen isotonic score map, without priors.

python -m knowledge_estimation.calibrate DATASET --mode sources --output params.yaml
python -m knowledge_estimation.calibrate DATASET --mode scores --output calibration.json
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .measurement import decision_pairs, load_dataset, write_json, time_value
from .parameters import CONTEXT_FIELDS, PROBABILITIES, DISTRIBUTIONS, MissingMeasurement, context_key
from .probabilistic.projection import source_key


def _id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit_sources(root, *, allow_fixture=False, allow_simulation=False):
    manifest, rows, _ = load_dataset(root, allow_fixture=allow_fixture, allow_simulation=allow_simulation)
    groups, seen = defaultdict(list), {}
    for row in rows:
        if row["split"] != "calibration":
            continue
        name = row["source"]
        if name not in PROBABILITIES + DISTRIBUTIONS:
            raise ValueError(f"unsupported measurement source {name}")
        if row["success"] not in {"0", "1"}:
            raise ValueError("each opportunity needs a binary externally checked outcome")
        if not row["opportunity_id"].strip() or not row["evidence_id"].strip():
            raise ValueError("opportunities and raw evidence must be identifiable")
        context = {field: row[field] for field in CONTEXT_FIELDS}
        context_key(context)
        if name in {"p_acq", "F_pass"} and (not row["route_id"] or not row["site_id"] or row["kind"] not in {"N", "I"}):
            raise ValueError("pass calibration requires route, site, and N/I kind")
        expected_kind = {"q_learn": "I", "r_crop": "I", "r_current": "C"}.get(name)
        if expected_kind and row["kind"] != expected_kind:
            raise ValueError(f"{name} must use kind {expected_kind}")
        identity = (name, context_key(context), row["episode_id"], row["opportunity_id"])
        if identity in seen:
            raise ValueError("duplicate acquisition opportunity; snapshots cannot be counted as extra trials")
        seen[identity] = row
        groups[(name, context_key(context))].append(row)
    if not groups:
        raise MissingMeasurement("no calibration measurement opportunities")
    for (name, context, episode, opportunity), row in seen.items():
        parent = {"p_ack": "p_recv", "F_recv": "p_recv", "F_pass": "p_acq"}.get(name)
        if parent:
            parent_row = seen.get((parent, context, episode, opportunity))
            if parent_row is None or parent_row["success"] != "1":
                raise ValueError(f"{name} is conditional on a successful {parent} with the same opportunity")
    entries = []
    for (name, key), trials in sorted(groups.items()):
        entry = {"name": name, "context": dict(zip(CONTEXT_FIELDS, key)),
                 "trials": len(trials), "successes": sum(int(r["success"]) for r in trials),
                 "episodes": sorted({r["episode_id"] for r in trials}),
                 "session_ids": sorted({r["session_id"] for r in trials}),
                 "object_instances": sorted({r["object_instance"] for r in trials}),
                 "route_group_ids": sorted({r["route_group_id"] for r in trials}),
                 "evidence_ids": sorted({r["evidence_id"] for r in trials}),
                 "input_sha256": manifest["sha256"]["trials.csv"], "method": "empirical_frequency_no_prior"}
        if name in PROBABILITIES:
            entry["value"] = entry["successes"]/entry["trials"]
        horizons = {r["deadline_seconds"] for r in trials}
        if name in DISTRIBUTIONS or name in {"p_recv", "p_ack"}:
            if len(horizons) != 1 or "" in horizons:
                raise ValueError("one context must have one explicitly measured/protocol deadline")
            horizon = float(horizons.pop())
            time_value(horizon, "deadline")
            if horizon == 0:
                raise ValueError("deadline must be positive")
            entry["deadline_seconds"] = horizon
        if name in DISTRIBUTIONS:
            samples = []
            for row in trials:
                if row["success"] == "1":
                    duration = float(row["duration_seconds"])
                    time_value(duration, "duration_seconds")
                    if duration > horizon:
                        raise ValueError("duration exceeds observation horizon")
                    samples.append(duration)
                elif name != "F_task":
                    raise ValueError("conditional F_pass/F_recv include only successful parent trials")
            entry["samples_seconds"] = sorted(samples)
            entry["observation_horizon_seconds"] = horizon
            entry["method"] = "empirical_cdf_with_fixed_observation_horizon"
        entry["parameter_id"] = _id(entry)
        entries.append(entry)
    # The no-ACK conditional must refer to the same cutoff as p_recv.
    table = {(e["name"], context_key(e["context"])): e for e in entries}
    for entry in entries:
        if entry["name"] == "p_ack":
            parent = table[("p_recv", context_key(entry["context"]))]
            if parent["deadline_seconds"] != entry["deadline_seconds"]:
                raise ValueError("p_recv and conditional p_ack use different deadlines")
    return {"schema_version": 1, "origin": manifest["origin"], "protocol_id": manifest["protocol_id"],
            "parameters": entries, "fit_split": "calibration", "input_sha256": manifest["sha256"],
            "groups": {field: sorted({r[field] for r in rows if r["split"] == "calibration"})
                       for field in ("episode_id", "session_id", "object_instance", "route_group_id")}}


def monotone_map(pairs):
    grouped = defaultdict(lambda: [0, 0])
    for score, label in pairs:
        grouped[score][0] += 1
        grouped[score][1] += label
    blocks = []
    for score, (n, s) in sorted(grouped.items()):
        blocks.append({"upper": score, "count": n, "positive": s})
        while len(blocks) > 1 and blocks[-2]["positive"]/blocks[-2]["count"] > blocks[-1]["positive"]/blocks[-1]["count"]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append({"upper": right["upper"], "count": left["count"]+right["count"],
                           "positive": left["positive"]+right["positive"]})
    if not blocks:
        raise MissingMeasurement("no known raw calibration scores for one item")
    for block in blocks:
        block["value"] = block["positive"]/block["count"]
    return {"support": [min(grouped), max(grouped)], "blocks": blocks}


def fit_scores(root, *, allow_fixture=False, allow_simulation=False, conditioning="pooled"):
    if conditioning not in {"pooled", "evidence_sources_v1"}:
        raise ValueError("unsupported calibration conditioning")
    manifest, _, _ = load_dataset(root, allow_fixture=allow_fixture, allow_simulation=allow_simulation)
    pairs, fingerprints, receivers, selected = {k: [] for k in "NCI"}, set(), set(), []
    context_pairs = {k: defaultdict(list) for k in "NCI"}
    for row, _, _, truth in decision_pairs(root, split="calibration"):
        if row["split"] != "calibration":
            continue
        fingerprints.add(row["estimator_fingerprint"])
        receivers.add(row["receiver_id"])
        selected.append(row)
        for kind in "NCI":
            score = row["b_raw"][f"b_{kind}"]
            if score is None:
                raise MissingMeasurement("raw score is unknown: fit sources, replay A history with fixed params, then fit scores")
            if not row["hard_invalid"][kind]:
                pairs[kind].append((score, truth[kind]["K"]))
                key = source_key(e["source"] for e in row["evidence"][kind])
                context_pairs[kind][key].append((score, truth[kind]["K"]))
    if len(fingerprints) != 1:
        raise ValueError("score calibration needs one frozen raw estimator/parameter set")
    if len(receivers) != 1:
        raise ValueError("fit a separate score calibration per receiver; do not pool robots")
    model = {"schema_version": 1, "origin": manifest["origin"], "protocol_id": manifest["protocol_id"],
             "estimator_fingerprint": fingerprints.pop(), "receiver_ids": sorted(receivers),
             "maps": {k: monotone_map(v) for k, v in pairs.items()}, "fit_split": "calibration",
             "method": "isotonic_PAV_no_prior_no_extrapolation", "decisions": len(selected),
             "input_sha256": manifest["sha256"],
             "groups": {field: sorted({r[field] for r in selected})
                        for field in ("episode_id", "session_id", "object_instance", "route_group_id")}}
    model["source_fit_groups"] = {field: sorted({value for r in selected for value in r["parameter_fit_groups"].get(field, [])})
                                  for field in ("episode_id", "session_id", "object_instance", "route_group_id")}
    if conditioning != "pooled":
        model["conditioning"] = conditioning
        model["context_maps"] = {kind: {key: monotone_map(values) for key, values in groups.items()}
                                 for kind, groups in context_pairs.items()}
        model["method"] = "A_evidence_sources_conditioned_isotonic_PAV_no_fallback"
    model["model_id"] = _id(model)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--mode", choices=("sources", "scores"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--allow-simulation", action="store_true")
    parser.add_argument("--conditioning", choices=("pooled", "evidence_sources_v1"), default="pooled")
    args = parser.parse_args(argv)
    try:
        options = {"allow_fixture": args.allow_fixture, "allow_simulation": args.allow_simulation}
        if args.mode == "scores":
            options["conditioning"] = args.conditioning
        elif args.conditioning != "pooled":
            raise ValueError("conditioning applies to score calibration only")
        result = (fit_sources if args.mode == "sources" else fit_scores)(args.dataset, **options)
        write_json(args.output, result)  # JSON syntax is valid YAML 1.2.
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Calibration stopped: {error}\n")
    print(json.dumps({"output": str(args.output), "origin": result["origin"], "mode": args.mode}))


if __name__ == "__main__":
    main()
