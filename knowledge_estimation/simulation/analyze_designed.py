"""Describe all declared marginal groups without refitting or selection."""

import argparse
from collections import defaultdict
import json
from math import log, sqrt
from pathlib import Path

from ..measurement import read_jsonl, write_json
from .analyze import write_csv
from .collection_design import REFERENCE_SPLITS, OOD_SPLITS, validate_plan
from .collect_designed import sha


def analyze(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("collection checksum mismatch")
    plan = json.loads((root/"collection_plan.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    quality = json.loads((root/"quality_report.json").read_text(encoding="utf-8"))
    if not quality["passed"]:
        raise ValueError("collection failed quality checks")
    draws = {r["trial_id"]: r for r in read_jsonl(root/"allocation.private.jsonl")}
    groups = defaultdict(list)
    for split in (*REFERENCE_SPLITS, *OOD_SPLITS):
        # Calibration remains reserved: only record counts, never analyze labels.
        if split == "calibration":
            continue
        prediction_file = root/split/"raw_predictions.jsonl"
        predictions = {r["trial_id"]: r["raw"] for r in read_jsonl(prediction_file)} if prediction_file.exists() else {}
        for label in read_jsonl(root/split/"B_labels.private.jsonl"):
            draw = draws[label["trial_id"]]
            di = int(draw["stratum_id"].split("-d")[1].split("-h")[0])
            hi = int(draw["stratum_id"].split("-h")[1].split("-w")[0])
            distances = plan["ood_distance_bins_m"] if split == "ood_distance" else plan["distance_bins_m"]
            factors = [("all", "all"), ("speed_mps", draw["speed_mps"]),
                ("distance_bin_m", json.dumps(distances[di])),
                ("heading_bin_degrees", json.dumps(plan["heading_bins_degrees"][hi])),
                ("walls", draw["walls"]), ("stratum", draw["stratum_id"])]
            for kind in "NI":
                p = predictions.get(draw["trial_id"], {}).get(kind)
                sensitive = any((kind in label["finer_acquired_at"][d]) != bool(label["K"][kind]) for d in ("2", "4"))
                for factor, value in factors:
                    groups[split, kind, factor, value].append((p, label["K"][kind], sensitive))
    rows = []
    target = plan["precision_target"]
    for (split, kind, factor, value), values in groups.items():
        known = [(p, y) for p, y, _ in values if p is not None]
        marginal = factor not in {"all", "stratum"}
        scope = split in REFERENCE_SPLITS and marginal
        radius = sqrt(log(2*target["simultaneous_means"]/target["failure_probability_per_split"])/(2*len(values))) if scope else None
        rate = sum(y for _, y, _ in values)/len(values)
        rows.append(dict(split=split, kind=kind, factor=factor, value=value, unique_draws=len(values),
            acquired=sum(y for _, y, _ in values), actual_K_rate=rate,
            predicted=len(known), missing=len(values)-len(known),
            mean_prediction=sum(p for p, _ in known)/len(known) if known else None,
            actual_rate_on_known=sum(y for _, y in known)/len(known) if known else None,
            brier=sum((p-y)**2 for p, y in known)/len(known) if known else None,
            resolution_sensitive=sum(s for _, _, s in values),
            reference_mean_bound_radius=radius,
            reference_mean_bound_lower=max(0., rate-radius) if radius is not None else None,
            reference_mean_bound_upper=min(1., rate+radius) if radius is not None else None))
    for split in ("train", "validation", "test"):
        marginal = [r for r in rows if r["split"] == split and r["factor"] not in {"all", "stratum"}]
        if len(marginal) != target["simultaneous_means"] or any(r["reference_mean_bound_radius"] > target["absolute_error"] for r in marginal):
            raise ValueError("declared marginal precision not met")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"all_conditions.csv", rows)
    write_json(output/"summary.json", dict(plan_id=plan["plan_id"], source_manifest_sha256=sha(root/"manifest.json"),
        calibration="reserved; labels not analyzed here", fitting="none",
        reference_totals=[r for r in rows if r["factor"] == "all" and r["split"] in REFERENCE_SPLITS],
        bound_scope="Synthetic reference group outcome mean, 28 groups simultaneously per split; not prediction error or a train-test difference bound. No such claim for 20-draw interaction cells or OOD.",
        target=target))
    return dict(rows=len(rows), output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.output), indent=2))
