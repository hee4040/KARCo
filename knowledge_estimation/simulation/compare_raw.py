"""Uncalibrated paired audit; no fitting or score-map application.

python -m knowledge_estimation.simulation.compare_raw --output knowledge_estimation/data/raw_revision_v1
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path

from ..measurement import write_json
from ..parameters import ParameterStore
from ..probabilistic import KnowledgeEstimator, RawKnowledgeEstimatorV2
from .analyze import write_csv
from .run import items
from .verify_estimator import default_plan, run as audit, sha


EVENTS = ("duplicate_ack", "duplicate_send", "timeout_after_ack", "duplicate_result", "duplicate_verification")


def expiry_probe(factory, parameters, config, event, delay):
    """Controlled API contract, distinct from physical simulation episodes."""
    est = factory(parameters=parameters)
    content = items(config)
    payload = list(content.values())
    est.record_send("m", payload, channel_condition="los-channel", content_condition="ideal-content", now=0.)
    est.observe_ack("m", payload, memory_epoch=0, received_at=.1, now=.2)
    est.record_result("task", now=.2)
    est.verify_task("task", success=True, now=.2)
    at = config["current_ttl_seconds"] + delay
    if event == "duplicate_ack":
        est.observe_ack("m", payload, memory_epoch=0, received_at=.1, now=at)
    elif event == "duplicate_send":
        est.record_send("m", payload, channel_condition="los-channel", content_condition="ideal-content", now=at)
    elif event == "timeout_after_ack":
        est.observe_ack_timeout("m", now=at)
    elif event == "duplicate_result":
        est.record_result("task", now=at)
    elif event == "duplicate_verification":
        est.verify_task("task", success=True, now=at)
    else:
        raise ValueError(event)
    # Deliberately no snapshot(now=at): it would mask the event-clock defect.
    belief = est.estimate(content["C"])
    return belief.probability, belief.stored_probability, len(belief.evidence)


def paired_rows(before, after):
    def read(root):
        with (root/"predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    old, new = read(before), read(after)
    if len(old) != len(new):
        raise AssertionError("prediction counts differ")
    output = []
    for a, b in zip(old, new):
        for key in ("episode_id", "case", "phase", "time", "geometry_id", "kind", "R", "K"):
            if a[key] != b[key]:
                raise AssertionError(f"unpaired evaluation: {key}")
        output.append({**{k: a[k] for k in ("episode_id", "case", "phase", "kind", "speed_mps", "walls", "heading_degrees", "route_standoff_m")},
            "K": int(a["K"]), "old": float(a["raw"]) if a["raw"] else None,
            "new": float(b["raw"]) if b["raw"] else None})
    return output


def metrics(records, detailed=False):
    groups = defaultdict(list)
    for row in records:
        factors = [("all", "all")]
        if detailed:
            factors += [(key, row[key]) for key in ("speed_mps", "walls", "heading_degrees", "route_standoff_m")]
        for factor, value in factors:
            groups[row["case"], row["phase"], row["kind"], factor, value].append(row)
    result = []
    for (case, phase, kind, factor, value), rows in groups.items():
        joint = [r for r in rows if r["old"] is not None and r["new"] is not None]
        record = dict(case=case, phase=phase, kind=kind, factor=factor, value=value,
                      samples=len(rows), paired=len(joint),
                      old_missing=sum(r["old"] is None for r in rows), new_missing=sum(r["new"] is None for r in rows))
        for name in ("old", "new"):
            record[name+"_brier"] = sum((r[name]-r["K"])**2 for r in joint)/len(joint) if joint else None
            record[name+"_mean"] = sum(r[name] for r in joint)/len(joint) if joint else None
        record["actual_K_rate"] = sum(r["K"] for r in joint)/len(joint) if joint else None
        delta = record["new_brier"]-record["old_brier"] if joint else None
        record.update(delta=delta, change="unsupported" if delta is None else "improved" if delta < 0 else "worsened" if delta > 0 else "equal")
        result.append(record)
    return result


def run(output, model_root):
    root, model_root = Path(output), Path(model_root)
    base = default_plan(model_root)
    plan = {"base": base, "calibration": "OFF: no calibration map is applied; no fitting or selection",
        "revision": "Clock-only fix frozen before either evaluation; same source parameters",
        "validation_sweep": {"offsets_m": [.32, 1.32], "headings_degrees": [42., 132.], "speeds_mps": [.5, 1., 2.]},
        "test_sweep": {"offsets_m": [.38, 1.38], "headings_degrees": [48., 138.], "speeds_mps": [.5, 1., 2.]},
        "source_parameter_sha256": sha(model_root/"simulation_params.yaml"),
        "v2_code_sha256": sha(Path(__file__).parents[1]/"probabilistic"/"raw_v2.py"),
        "sampling": "Designed 2D grid, shared geometries across interventions; not IID trials",
        "contract_probe": "5 duplicate event paths x 4 later times per split; deterministic API/TTL checks, not physical episodes"}
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"experiment_plan.json", plan)
    parameters = ParameterStore.load(model_root/"simulation_params.yaml", allow_simulation=True)
    summary = {}
    for split, delays in (("validation", (.01, .2, .6, 1.2)), ("test", (.03, .4, .8, 1.6))):
        stage = deepcopy(base)
        stage["split"] = split
        stage["config"]["episode_namespace"] = "raw-revision-v1/"+split
        stage["config"]["sweeps"]["test"] = plan[split+"_sweep"]
        reports = []
        for name, factory in (("old", KnowledgeEstimator), ("new", RawKnowledgeEstimatorV2)):
            reports.append(audit(root/split/name, model_root=model_root, plan=stage,
                                 estimator_factory=factory, raw_only=True))
        # The two estimators receive byte-identical observations and labels.
        for file in ("observations.jsonl", "labels.jsonl", "world_trace.jsonl"):
            a, b = root/split/"old"/"episodes"/file, root/split/"new"/"episodes"/file
            if sha(a) != sha(b):
                raise AssertionError(f"different evaluation inputs/truth: {file}")
        records = paired_rows(root/split/"old", root/split/"new")
        table = metrics(records)
        write_csv(root/split/"paired_predictions.csv", records)
        write_csv(root/split/"metrics.csv", table)
        write_csv(root/split/"conditions.csv", metrics(records, detailed=True))
        probes = []
        for event in EVENTS:
            for delay in delays:
                old = expiry_probe(KnowledgeEstimator, parameters, base["config"], event, delay)
                new = expiry_probe(RawKnowledgeEstimatorV2, parameters, base["config"], event, delay)
                if new != (0., 1., 1):
                    raise AssertionError("expiry fix lost storage or duplicated evidence")
                probes.append(dict(event=event, delay_after_TTL=delay, K=0, R=1,
                                   old=old[0], new=new[0], old_brier=old[0]**2, new_brier=new[0]**2))
        write_csv(root/split/"expiry_contract_probes.csv", probes)
        summary[split] = {"episodes_per_estimator": reports[0]["episodes"], "item_predictions": len(records),
            "logic_checks_per_estimator": reports[0]["logic_checks"],
            "groups": dict(Counter(r["change"] for r in table)),
            "missing_predictions": sum(r["new"] is None for r in records),
            "contract_samples": len(probes), "contract_old_brier": sum(r["old_brier"] for r in probes)/len(probes),
            "contract_new_brier": sum(r["new_brier"] for r in probes)/len(probes)}
    if sha(model_root/"simulation_params.yaml") != plan["source_parameter_sha256"] or sha(Path(__file__).parents[1]/"probabilistic"/"raw_v2.py") != plan["v2_code_sha256"]:
        raise AssertionError("frozen code/parameters changed during comparison")
    write_json(root/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("knowledge_estimation/data/calibration_repair_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.model_root), indent=2))
