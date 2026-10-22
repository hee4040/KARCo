"""Evaluate frozen calibrated beliefs against decision-time B DB truth."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

from .measurement import decision_pairs, load_dataset, read_jsonl, write_json
from .parameters import MissingMeasurement
from .probabilistic.projection import validate_model, calibrated_value
from .types import probability


def _quantile(values, fraction):
    values = sorted(values)
    index = (len(values)-1)*fraction
    low = int(index)
    high = min(len(values)-1, low+1)
    return values[low] + (values[high]-values[low])*(index-low)


def _cdf(k, n, p):
    if p == 0:
        return 1.
    if p == 1:
        return float(k >= n)
    logs = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
            + j*math.log(p)+(n-j)*math.log1p(-p) for j in range(k+1)]
    peak = max(logs)
    return min(1., math.exp(peak)*sum(math.exp(v-peak) for v in logs))


def binomial_interval(successes, n, confidence):
    """Two-sided Clopper-Pearson; assumes independent Bernoulli units."""
    if not 0 < confidence < 1 or type(n) is not int or type(successes) is not int or not 0 <= successes <= n:
        raise ValueError("invalid binomial interval inputs")
    if n == 0:
        return None
    alpha = 1-confidence
    lower, upper = 0., 1.
    if successes:
        lo, hi = 0., 1.
        for _ in range(64):  # Binary64 numerical resolution, not a physical parameter.
            mid = (lo+hi)/2
            if 1-_cdf(successes-1, n, mid) < alpha/2:
                lo = mid
            else:
                hi = mid
        lower = (lo+hi)/2
    if successes < n:
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo+hi)/2
            if _cdf(successes, n, mid) > alpha/2:
                lo = mid
            else:
                hi = mid
        upper = (lo+hi)/2
    return [lower, upper]


def _protocol(plan):
    names = {"high_confidence_threshold", "reliability_bin_edges", "confidence_level",
             "bootstrap_repetitions", "bootstrap_seed", "bootstrap_unit"}
    if not names <= set(plan) or any(plan[key] is None for key in names):
        raise MissingMeasurement("predeclare evaluation threshold/bins/confidence/bootstrap protocol; no defaults")
    probability(plan["high_confidence_threshold"], "high confidence threshold")
    if not 0 < plan["confidence_level"] < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    edges = plan["reliability_bin_edges"]
    if not isinstance(edges, list) or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1:
        raise ValueError("reliability bins must span [0,1]")
    for x in edges:
        probability(x, "bin edge")
    if any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("bin edges must increase strictly")
    if type(plan["bootstrap_repetitions"]) is not int or plan["bootstrap_repetitions"] <= 0 or type(plan["bootstrap_seed"]) is not int:
        raise ValueError("bootstrap count and seed must be declared integers")
    if plan["bootstrap_unit"] not in {"episode_id", "session_id"}:
        raise ValueError("bootstrap unit must be episode_id or session_id")
    caps = plan.get("safe_caps", {})
    if not isinstance(caps, dict) or set(caps) - {"b_N", "b_C", "b_I"}:
        raise ValueError("safe_caps must be an explicit per-item policy ablation map")
    for value in caps.values():
        probability(value, "safe policy cap")


def _metrics(values, plan):
    """Each value is (prediction, K, bootstrap group)."""
    if not values:
        return {"samples": 0, "brier": None, "reliability": [], "high_confidence_false_rate": None}
    groups = defaultdict(list)
    for value in values:
        groups[value[2]].append(value)
    rng = random.Random(plan["bootstrap_seed"])
    keys = sorted(groups)
    designed_sweep = plan.get("sampling_design") == "deterministic_factorial_sweep"
    resampled = [[v for key in rng.choices(keys, k=len(keys)) for v in groups[key]]
                 for _ in range(0 if designed_sweep else plan["bootstrap_repetitions"])]
    tail = (1-plan["confidence_level"])/2
    def interval(statistic):
        if designed_sweep:
            return {"interval": None, "valid_repetitions": 0,
                    "reason": "deterministic scenario sweep; no independent random sampling claim"}
        if len(keys) < 2:
            return {"interval": None, "valid_repetitions": 0, "reason": "at least two independent bootstrap groups are needed"}
        results = [statistic(sample) for sample in resampled]
        results = [v for v in results if v is not None]
        return {"interval": [_quantile(results, tail), _quantile(results, 1-tail)] if results else None,
                "valid_repetitions": len(results)}
    def brier(rows):
        return sum((p-y)**2 for p, y, _ in rows)/len(rows)
    def high_error(rows):
        high = [1-y for p, y, _ in rows if p >= plan["high_confidence_threshold"]]
        return sum(high)/len(high) if high else None
    high = [y for p, y, _ in values if p >= plan["high_confidence_threshold"]]
    bins = []
    edges = plan["reliability_bin_edges"]
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        def selected(rows):
            return [(p, y, g) for p, y, g in rows if lower <= p < upper or (index == len(edges)-2 and p == upper)]
        bucket = selected(values)
        def rate(rows):
            found = selected(rows)
            return sum(y for _, y, _ in found)/len(found) if found else None
        bins.append({"lower": lower, "upper": upper, "samples": len(bucket),
                     "mean_prediction": sum(p for p, _, _ in bucket)/len(bucket) if bucket else None,
                     "positive_rate": rate(values), "positive_rate_bootstrap": interval(rate)})
    episode_failures = {key: int(any(p >= plan["high_confidence_threshold"] and y == 0 for p, y, _ in rows))
                        for key, rows in groups.items() if any(p >= plan["high_confidence_threshold"] for p, _, _ in rows)}
    count = len(episode_failures)
    failures = sum(episode_failures.values())
    return {"samples": len(values), "bootstrap_groups": len(groups), "positive": sum(y for _, y, _ in values),
            "brier": brier(values), "brier_bootstrap": interval(brier), "reliability": bins,
            "high_confidence_samples": len(high), "high_confidence_false_rate": high_error(values),
            "high_confidence_false_rate_bootstrap": interval(high_error),
            "groups_with_any_high_confidence_error": {"groups": count, "failed_groups": failures,
                "exact_binomial_interval_if_independent": None if designed_sweep else binomial_interval(failures, count, plan["confidence_level"]),
                "zero_failures_one_sided_upper": 1-(1-plan["confidence_level"])**(1/count) if count and failures == 0 and not designed_sweep else None}}


def evaluate(root, model, protocol, *, split="test", allow_fixture=False, allow_simulation=False):
    _protocol(protocol)
    validate_model(model)
    manifest, _, _ = load_dataset(root, allow_fixture=allow_fixture, allow_simulation=allow_simulation)
    if model["origin"] != manifest["origin"] or (model["origin"] == "fixture" and not allow_fixture):
        raise ValueError("calibration and evaluation origins must match")
    if split not in {"test", "validation"}:
        raise ValueError("evaluation uses validation or untouched final test only")
    values = {level: {kind: [] for kind in "NCI"} for level in ("raw", "cal", "safe")}
    counts, causes, results = Counter(), {k: Counter() for k in "NCI"}, []
    strata = {"source": defaultdict(list), "route_condition": defaultdict(list)}
    for row, before, after, truth in decision_pairs(root, split=split):
        if row["split"] != split:
            continue
        for field, groups in {field: set(model["groups"].get(field, [])) | set(model.get("application_validation_groups", {}).get(field, []))
                              for field in model["groups"]}.items():
            if row[field] in groups:
                raise ValueError(f"calibration/evaluation leakage in {field}")
        for field, groups in model["source_fit_groups"].items():
            if row[field] in groups:
                raise ValueError(f"source parameter/evaluation leakage in {field}")
        if row["estimator_fingerprint"] != model["estimator_fingerprint"] or row["receiver_id"] not in model["receiver_ids"]:
            raise ValueError("fixed model does not match raw estimator/receiver")
        result = {"decision_id": row["decision_id"], "episode_id": row["episode_id"], "raw": {}, "cal": {}, "K": {}}
        for kind in "NCI":
            p = row["b_raw"][f"b_{kind}"]
            if p is None:
                raise MissingMeasurement(f"unknown raw score at {row['decision_id']}/{kind}; evaluation cannot silently drop it")
            probability(p, "raw score")
            q = calibrated_value(model, kind, p, hard_invalid=row["hard_invalid"][kind],
                                 sources=[e["source"] for e in row["evidence"][kind]],
                                 timestamps=[e["timestamp"] for e in row["evidence"][kind]], now=row["time"])
            safe = min(q, protocol.get("safe_caps", {}).get(f"b_{kind}", 1.))
            probability(safe, "safe value")
            y = truth[kind]["K"]
            for level, prediction in (("raw", p), ("cal", q), ("safe", safe)):
                values[level][kind].append((prediction, y, row[protocol["bootstrap_unit"]]))
            sources = {e["source"] for e in row["evidence"][kind]} or {"no_evidence"}
            for source in sources:
                strata["source"][(kind, source)].append((q, y, row[protocol["bootstrap_unit"]]))
            for entry in row["source_contexts"][kind]:
                if entry["name"] == "p_acq":
                    ctx = entry["context"]
                    strata["route_condition"][(kind, ctx["route_id"], ctx["site_id"], ctx["condition_id"])].append(
                        (q, y, row[protocol["bootstrap_unit"]]))
            result["raw"][kind], result["cal"][kind], result["K"][kind] = p, q, y
            if q >= protocol["high_confidence_threshold"] and not y:
                causes[kind].update(truth[kind]["causes"])
                if before["snapshot"]["memory_epoch"] != row["known_epoch"]:
                    causes[kind]["unannounced_epoch_change"] += 1
            outcome = after.get("outcome") or {}
            if kind not in row["selected_items"] and kind in row["required_items"] and not y:
                counts[f"{kind}:omission_failure"] += 1
                if kind in outcome.get("request_items", []):
                    counts[f"{kind}:request_triggered"] += 1
                    from .measurement import label_before
                    post_truth = label_before(after["snapshot"], after["world"], after["snapshot"]["time"])
                    counts[f"{kind}:recovered_by_request" if post_truth[kind]["K"] else f"{kind}:request_pending_or_failed"] += 1
                elif kind in outcome.get("silent_wrong_causal_items", []):
                    counts[f"{kind}:silent_wrong_action"] += 1
                else:
                    counts[f"{kind}:outcome_not_attributed"] += 1
            if kind in row["selected_items"] and y and row["known_versions"][kind] == before["world"]["versions"][kind]:
                counts[f"{kind}:unnecessary_transmission"] += 1
        results.append(result)
    if not results:
        raise ValueError("no decision records in evaluation split")
    episode_ids = {r["episode_id"] for r in results}
    packets = [p for p in read_jsonl(Path(root) / "transport.jsonl") if p["episode_id"] in episode_ids]
    communication = {"logged_application_bytes": sum(p["bytes"] for p in packets) if packets else None,
                     "recorded_messages": len(packets),
                     "by_kind": {kind: sum(p["bytes"] for p in packets if p["message_kind"] == kind)
                                 for kind in sorted({p["message_kind"] for p in packets})},
                     "by_category": {category: sum(p["bytes"] for p in packets if p["category"] == category)
                                     for category in sorted({p["category"] for p in packets})},
                     "by_direction": {direction: sum(p["bytes"] for p in packets if p["direction"] == direction)
                                      for direction in sorted({p["direction"] for p in packets})},
                     "retry_bytes": sum(p["bytes"] for p in packets if p["is_retry"]) if packets else None,
                     "interpretation": "actual logged serializer bytes; no counterfactual episode totals"}
    return {"schema_version": 1, "origin": manifest["origin"], "model_id": model["model_id"], "split": split,
            "protocol": protocol, "decisions": len(results), "episodes": len(episode_ids),
            "metrics": {level: {k: _metrics(v, protocol) for k, v in by_kind.items()} for level, by_kind in values.items()},
            "stratified_calibration": {name: [{"stratum": list(key), "metrics": _metrics(v, protocol)}
                                               for key, v in sorted(groups.items())] for name, groups in strata.items()},
            "high_confidence_failure_causes": {k: dict(v) for k, v in causes.items()},
            "omission_and_redundancy_counts": dict(counts), "communication": communication,
            "paired_decision_rows": results,
            "interpretation": "simulation-only; idealized geometry and declared scenario sweep, not hardware performance" if manifest["origin"] == "simulation" else
                "fixture outputs test plumbing only" if manifest["origin"] == "fixture" else
                "recorded-decision evaluation; correlations and dataset scope limit generalization"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--allow-simulation", action="store_true")
    args = parser.parse_args(argv)
    try:
        model = json.loads(args.model.read_text(encoding="utf-8"))
        plan = json.loads(args.protocol.read_text(encoding="utf-8"))
        plan = plan.get("evaluation_protocol", plan)
        result = evaluate(args.dataset, model, plan, split=args.split, allow_fixture=args.allow_fixture,
                          allow_simulation=args.allow_simulation)
        write_json(args.output, result)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Evaluation stopped: {error}\n")
    print(json.dumps({"output": str(args.output), "origin": result["origin"], "decisions": result["decisions"]}))


if __name__ == "__main__":
    main()
