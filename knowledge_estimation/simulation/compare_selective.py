"""Separate validation and final audit of optional calibration with raw fallback.

python -m knowledge_estimation.simulation.compare_selective --output knowledge_estimation/data/selective_calibration_v1
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from ..measurement import read_jsonl, decision_pairs, write_json, load_dataset
from ..probabilistic.projection import calibration_result, validate_model
from ..probabilistic.selective import fit_application_policy
from ..parameters import MissingMeasurement
from .analyze import conditions, write_csv
from .verify_estimator import default_plan, run as audit, sha


def experiment_plan(model_root):
    plan = default_plan(model_root)
    plan["fit_policy"] = "Base maps/source parameters frozen; select optional application per A context on separate validation only"
    return {"schema_version": 1, "config": deepcopy(plan["config"]), "base_audit_plan": plan,
        "validation_sweep": {"offsets_m": [.12, 1.12], "headings_degrees": [28., 118.], "speeds_mps": [.5, 1., 2.]},
        "final_sweep": {"offsets_m": [.24, 1.24], "headings_degrees": [36., 126.], "speeds_mps": [.5, 1., 2.]},
        "selection": "Per A source+exact-event-freshness context: require training support, complete validation calibration support, and >1 geometry; apply only if geometry-weighted validation Brier strictly improves raw. Tie/unseen/out-of-support preserves raw. Unknown raw stays unknown.",
        "fresh_ACK": "Preserve original per-source storage/content-validity estimate at ACK processing time; never force K=1",
        "test_access": "Final grid frozen now; generate final labels only after saving selected model and selection hashes",
        "scope": "Designed 2D conditions, not IID sampling or a claim that two geometries statistically suffice"}


def predictions(root, old, new):
    root = Path(root)
    load_dataset(root, allow_simulation=True)
    validate_model(old)
    validate_model(new)
    records = []
    # Compute from A inputs first; join privileged truth only afterwards.
    for row in read_jsonl(root/"inputs.jsonl"):
        if row["estimator_fingerprint"] != new["estimator_fingerprint"] or row["receiver_id"] not in new["receiver_ids"]:
            raise ValueError("comparison raw estimator/receiver mismatch")
        for kind in "NCI":
            evidence = row["evidence"][kind]
            args = {"hard_invalid": row["hard_invalid"][kind], "sources": [e["source"] for e in evidence],
                    "timestamps": [e["timestamp"] for e in evidence], "now": row["time"]}
            raw = row["b_raw"]["b_"+kind]
            old_reason = ""
            try:
                before, old_reason = calibration_result(old, kind, raw, **args)
            except MissingMeasurement as error:
                before, old_reason = None, str(error)
            after, reason = calibration_result(new, kind, raw, **args)
            records.append({"decision_id": row["decision_id"], "episode_id": row["episode_id"],
                "case": row["episode_id"].split("/")[-2], "phase": row["decision_id"].rsplit("/", 1)[-1],
                "kind": kind, "raw": raw, "old": before, "new": after, "old_reason": old_reason, "new_reason": reason})
    truth = {r["decision_id"]: t for r, _, _, t in decision_pairs(root)}
    traces = {t["episode_id"]: t for t in read_jsonl(root/"world_trace.jsonl")}
    config = json.loads((root/"scenario.json").read_text(encoding="utf-8"))
    for row in records:
        row.update(K=truth[row["decision_id"]][row["kind"]]["K"], R=truth[row["decision_id"]][row["kind"]]["R"])
        c = conditions(traces[row["episode_id"]], config)
        row.update({k: c[k] for k in ("speed_mps", "route_standoff_m", "heading_degrees", "walls")})
    return records


def metric_rows(records, *, detailed=False):
    groups = defaultdict(list)
    for row in records:
        factors = [("all", "all")]
        if detailed:
            factors += [(k, row[k]) for k in ("speed_mps", "route_standoff_m", "heading_degrees", "walls")]
        for factor, value in factors:
            groups[(row["case"], row["phase"], row["kind"], factor, value)].append(row)
    metrics = []
    for (case, phase, kind, factor, value), rows in groups.items():
        m = {"case": case, "phase": phase, "kind": kind, "factor": factor, "value": value,
             "samples": len(rows), "actual_K_rate": sum(r["K"] for r in rows)/len(rows)}
        for name in ("raw", "old", "new"):
            known = [r for r in rows if r[name] is not None]
            m[name+"_known"] = len(known)
            m[name+"_brier"] = sum((r[name]-r["K"])**2 for r in known)/len(known) if known else None
        paired = [r for r in rows if r["old"] is not None and r["new"] is not None]
        m["paired_samples"] = len(paired)
        for name in ("old", "new"):
            m[name+"_paired_brier"] = sum((r[name]-r["K"])**2 for r in paired)/len(paired) if paired else None
        delta = m["new_paired_brier"]-m["old_paired_brier"] if paired else None
        m["delta_new_minus_old"] = delta
        m["change"] = "no_paired_support" if delta is None else "improved" if delta < 0 else "worsened" if delta > 0 else "equal"
        metrics.append(m)
    return metrics


def run(output, *, model_root, plan):
    root, model_root = Path(output), Path(model_root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"experiment_plan.json", plan)
    old = json.loads((model_root/"selected_model.json").read_text(encoding="utf-8"))
    validation_plan = deepcopy(plan["base_audit_plan"])
    validation_plan["split"] = "validation"
    validation_plan["config"]["episode_namespace"] = "selective-v1/validation"
    validation_plan["config"]["sweeps"]["test"] = plan["validation_sweep"]
    audit(root/"validation", model_root=model_root, plan=validation_plan)
    new = fit_application_policy(old, model_root/"calibration_replayed", root/"validation"/"episodes", allow_simulation=True)
    write_json(root/"selected_model.json", new)
    write_json(root/"simulation_params.yaml", json.loads((model_root/"simulation_params.yaml").read_text(encoding="utf-8")))
    selection = {"selected_model_id": new["model_id"], "selected_model_sha256": sha(root/"selected_model.json"),
        "plan_sha256": sha(root/"experiment_plan.json"), "validation_manifest_sha256": sha(root/"validation"/"episodes"/"manifest.json"),
        "final_test_exists": (root/"final_test").exists(),
        "gates": new["application_policy"]["gates"]}
    if selection["final_test_exists"]:
        raise ValueError("final labels existed before selection freeze")
    write_json(root/"selection.json", selection)
    final_plan = deepcopy(plan["base_audit_plan"])
    final_plan["config"]["episode_namespace"] = "selective-v1/final"
    final_plan["config"]["sweeps"]["test"] = plan["final_sweep"]
    # Capture the same causal histories once with frozen old estimator, then
    # compare fixed old/new applications. No counterfactual policy rollout.
    final_summary = audit(root/"final_test", model_root=model_root, plan=final_plan)
    final = predictions(root/"final_test"/"episodes", old, new)
    validation = predictions(root/"validation"/"episodes", old, new)
    write_csv(root/"final_predictions.csv", final)
    metrics = metric_rows(final)
    details = metric_rows(final, detailed=True)
    write_csv(root/"final_by_case_phase.csv", metrics)
    write_csv(root/"final_by_condition.csv", details)
    write_csv(root/"validation_by_case_phase.csv", metric_rows(validation))
    for field in ("episode_id", "session_id", "object_instance", "route_group_id"):
        _, _, val_inputs = load_dataset(root/"validation"/"episodes", allow_simulation=True)
        _, _, test_inputs = load_dataset(root/"final_test"/"episodes", allow_simulation=True)
        if {r[field] for r in val_inputs} & {r[field] for r in test_inputs}:
            raise ValueError("validation/final groups overlap")
    if sha(root/"selected_model.json") != selection["selected_model_sha256"]:
        raise ValueError("selection changed after final evaluation")
    summary = {"origin": "simulation", "base_model_id": old["model_id"], "new_model_id": new["model_id"],
        "validation_episodes": len(plan["validation_sweep"]["offsets_m"])*len(plan["validation_sweep"]["headings_degrees"])*2*len(plan["validation_sweep"]["speeds_mps"])*len(final_plan["cases"]),
        "final_episodes": final_summary["episodes"], "final_item_predictions": len(final),
        "coverage": {name: sum(r[name] is not None for r in final) for name in ("raw", "old", "new")},
        "application_reasons": dict(Counter(r["new_reason"] for r in final)),
        "case_phase_changes": dict(Counter(m["change"] for m in metrics)),
        "metrics": metrics, "selection_sha256": sha(root/"selection.json"),
        "limitations": "No hidden reset detection; raw retention can worsen hidden-reset predictions. Calibration availability is not physical validity. Designed 2D sweep only; no IID confidence claims."}
    write_json(root/"summary.json", summary)
    text = ["# 선택적 보정 비교", "", "새 validation에서 적용 여부를 선택·고정한 뒤 별도 final test를 생성했다.",
        "ACK는 R만 확인한다. 새 ACK에서는 원래 내용 유효성/버전/만료 계산을 유지하며 K를 1로 강제하지 않는다.", "",
        "| 상황 | 시점 | 항목 | 전체 n | raw Brier | 기존 보정(n) | 수정 후(n) | 짝지은 변화 |",
        "|---|---|---|---:|---:|---:|---:|---|"]
    phases = {"after_pass", "at_ACK", "after_channel", "pending_delivery", "after_reset", "at_TTL", "after_version_change", "after_retry_ACK"}
    for m in metrics:
        if m["phase"] in phases:
            def fmt(v):
                return "N/A" if v is None else f"{v:.5f}"
            text.append(f"| {m['case']} | {m['phase']} | {m['kind']} | {m['samples']} | {fmt(m['raw_brier'])} | {fmt(m['old_brier'])} ({m['old_known']}) | {fmt(m['new_brier'])} ({m['new_known']}) | {m['change']} / n={m['paired_samples']} |")
    text += ["", "기존 보정이 없던 표본은 오류 감소로 세지 않고 지원 범위 증가로 구분한다.",
        "final_by_condition.csv는 악화된 조건도 모두 포함한다. 테스트를 보고 게이트를 재선택하지 않았다.", summary["limitations"]]
    (root/"RESULTS.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("knowledge_estimation/data/calibration_repair_v1"))
    args = parser.parse_args()
    result = run(args.output, model_root=args.model_root, plan=experiment_plan(args.model_root))
    print(json.dumps({k: result[k] for k in ("validation_episodes", "final_episodes", "coverage", "case_phase_changes")}, indent=2))


if __name__ == "__main__":
    main()
