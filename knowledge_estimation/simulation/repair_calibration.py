"""Diagnose pooled calibration, select on fresh validation, open fresh test last.

python -m knowledge_estimation.simulation.repair_calibration --output knowledge_estimation/data/calibration_repair_v1
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ..calibrate import fit_sources, fit_scores
from ..collect import replay
from ..evaluate import evaluate
from ..measurement import decision_pairs, load_dataset, read_jsonl, write_json
from ..parameters import ParameterStore
from ..probabilistic.projection import calibrated_value, source_key, validate_model
from .analyze import conditions, buckets, write_csv
from .run import capture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(root):
    """Old calibration/test are descriptive evidence, never new fit/selection."""
    root = Path(root)
    load_dataset(root/"replayed", allow_simulation=True)
    model = json.loads((root/"calibration.json").read_text(encoding="utf-8"))
    validate_model(model)
    groups = defaultdict(list)
    for row, _, _, truth in decision_pairs(root/"replayed", split="calibration"):
        phase = row["decision_id"].rsplit("/", 1)[-1]
        for k in "NCI":
            if row["hard_invalid"][k]:
                continue
            score = row["b_raw"]["b_"+k]
            sources = source_key(e["source"] for e in row["evidence"][k])
            groups[(k, score, phase, sources)].append(truth[k]["K"])
    rows = []
    for (kind, score, phase, sources), labels in groups.items():
        rows.append({"kind": kind, "raw": score, "phase_for_diagnosis_only": phase,
            "A_evidence_sources": sources, "samples": len(labels), "actual_K_rate": sum(labels)/len(labels),
            "pooled_cal": calibrated_value(model, kind, score, hard_invalid=False, sources=json.loads(sources))})
    return {"old_model_id": model["model_id"], "old_manifest_sha256": digest(root/"replayed"/"manifest.json"),
            "cause": "Pooled PAV fits decision-weighted K frequencies across unlike A evidence histories. Equal raw scores after pass and after timeout/reset have different label rates.",
            "scope": "phase and privileged truth used for diagnosis only; model conditions on A evidence source types only",
            "groups": rows}


def build_plan(base):
    config = deepcopy(base)
    config["protocol_id"] = "d455-source-conditioned-calibration-repair-v1"
    config["episode_namespace"] = "repair-v1"
    # Predeclared designed points: no reuse of old source/calibration/test grid.
    points = {"source": ([.05, .55, 1.05, 1.55], [2., 88., 178.]),
              "calibration": ([.15, .65, 1.15, 1.65], [6., 84., 174.]),
              "validation": ([.25, .75, 1.25, 1.75], [12., 78., 168.]),
              "test": ([.35, .85, 1.35, 1.85], [16., 74., 164.])}
    config["sweeps"] = {stage: {"offsets_m": offset, "headings_degrees": angle, "speeds_mps": [.5, 1., 2.]}
                        for stage, (offset, angle) in points.items()}
    return {"schema_version": 1, "origin": "simulation", "config": config,
        "candidates": ["pooled", "evidence_sources_v1"],
        "selection": "Min validation mean(N,I) Brier at after_pass; tie by overall mean(N,C,I) Brier, then candidate name. No test labels or hidden variables in selection.",
        "primary": "after_pass mean N/I Brier", "secondary": "overall mean N/C/I Brier",
        "test_access": "Generate fresh final test only after persisting selected model ID and validation metrics. Compare frozen selected candidate to frozen pooled baseline; never reselect on test.",
        "sampling": "Deterministic designed grid; no IID CI or hardware generalization claim",
        "old_data_use": "diagnosis only; no old records in fitting/validation/final test"}


def summarize(report):
    groups = defaultdict(list)
    for row in report["paired_decision_rows"]:
        phase = row["decision_id"].rsplit("/", 1)[-1]
        groups[phase].append(row)
    phases = {}
    for phase, rows in groups.items():
        phases[phase] = {"decisions": len(rows), **{level: {k: sum((r[level][k]-r["K"][k])**2 for r in rows)/len(rows)
                                                      for k in "NCI"} for level in ("raw", "cal")}}
    overall = {level: {k: report["metrics"][level][k]["brier"] for k in "NCI"} for level in ("raw", "cal")}
    return {"phases": phases, "overall": overall,
            "primary": sum(phases["after_pass"]["cal"][k] for k in "NI")/2,
            "secondary": sum(overall["cal"].values())/3}


def choose(validation):
    return min(validation, key=lambda name: (validation[name]["primary"], validation[name]["secondary"], name))


def check_disjoint(roots):
    seen = {field: set() for field in ("episode_id", "session_id", "object_instance", "route_group_id")}
    for root in roots:
        _, trials, inputs = load_dataset(root, allow_simulation=True)
        for field in seen:
            values = {r[field] for r in trials+inputs}
            if seen[field] & values:
                raise ValueError(f"experiment datasets overlap in {field}")
            seen[field].update(values)


def condition_comparison(root, config, reports):
    traces = {r["episode_id"]: r for r in read_jsonl(root/"test_runs"/"world_trace.jsonl")}
    groups = defaultdict(list)
    for name, report in reports.items():
        for r in report["paired_decision_rows"]:
            phase = r["decision_id"].rsplit("/", 1)[-1]
            for factor, value in buckets(conditions(traces[r["episode_id"]], config)):
                for k in "NCI":
                    groups[(name, phase, factor, value, k)].append(r)
    rows = []
    for (name, phase, factor, value, k), records in groups.items():
        rows.append({"candidate": name, "phase": phase, "factor": factor, "value": value, "kind": k,
            "decisions": len(records), "actual_K_rate": sum(r["K"][k] for r in records)/len(records),
            "raw_brier": sum((r["raw"][k]-r["K"][k])**2 for r in records)/len(records),
            "cal_brier": sum((r["cal"][k]-r["K"][k])**2 for r in records)/len(records)})
    return rows


def run(output, *, old_data, plan):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    # Freeze design and implementation before collecting any new labels.
    write_json(root/"experiment_plan.json", plan)
    write_json(root/"implementation_sha256.json", {name: digest(Path(__file__).parents[1]/name)
        for name in ("calibrate.py", "evaluate.py", "probabilistic/projection.py", "simulation/run.py",
                     "simulation/world.py", "simulation/repair_calibration.py")})
    write_json(root/"diagnosis.json", diagnose(old_data))
    config = plan["config"]
    counts = {}
    counts["source"] = capture(root/"source_runs", config, ("source",), sources_only=True)
    params = fit_sources(root/"source_runs", allow_simulation=True)
    write_json(root/"simulation_params.yaml", params)
    store = ParameterStore(params, allow_simulation=True)
    models = {}
    counts["calibration"] = capture(root/"calibration_runs", config, ("calibration",))
    replay(root/"calibration_runs", root/"calibration_replayed", store, allow_simulation=True)
    for name in plan["candidates"]:
        models[name] = fit_scores(root/"calibration_replayed", allow_simulation=True, conditioning=name)
        write_json(root/f"model_{name}.json", models[name])
    counts["validation"] = capture(root/"validation_runs", config, ("validation",))
    replay(root/"validation_runs", root/"validation_replayed", store, allow_simulation=True)
    check_disjoint([root/"source_runs", root/"calibration_runs", root/"validation_runs"])
    validation = {}
    for name, model in models.items():
        report = evaluate(root/"validation_replayed", model, config["evaluation_protocol"],
                          split="validation", allow_simulation=True)
        write_json(root/f"validation_{name}.json", report)
        validation[name] = summarize(report)
    selected = choose(validation)
    selection = {"selected": selected, "selected_model_id": models[selected]["model_id"],
        "plan_sha256": digest(root/"experiment_plan.json"), "validation": validation,
        "validation_manifest_sha256": digest(root/"validation_replayed"/"manifest.json"),
        "final_test_exists_when_selecting": (root/"test_runs").exists()}
    if selection["final_test_exists_when_selecting"]:
        raise ValueError("test was opened before model selection")
    write_json(root/"selection.json", selection)
    write_json(root/"selected_model.json", models[selected])
    # This is the FIRST creation/access of final-test labels.
    counts["test"] = capture(root/"test_runs", config, ("test",))
    replay(root/"test_runs", root/"test_replayed", store, allow_simulation=True)
    check_disjoint([root/f"{stage}_runs" for stage in ("source", "calibration", "validation", "test")])
    final, reports = {}, {}
    for name in dict.fromkeys(("pooled", selected)):
        report = evaluate(root/"test_replayed", models[name], config["evaluation_protocol"], allow_simulation=True)
        write_json(root/f"test_{name}.json", report)
        reports[name], final[name] = report, summarize(report)
    if digest(root/"selected_model.json") != digest(root/f"model_{selected}.json"):
        raise ValueError("selected model changed after validation")
    write_csv(root/"test_by_condition.csv", condition_comparison(root, config, reports))
    summary = {"origin": "simulation", "counts": counts,
        "episodes_total": sum(c["episodes"] for c in counts.values()),
        "selected": selected, "selection_sha256": digest(root/"selection.json"),
        "validation": validation, "test": final,
        "limitations": "Ideal D455 planar model and designed sweep only; evidence-source conditioning cannot detect an unreported reset or guarantee every condition improves."}
    write_json(root/"summary.json", summary)
    lines = ["# 보정 수정 검증", "", f"선택: {selected}. 새 에피소드 {summary['episodes_total']}개.",
        "validation에서 선택을 고정한 뒤 최종 test를 생성했다. 기존 데이터는 원인 분석에만 사용했다.", "",
        "A가 관측한 증거 종류만 보정에 사용한다. phase·숨은 reset·벽·B 정답은 모델 입력이 아니다.", "",
        "| 평가 | 보정 방식 | 통과 직후 N Brier | 통과 직후 I Brier | 전체 N | 전체 C | 전체 I |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for split, results in (("validation", validation), ("final test", final)):
        for name, result in results.items():
            p, overall = result["phases"]["after_pass"]["cal"], result["overall"]["cal"]
            lines.append(f"| {split} | {name} | {p['N']:.5f} | {p['I']:.5f} | {overall['N']:.5f} | {overall['C']:.5f} | {overall['I']:.5f} |")
        ref = next(iter(results.values()))
        p, overall = ref["phases"]["after_pass"]["raw"], ref["overall"]["raw"]
        lines.append(f"| {split} | raw 기준선 | {p['N']:.5f} | {p['I']:.5f} | {overall['N']:.5f} | {overall['C']:.5f} | {overall['I']:.5f} |")
    lines += ["", "낮은 Brier가 좋다. test_by_condition.csv에서 속도·거리·방향·벽·reset별 차이를 확인한다.",
        "전체 개선을 모든 조건 개선으로 해석하지 않는다. test 결과를 보고 재선택/재적합하지 않았다.",
        summary["limitations"]]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-data", type=Path, default=Path("knowledge_estimation/data/d455_conditions_v1"))
    args = parser.parse_args()
    base = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    summary = run(args.output, old_data=args.old_data, plan=build_plan(base))
    print(json.dumps({"origin": summary["origin"], "episodes": summary["episodes_total"],
                      "selected": summary["selected"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
