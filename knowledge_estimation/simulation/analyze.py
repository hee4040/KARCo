"""Offline condition analysis. Hidden geometry is joined AFTER prediction only.

python -m knowledge_estimation.simulation.analyze knowledge_estimation/data/d455_v1
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from ..measurement import load_dataset, read_jsonl, write_json


def conditions(trace, config):
    # The path crosses object x: absolute y separation is closest geometric
    # distance to the route, not necessarily distance at first detection.
    return {"speed_mps": trace.get("speed_mps", config["speed_mps"]),
            "route_standoff_m": round(abs(config["object_xy"][1]-trace["offset"]), 10),
            "heading_degrees": trace["heading"], "walls": bool(trace["walls"]),
            "hidden_reset": trace["hidden_reset"]}


def buckets(condition, *, acquisition=False):
    yield "all", "all"
    for name, value in condition.items():
        if acquisition and name == "hidden_reset":
            continue
        yield name, value


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty condition analysis")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    root = Path(root)
    config = json.loads((root/"scenario.json").read_text(encoding="utf-8"))
    report = json.loads((root/"report.json").read_text(encoding="utf-8"))
    if report["origin"] != "simulation":
        raise ValueError("this analyzer handles geometry simulation only")
    for name in ("source_runs", "decision_runs", "replayed"):
        manifest, _, _ = load_dataset(root/name, allow_simulation=True)
        if manifest["origin"] != "simulation":
            raise ValueError("mixed origins in simulation analysis")
    if report["model_id"] != json.loads((root/"calibration.json").read_text(encoding="utf-8"))["model_id"]:
        raise ValueError("report and calibration model differ")
    traces = {}
    for name in ("source_runs", "decision_runs"):
        for trace in read_jsonl(root/name/"world_trace.jsonl"):
            if trace["episode_id"] in traces:
                raise ValueError("duplicate simulation episode trace")
            traces[trace["episode_id"]] = trace
    acquired = defaultdict(list)
    seen = set()
    for episode, trace in traces.items():
        stage = episode.rsplit("/", 1)[-1].rsplit("-", 1)[0]
        c = conditions(trace, config)
        # Reset/no-reset episodes share the exact same preceding pass. Count
        # that physical pass design once, not as two independent observations.
        identity = (stage, c["speed_mps"], trace["offset"], c["heading_degrees"],
                    json.dumps(trace["walls"], sort_keys=True))
        if identity in seen:
            continue
        seen.add(identity)
        for factor, value in buckets(c, acquisition=True):
            for kind in "NI":
                acquired[(stage, factor, value, kind)].append(trace["acquired_at"].get(kind))
    acquisition = []
    for (stage, factor, value, kind), times in acquired.items():
        successes = [t for t in times if t is not None]
        acquisition.append({"stage": stage, "factor": factor, "value": value, "kind": kind,
            "unique_pass_designs": len(times), "acquired": len(successes),
            "acquisition_rate": len(successes)/len(times),
            "mean_first_store_seconds_given_acquired": sum(successes)/len(successes) if successes else None})
    inputs = {r["decision_id"]: r for r in read_jsonl(root/"replayed"/"inputs.jsonl")}
    estimation, rows = defaultdict(list), []
    threshold = report["protocol"]["high_confidence_threshold"]
    for result in report["paired_decision_rows"]:
        entry = inputs[result["decision_id"]]
        if entry["split"] != "test":
            raise ValueError("expected untouched test predictions")
        trace = traces[entry["episode_id"]]
        c = conditions(trace, config)
        phase = entry["decision_id"].rsplit("/", 1)[1]
        for kind in "NCI":
            row = {"decision_id": entry["decision_id"], "episode_id": entry["episode_id"],
                   "phase": phase, **c, "kind": kind, "K": result["K"][kind],
                   "raw": result["raw"][kind], "cal": result["cal"][kind]}
            rows.append(row)
            for factor, value in buckets(c):
                estimation[(phase, factor, value, kind)].append(row)
    errors = []
    for (phase, factor, value, kind), group in estimation.items():
        item = {"phase": phase, "factor": factor, "value": value, "kind": kind,
                "samples": len(group), "episodes": len({g["episode_id"] for g in group}),
                "actual_K_rate": sum(g["K"] for g in group)/len(group)}
        for level in ("raw", "cal"):
            high = [g for g in group if g[level] >= threshold]
            item.update({f"{level}_mean": sum(g[level] for g in group)/len(group),
                         f"{level}_brier": sum((g[level]-g["K"])**2 for g in group)/len(group),
                         f"{level}_high_confidence_samples": len(high),
                         f"{level}_high_confidence_false": sum(1-g["K"] for g in high),
                         f"{level}_high_confidence_false_rate": sum(1-g["K"] for g in high)/len(high) if high else None})
        errors.append(item)
    result = {"origin": "simulation", "model_id": report["model_id"],
        "report_sha256": hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),
        "scope": "Designed condition sweep, not physical camera accuracy; no IID confidence intervals.",
        "A_observable_condition": "planned speed in pass condition_id; heading/walls/reset and evaluation labels are offline-only",
        "acquisition_denominator": "unique preceding pass designs, reset/no-reset duplicates collapsed",
        "distance_definition": "minimum planar distance between the point object and straight route, not depth at detection",
        "acquisition": acquisition, "estimation": errors}
    write_json(root/"condition_report.json", result)
    write_csv(root/"acquisition_by_condition.csv", acquisition)
    write_csv(root/"estimation_by_condition.csv", errors)
    write_csv(root/"decision_condition_rows.csv", rows)
    text = ["# D455 2D 조건별 결과", "", "설계한 시뮬레이션 조건의 결과이며 실측 성능이 아니다.", "",
            "통과 획득률은 메모리 초기화 유무로 중복 실행한 동일 통과를 한 번만 센다.", "",
            "| 분할 | 조건 | 값 | 항목 | 획득 / 고유 통과 | 획득률 |", "|---|---|---|---|---:|---:|"]
    for a in acquisition:
        if a["factor"] in {"speed_mps", "walls", "route_standoff_m"} and a["stage"] in {"source", "test"}:
            text.append(f"| {a['stage']} | {a['factor']} | {a['value']} | {a['kind']} | {a['acquired']} / {a['unique_pass_designs']} | {a['acquisition_rate']:.3f} |")
    text += ["", "통과 직후 시점의 test 오류. 낮은 Brier가 좋으며 평균과 실제 보유율도 함께 비교한다.", "",
             "| 조건 | 값 | 항목 | 표본 | 실제 K 비율 | raw 평균 | cal 평균 | raw Brier | cal Brier |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for e in errors:
        if e["phase"] == "after_pass" and e["factor"] in {"speed_mps", "walls"} and e["kind"] in "NI":
            text.append(f"| {e['factor']} | {e['value']} | {e['kind']} | {e['samples']} | {e['actual_K_rate']:.3f} | {e['raw_mean']:.3f} | {e['cal_mean']:.3f} | {e['raw_brier']:.3f} | {e['cal_brier']:.3f} |")
    text += ["", "전체 단계·방향·거리·재시작별 상세 수치는 CSV/JSON을 따른다.",
             "가림을 바꾸면 채널도 바뀌는 설계이므로 통과 이후 전체 오류를 순수 시각 가림 효과로 해석하지 않는다.",
             "규칙 선택의 실행 결과와 작업 성공률은 이 보고서에 포함하지 않는다."]
    (root/"CONDITIONS.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    result = analyze(parser.parse_args().dataset)
    print(json.dumps({"origin": result["origin"], "acquisition_groups": len(result["acquisition"]),
                      "estimation_groups": len(result["estimation"])}))


if __name__ == "__main__":
    main()
