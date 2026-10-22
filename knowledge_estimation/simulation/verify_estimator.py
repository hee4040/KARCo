"""Frozen receiver estimator audit: supported performance + explicit coverage.

python -m knowledge_estimation.simulation.verify_estimator --output knowledge_estimation/data/estimator_verification_v1
No fitting, model selection, rule-policy decisions, or robot performance claims.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path

from ..collect import MeasurementRecorder
from ..measurement import decision_pairs, label_before, load_dataset, write_json
from ..parameters import MissingMeasurement, ParameterStore
from ..probabilistic import KnowledgeEstimator
from ..probabilistic.projection import calibrated_value, source_key, validate_model
from ..protocol import Observation, item_to_dict
from .analyze import write_csv
from .run import items, meta, add_provenance, append_trace, snapshot, world, store, view_condition
from .world import blocked, run_pass, validate_config


CASES = (
    "pass_only", "normal_channel", "ack_lost_fault", "ttl_expiry", "reported_reset",
    "hidden_reset", "version_change", "repeat_pass", "retry_clear_path",
    "ack_without_pass", "unknown_route", "unknown_channel",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_plan(model_root):
    model_root = Path(model_root)
    config = json.loads((model_root/"experiment_plan.json").read_text(encoding="utf-8"))["config"]
    config = deepcopy(config)
    config["episode_namespace"] = "estimator-verification-v1"
    # Fresh actual geometry; existing calibrated planned-speed contexts retained.
    config["sweeps"] = {"test": {"offsets_m": [.45, .95, 1.45, 1.95],
                                   "headings_degrees": [22., 68., 158.], "speeds_mps": [.5, 1., 2.]}}
    return {"schema_version": 1, "origin": "simulation", "cases": list(CASES), "config": config,
        "model_sha256": sha(model_root/"selected_model.json"),
        "parameter_sha256": sha(model_root/"simulation_params.yaml"),
        "fit_policy": "Frozen existing model and source parameters; never fit or select on this audit",
        "sampling": "Designed geometry x intervention matrix; shared geometry is not independent repetition",
        "performance": "Brier/mean score/high-confidence false counts with coverage denominators; no guessed values for missing predictions",
        "fault_scope": "ack_lost_fault suppresses ACK after geometric delivery although fitted p_ack=1; deliberate model-mismatch stress, not a measured loss distribution",
        "unknown_scope": "unknown_route/channel deliberately reference unmeasured contexts; expected to abstain",
        "channel": "Frozen ideal LOS model except declared ACK suppression or retry move to sender position",
        "rule_policy": "not executed"}


class Verification:
    def __init__(self, writer, model):
        self.writer, self.model = writer, model
        self.predictions, self.checks = [], []

    def check(self, episode, name, condition, detail=""):
        self.checks.append({"episode_id": episode, "check": name, "passed": bool(condition), "detail": detail})

    def point(self, episode, case, phase, now, memory, epoch, content, details):
        writer = self.writer
        beliefs = writer.adapter.belief("Q", "S2", now=now)
        key = f"{episode}/{phase}"
        writer.record_decision(decision_id=key, meta=details["meta"], time=now, object_ref="Q", site_id="S2",
            event_id=key, task_version=1, attempt_id=1,
            selected_items=list("ENCI") if phase == "before_send" else [],
            required_items=list("ENCI") if phase == "before_send" else [],
            before_snapshot=snapshot(memory, now, epoch), world_before=world(content, now))
        writer.record_after_receipt(key, snapshot=snapshot(memory, now, epoch), world=world(content, now))
        truth = label_before(snapshot(memory, now, epoch), world(content, now), now)
        for kind, belief in (("N", beliefs.normal), ("C", beliefs.current), ("I", beliefs.appearance)):
            sources = [e.source for e in belief.evidence]
            reason, calibrated = "", None
            if belief.probability is None:
                reason = "unmeasured_raw_parameters"
            elif self.model is not None:
                try:
                    calibrated = calibrated_value(self.model, kind, belief.probability,
                        hard_invalid=belief.hard_invalid, sources=sources,
                        timestamps=[e.timestamp for e in belief.evidence], now=now)
                except MissingMeasurement as error:
                    reason = str(error)
            self.predictions.append({"episode_id": episode, "case": case, "phase": phase, "time": now,
                "geometry_id": details["geometry_id"], "speed_mps": details["speed_mps"],
                "route_standoff_m": details["standoff"], "heading_degrees": details["heading"],
                "walls": details["walls"], "kind": kind,
                "R": truth[kind]["R"], "K": truth[kind]["K"], "raw": belief.probability,
                "cal": calibrated, "calibration_reason": reason, "hard_invalid": belief.hard_invalid,
                "stored_probability": belief.stored_probability, "A_evidence_sources": source_key(sources),
                "truth_causes": ",".join(truth[kind]["causes"]), "A_epoch": beliefs.memory_epoch, "B_epoch": epoch})
        return beliefs


def run(output, *, model_root, plan, estimator_factory=KnowledgeEstimator, raw_only=False):
    model_root, root = Path(model_root), Path(output)
    model = json.loads((model_root/"selected_model.json").read_text(encoding="utf-8"))
    validate_model(model)
    if model["origin"] != "simulation":
        raise ValueError("this audit uses simulation models only")
    if sha(model_root/"selected_model.json") != plan["model_sha256"] or sha(model_root/"simulation_params.yaml") != plan["parameter_sha256"]:
        raise ValueError("model/parameter changed since audit plan creation")
    parameters = ParameterStore.load(model_root/"simulation_params.yaml", allow_simulation=True)
    config = plan["config"]
    validate_config(config)
    if config["protocol_id"] != model["protocol_id"]:
        raise ValueError("audit must preserve frozen calibration protocol")
    if not plan["cases"] or len(set(plan["cases"])) != len(plan["cases"]) or set(plan["cases"])-set(CASES):
        raise ValueError("unknown/repeated audit case")
    if KnowledgeEstimator(parameters=parameters).fingerprint != model["estimator_fingerprint"]:
        raise ValueError("raw estimator changed; audit requires frozen matching code/parameters")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"verification_plan.json", plan)
    write_json(root/"frozen_model.json", model)
    write_json(root/"frozen_parameters.yaml", parameters.artifact)
    writer = MeasurementRecorder(root/"episodes", protocol_id=config["protocol_id"], episode_id="audit-initial",
        origin="simulation", estimator=estimator_factory(parameters=parameters))
    add_provenance(writer, config)
    (writer.root/"model_verify_estimator.py").write_bytes(Path(__file__).read_bytes())
    writer.manifest["simulation_assumptions"]["policy"] = "fixed verification interventions, not rule/PPO"
    writer.manifest["frozen_model_id"] = model["model_id"]
    # In raw-only mode the existing artifact is checked solely for provenance
    # and disjoint groups. Its score map is never passed to the evaluator.
    audit = Verification(writer, None if raw_only else model)
    sweep = config["sweeps"]["test"]
    geometries = list(product(sweep["offsets_m"], sweep["headings_degrees"], (False, True), sweep["speeds_mps"]))
    # One physical pass design is reused across interventions, not claimed as
    # independent extra evidence. Outcomes come from geometry, never from p_acq.
    for index, (offset, heading, walls, speed) in enumerate(geometries):
        episode_config = {**config, "speed_mps": speed}
        duration, _ = validate_config(episode_config)
        physical_pass = run_pass(episode_config, offset, heading, walls)
        delivered = not blocked(config["sender_xy"], physical_pass.final_pose[:2], physical_pass.walls)
        for case in plan["cases"]:
            case_config = {**config, "episode_namespace": config["episode_namespace"]+"/"+case}
            split = plan.get("split", "test")
            if split not in {"validation", "test"}:
                raise ValueError("audit split must be validation or test")
            metadata = meta(case_config, split, index)
            episode = metadata["episode_id"]
            for field in ("episode_id", "session_id", "object_instance", "route_group_id"):
                if any(metadata[field] in model.get(name, {}).get(field, []) for name in ("groups", "source_fit_groups", "application_validation_groups")):
                    raise ValueError(f"audit overlaps frozen fitting groups in {field}")
            writer.begin_episode(episode, estimator=estimator_factory(parameters=parameters))
            content, memory, epoch = items(config), {}, 0
            details = {"meta": metadata, "geometry_id": f"geometry-{index:03d}", "speed_mps": speed,
                       "heading": heading, "walls": walls, "standoff": round(abs(config["object_xy"][1]-offset), 10)}
            trace = {"episode_id": episode, "case": case, "offset": offset, "heading": heading,
                "speed_mps": speed, "walls": physical_pass.walls, "frames": physical_pass.frames,
                "acquired_at": physical_pass.acquired_at if case != "ack_without_pass" else {},
                "delivered": delivered, "hidden_reset": case == "hidden_reset"}
            if case == "ack_without_pass":
                trace["frames"] = []  # No claim that this case physically ran an acquisition pass.
            if case == "retry_clear_path":
                trace["retry_pose_xy"] = config["sender_xy"]
            append_trace(writer, trace)
            for item in content.values():
                writer.observe(Observation("version", 0., {"item": item_to_dict(item)}))
            audit.point(episode, case, "initial", 0., memory, epoch, content, details)
            passive = case != "ack_without_pass"
            if passive:
                for kind, at in physical_pass.acquired_at.items():
                    store(memory, content[kind], at)
                route = "unmeasured-route" if case == "unknown_route" else "table-route"
                writer.observe(Observation("pass_by", duration, {"pass_id": "p1", "items": [item_to_dict(content[k]) for k in "NI"],
                    "normal_before": True, "complete": True, "route_id": route, "site_id": "S2",
                    "condition_id": view_condition(config, speed), "dependency_group": "shared-static-occluder"}))
                writer.observe(Observation("pass_confirmation", duration, {"pass_id": "p1", "normal_after": True}))
                first = audit.point(episode, case, "after_pass", duration, memory, epoch, content, details)
            if case in {"pass_only", "unknown_route"}:
                if case == "unknown_route":
                    audit.check(episode, "unmeasured_route_abstains", first.normal.probability is None and first.appearance.probability is None)
                continue
            if case == "repeat_pass":
                writer.observe(Observation("pass_by", 2*duration, {"pass_id": "p2", "items": [item_to_dict(content[k]) for k in "NI"],
                    "normal_before": True, "complete": True, "route_id": "table-route", "site_id": "S2",
                    "condition_id": view_condition(config, speed), "dependency_group": "shared-static-occluder"}))
                writer.observe(Observation("pass_confirmation", 2*duration, {"pass_id": "p2", "normal_after": True}))
                second = audit.point(episode, case, "second_same_pass", 2*duration, memory, epoch, content, details)
                audit.check(episode, "shared_pass_does_not_inflate_NI", first.normal.probability == second.normal.probability
                    and first.appearance.probability == second.appearance.probability)
                continue

            sent = round(duration+config["dt_seconds"], 10)
            content = items(config, current_time=sent, current_version=2)
            writer.observe(Observation("version", sent, {"item": item_to_dict(content["C"])}))
            audit.point(episode, case, "before_send", sent, memory, epoch, content, details)
            payload = [item_to_dict(content[k]) for k in "NCI"]
            channel = "unmeasured-channel" if case == "unknown_channel" else "los-channel"
            send_data = {"msg_id": "m", "items": payload, "channel_condition": channel,
                         "content_condition": "ideal-content"}
            writer.observe(Observation("send", sent, send_data))
            task_bytes = json.dumps({"type": "TASK", "event_id": episode, "items": payload}, sort_keys=True).encode()
            writer.record_transport(direction="A_to_B", message_kind="TASK", encoded_payload=task_bytes,
                decision_id=f"{episode}/before_send", category="initial", time=sent, is_retry=False)
            audit.point(episode, case, "pending_delivery", sent, memory, epoch, content, details)
            received_at = round(sent+config["one_way_delay_seconds"], 10)
            ack_at = round(sent+2*config["one_way_delay_seconds"], 10)
            if delivered:
                for item in content.values():
                    store(memory, item, received_at)
            acked = delivered and case != "ack_lost_fault"
            ack_data = {"msg_id": "m", "stored_items": payload, "memory_epoch": 0, "received_at": received_at}
            if acked:
                writer.observe(Observation("ack", ack_at, ack_data))
                writer.record_transport(direction="B_to_A", message_kind="ACK", encoded_payload=json.dumps(ack_data, sort_keys=True).encode(),
                    decision_id=f"{episode}/before_send", category="roundtrip", time=ack_at, is_retry=False)
                audit.point(episode, case, "at_ACK", ack_at, memory, epoch, content, details)
            settled = round(sent+config["ack_deadline_seconds"], 10)
            if not acked:
                writer.observe(Observation("timeout", settled, {"msg_id": "m"}))
            after = audit.point(episode, case, "after_channel", settled, memory, epoch, content, details)
            if acked:
                audit.check(episode, "ACK_confirms_storage_R", all(b.stored_probability == 1. for b in (after.normal, after.current, after.appearance)))
            elif case != "unknown_channel":
                audit.check(episode, "timeout_replaces_receipt_C_prior", after.current.probability == 0.,
                            "Frozen measured-in-simulation p_recv=.5, p_ack=1 imply receipt posterior=0; deliberate ACK-loss faults violate this model")
            if case in {"normal_channel", "ack_without_pass", "ack_lost_fault", "unknown_channel"}:
                continue
            changed = round(settled+config["dt_seconds"], 10)
            if case in {"reported_reset", "hidden_reset"}:
                memory.clear()
                epoch = 1
                if case == "reported_reset":
                    writer.observe(Observation("status", changed, {"memory_epoch": 1}))
                reset = audit.point(episode, case, "after_reset", changed, memory, epoch, content, details)
                if case == "reported_reset":
                    audit.check(episode, "reported_reset_clears_all_K", all(v == 0. for v in reset.as_policy_input().values()))
                    if acked:
                        writer.observe(Observation("ack", changed, ack_data))
                        late = audit.point(episode, case, "stale_epoch_ack", changed, memory, epoch, content, details)
                        audit.check(episode, "old_epoch_ACK_cannot_revive", all(v == 0. for v in late.as_policy_input().values()))
                else:
                    audit.check(episode, "hidden_reset_does_not_leak_to_A", reset.as_policy_input() == after.as_policy_input())
            elif case == "ttl_expiry":
                expiry = round(sent+config["current_ttl_seconds"], 10)
                expired = audit.point(episode, case, "at_TTL", expiry, memory, epoch, content, details)
                audit.check(episode, "C_expires_by_observation_time", expired.current.probability == 0. and expired.current.hard_invalid)
                if delivered:
                    audit.check(episode, "expired_C_preserves_R", expired.current.stored_probability == 1.)
            elif case == "version_change":
                for kind, old in list(content.items()):
                    content[kind] = replace(old, version=old.version+1, observed_at=changed)
                    writer.observe(Observation("version", changed, {"item": item_to_dict(content[kind])}))
                newer = audit.point(episode, case, "after_version_change", changed, memory, epoch, content, details)
                audit.check(episode, "new_versions_start_without_K", all(v == 0. for v in newer.as_policy_input().values()))
                if acked:
                    writer.observe(Observation("ack", changed, ack_data))
                    late = audit.point(episode, case, "stale_version_ack", changed, memory, epoch, content, details)
                    audit.check(episode, "old_version_ACK_cannot_revive", all(v == 0. for v in late.as_policy_input().values()))
            elif case == "retry_clear_path":
                # Same serialized task/content, receiver moves to sender: now LOS.
                writer.observe(Observation("send", changed, {**send_data, "attempt_id": 2}))
                writer.record_transport(direction="A_to_B", message_kind="TASK", encoded_payload=task_bytes,
                    decision_id=f"{episode}/before_send", category="roundtrip", time=changed, is_retry=True)
                retry_received = round(changed+config["one_way_delay_seconds"], 10)
                retry_ack = round(changed+2*config["one_way_delay_seconds"], 10)
                for item in content.values():
                    store(memory, item, retry_received)
                writer.observe(Observation("ack", retry_ack, {**ack_data, "received_at": retry_received}))
                writer.record_transport(direction="B_to_A", message_kind="ACK",
                    encoded_payload=json.dumps({**ack_data, "received_at": retry_received}, sort_keys=True).encode(),
                    decision_id=f"{episode}/before_send", category="roundtrip", time=retry_ack, is_retry=True)
                retried = audit.point(episode, case, "after_retry_ACK", retry_ack, memory, epoch, content, details)
                audit.check(episode, "retry_ACK_confirms_same_payload", all(v == 1. for v in retried.as_policy_input().values()))
                audit.check(episode, "retry_does_not_refresh_C_observed_at", retried.current.item.observed_at == sent)
                audit.check(episode, "same_message_evidence_not_duplicated", len([e for e in retried.current.evidence if e.evidence_id == "message:m"]) == 1)

    writer.finalize()
    # Verify collection pairing and truth independently of prediction output.
    pairs = list(decision_pairs(writer.root))
    if len(pairs)*3 != len(audit.predictions):
        raise ValueError("audit prediction/label pairing mismatch")
    if sha(model_root/"selected_model.json") != plan["model_sha256"] or sha(model_root/"simulation_params.yaml") != plan["parameter_sha256"]:
        raise ValueError("frozen model or parameters changed during audit")
    for rows in (audit.checks, audit.predictions):
        if not rows:
            raise ValueError("empty audit")
    write_csv(root/"predictions.csv", audit.predictions)
    write_csv(root/"logic_checks.csv", audit.checks)
    write_json(root/"logic_checks.json", audit.checks)
    groups = defaultdict(list)
    for row in audit.predictions:
        groups[(row["case"], row["phase"], row["kind"])].append(row)
    threshold = config["evaluation_protocol"]["high_confidence_threshold"]
    metrics = [metrics_for(case, phase, kind, rows, threshold) for (case, phase, kind), rows in groups.items()]
    write_csv(root/"metrics_by_case_phase.csv", metrics)
    ready = defaultdict(list)
    for row in audit.predictions:
        ready[(row["case"], row["phase"], row["episode_id"])].append(row)
    ready_groups = defaultdict(list)
    for (case, phase, episode), records in ready.items():
        ready_groups[(case, phase)].append(all(r["cal"] is not None for r in records))
    coverage = [{"case": case, "phase": phase, "decision_points": len(flags),
                 "all_NCI_supported": sum(flags), "coverage": sum(flags)/len(flags)}
                for (case, phase), flags in ready_groups.items()]
    write_csv(root/"coverage_by_case_phase.csv", coverage)
    reliability = []
    edges = config["evaluation_protocol"]["reliability_bin_edges"]
    for (case, phase, kind), rows in groups.items():
        for level in ("raw", "cal"):
            for i, (lower, upper) in enumerate(zip(edges, edges[1:])):
                selected = [r for r in rows if r[level] is not None and
                    (lower <= r[level] < upper or i == len(edges)-2 and r[level] == upper)]
                reliability.append({"case": case, "phase": phase, "kind": kind, "level": level,
                    "lower": lower, "upper": upper, "samples": len(selected),
                    "mean_prediction": sum(r[level] for r in selected)/len(selected) if selected else None,
                    "actual_K_rate": sum(r["K"] for r in selected)/len(selected) if selected else None})
    write_csv(root/"reliability_by_case_phase.csv", reliability)
    failures = [c for c in audit.checks if not c["passed"]]
    summary = {"origin": "simulation", "raw_only": raw_only,
        "raw_estimator_fingerprint": estimator_factory(parameters=parameters).fingerprint,
        "model_id": model["model_id"],
        "model_sha256": plan["model_sha256"], "parameter_sha256": plan["parameter_sha256"],
        "unique_geometry_designs": len(geometries), "cases": len(plan["cases"]),
        "episodes": len(geometries)*len(plan["cases"]), "decision_points": len(pairs),
        "item_predictions": len(audit.predictions), "logic_checks": len(audit.checks), "logic_failures": failures,
        "all_NCI_supported_decision_points": sum(all(r["cal"] is not None for r in records) for records in ready.values()),
        "calibrated_predictions": sum(r["cal"] is not None for r in audit.predictions),
        "unsupported_predictions": sum(r["cal"] is None for r in audit.predictions),
        "unsupported_reasons": dict(Counter(r["calibration_reason"] for r in audit.predictions if r["cal"] is None)),
        "case_metrics": metrics, "coverage_by_case_phase": coverage,
        "interpretation": "Logic checks, conditional predictive accuracy and model coverage are distinct; no overall success claim from one average. No rule policy, refit or hardware test."}
    write_json(root/"summary.json", summary)
    text = ["# 고정 추정기 검증", "", f"기하 조건 {len(geometries)}개 × 개입 {len(plan['cases'])}개 = {summary['episodes']} 에피소드.",
        f"논리 검사 {summary['logic_checks']}개 / 실패 {len(failures)}개. 항목별 보정 지원 {summary['calibrated_predictions']} / {summary['item_predictions']}.",
        "고정한 모델·확률을 다시 학습하지 않았다. 같은 기하 조건을 공유한 개입들을 독립 실측 표본으로 세지 않는다.", "",
        "| 상황 | 시점 | 항목 | 보정 지원/전체 | raw Brier | cal Brier(지원된 경우만) |", "|---|---|---|---:|---:|---:|"]
    for m in metrics:
        if m["phase"] in {"after_pass", "after_channel", "after_reset", "at_TTL", "after_version_change", "second_same_pass", "after_retry_ACK", "pending_delivery"}:
            def fmt(v):
                return "N/A" if v is None else f"{v:.5f}"
            text.append(f"| {m['case']} | {m['phase']} | {m['kind']} | {m['cal_known']} / {m['samples']} | {fmt(m['raw_brier'])} | {fmt(m['cal_brier'])} |")
    text += ["", "N/A는 미측정·미지원이며 0점 오류가 아니다. raw와 cal의 지원 표본이 다르면 Brier를 직접 비교하지 않는다.",
        "ack_lost_fault는 기존 채널 모델(p_ack=1)을 의도적으로 어긴 스트레스 조건이다.",
        "숨은 메모리 초기화는 A 입력을 바꾸지 않는다. 따라서 사후 보정으로 실제 reset을 탐지했다고 해석할 수 없다.",
        "보정이 지원되지 않는 항목이 있으면 해당 시점 전체를 실제 정책에 자동 제공할 수 없다.",
        "실측 D455 인식/깊이 잡음, 규칙 선택, 조작 성공률은 검증하지 않았다."]
    (root/"RESULTS.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    if failures:
        raise AssertionError(f"{len(failures)} estimator logic failures; inspect saved reports")
    return summary


def metrics_for(case, phase, kind, rows, threshold):
    result = {"case": case, "phase": phase, "kind": kind, "samples": len(rows),
              "K_rate": sum(r["K"] for r in rows)/len(rows),
              "constant_zero_brier": sum(r["K"] for r in rows)/len(rows),
              "constant_one_brier": sum(1-r["K"] for r in rows)/len(rows)}
    for level in ("raw", "cal"):
        known = [r for r in rows if r[level] is not None]
        high = [r for r in known if r[level] >= threshold]
        result.update({level+"_known": len(known), level+"_missing": len(rows)-len(known),
            level+"_brier": sum((r[level]-r["K"])**2 for r in known)/len(known) if known else None,
            level+"_mean": sum(r[level] for r in known)/len(known) if known else None,
            level+"_K_rate_on_known": sum(r["K"] for r in known)/len(known) if known else None,
            level+"_high_confidence_samples": len(high), level+"_high_confidence_false": sum(1-r["K"] for r in high)})
    joint = [r for r in rows if r["raw"] is not None and r["cal"] is not None]
    result["paired_known"] = len(joint)
    for level in ("raw", "cal"):
        result[level+"_brier_on_paired"] = sum((r[level]-r["K"])**2 for r in joint)/len(joint) if joint else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("knowledge_estimation/data/calibration_repair_v1"))
    args = parser.parse_args()
    summary = run(args.output, model_root=args.model_root, plan=default_plan(args.model_root))
    print(json.dumps({k: summary[k] for k in ("episodes", "decision_points", "logic_checks", "calibrated_predictions", "unsupported_predictions")}, indent=2))


if __name__ == "__main__":
    main()
