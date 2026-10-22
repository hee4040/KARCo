"""Collect geometry-driven 2D episodes, fit on separate runs and evaluate.

python -m knowledge_estimation.simulation.run --output knowledge_estimation/data/simulation_v1
"""

import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path

from ..calibrate import fit_sources, fit_scores
from ..collect import MeasurementRecorder, replay
from ..evaluate import evaluate
from ..measurement import TRIAL_FIELDS, decision_pairs, write_json
from ..parameters import ParameterStore
from ..protocol import Observation, item_to_dict
from ..types import Information, KnowledgeItem, KnowledgeKey
from .world import blocked, run_pass, validate_config


def items(config, *, current_time=0., current_version=1):
    """Exact simulator coordinates: zero uncertainty is an ideal-model assumption."""
    x, y = config["object_xy"]
    return {k: KnowledgeItem(KnowledgeKey(Information(k), "Q", "S2" if k == "N" else None),
        current_version if k == "C" else 1, frame_id="map", observed_at=current_time if k == "C" else 0.,
        ttl_seconds=config["current_ttl_seconds"] if k == "C" else None,
        content_id="ideal-template-Q" if k == "I" else None,
        value="ideal-template-Q" if k == "I" else (x, y, 0., 0., 0., 0.),
        uncertainty=None if k == "I" else (0.,)*6,
        registration_evidence_id="sim-evidence" if k == "N" else None) for k in "NCI"}


def meta(config, stage, index):
    prefix = f"{stage}-{index:03d}"
    if config.get("episode_namespace"):
        prefix = config["episode_namespace"]+"/"+prefix
    return {"split": stage if stage in {"validation", "test"} else "calibration", "episode_id": prefix,
            "session_id": prefix, "object_instance": prefix, "route_group_id": prefix,
            "origin": "simulation", "protocol_id": config["protocol_id"]}


def scenarios(config, stage):
    sweep = config["sweeps"][stage]
    return product(sweep["offsets_m"], sweep["headings_degrees"], (False, True),
                   (False,) if stage == "source" else (False, True),
                   sweep.get("speeds_mps", [config["speed_mps"]]))


def view_condition(config, speed):
    # Planned speed is A-observable; hidden heading/occlusion remain marginalized.
    if any("speeds_mps" in sweep for sweep in config["sweeps"].values()):
        return f"unreported-view:speed={speed:g}mps"
    return "unreported-view"


def add_provenance(writer, config):
    write_json(writer.root/"scenario.json", config)
    # Store exact executed model code, not only a path to potentially edited code.
    for name in ("world.py", "run.py"):
        (writer.root/f"model_{name}").write_bytes((Path(__file__).parent/name).read_bytes())
    (writer.root/"world_trace.jsonl").touch()
    writer.add_evidence("sim-evidence", "world_trace.jsonl",
                        description="Privileged geometry, sensor dwell and B DB trace; simulation only")
    writer.manifest["simulation_assumptions"] = {
        "scenario_sha256": hashlib.sha256((writer.root/"scenario.json").read_bytes()).hexdigest(),
        "probability_generation": "none; empirical counts from deterministic geometry sweep",
        "sensor": "ideal point-object sector + opaque AABB + contiguous dwell; exact identity/pose",
        "cache_reset": "full factorial reset/no-reset after channel without STATUS; designed stress case, not a measured rate",
        "policy": "fixed one ENCI send; estimator evaluation, not rule/PPO comparison",
        "independence": "disjoint episode IDs do not establish independent physical samples"}


def append_trace(writer, record):
    with (writer.root/"world_trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False)+"\n")


def snapshot(memory, time, epoch):
    return {"time": time, "memory_epoch": epoch, "items": deepcopy(memory)}


def world(content, time):
    return {"time": time, "assessment_id": "sim-evidence", "assessment_method": "simulator_ground_truth",
            "physical_object_id": "physical-Q", "frame_id": "map", "versions": {k: i.version for k, i in content.items()}}


def store(memory, item, received_at):
    entry = {"version": item.version, "physical_object_id": "physical-Q", "valid_content": True,
             "frame_id": item.frame_id, "observed_at": item.observed_at, "received_at": received_at,
             "evidence_ids": ["sim-evidence"]}
    if item.ttl_seconds is not None:
        entry["ttl_seconds"] = item.ttl_seconds
    memory[item.key.kind.value] = entry


def trial(writer, metadata, name, success, *, kind="", opportunity="pass", duration=None, deadline=None,
          pass_condition="unreported-view"):
    row = dict.fromkeys(TRIAL_FIELDS, "")
    row.update(metadata, receiver_id="B", source=name, success=str(int(success)),
               opportunity_id=opportunity, evidence_id="sim-evidence", kind=kind,
               site_id="S2" if name in {"p_acq", "F_pass"} else "",
               route_id="table-route" if name in {"p_acq", "F_pass"} else "",
               condition_id=pass_condition if name in {"p_acq", "F_pass"} else
                            "ideal-content" if name in {"r_crop", "r_current"} else "los-channel",
               duration_seconds="" if duration is None else str(duration),
               deadline_seconds="" if deadline is None else str(deadline))
    writer.record_trial(row)


def capture(output, config, stages, *, sources_only=False):
    writer = MeasurementRecorder(output, protocol_id=config["protocol_id"], episode_id="initial",
                                 origin="simulation")
    add_provenance(writer, config)
    validate_config(config)
    counts = {"episodes": 0, "decisions": 0, "acquired_N": 0, "acquired_I": 0, "delivered": 0}
    for stage in stages:
        for index, (offset, heading, walls, hidden_reset, speed) in enumerate(scenarios(config, stage)):
            episode_config = {**config, "speed_mps": speed}
            duration, _ = validate_config(episode_config)
            condition = view_condition(config, speed)
            metadata = meta(config, stage, index)
            writer.begin_episode(metadata["episode_id"])
            result = run_pass(episode_config, offset, heading, walls)
            counts["episodes"] += 1
            memory, epoch, content = {}, 0, items(config)
            for kind, time in result.acquired_at.items():
                store(memory, content[kind], time)
                counts[f"acquired_{kind}"] += 1
            delivered = not blocked(config["sender_xy"], result.final_pose[:2], result.walls)
            counts["delivered"] += int(delivered)
            append_trace(writer, {"episode_id": metadata["episode_id"], "offset": offset,
                "heading": heading, "walls": result.walls, "frames": result.frames,
                "acquired_at": result.acquired_at, "delivered": delivered,
                "hidden_reset": hidden_reset, "speed_mps": speed, "pass_condition": condition})
            if sources_only:
                for kind in "NI":
                    success = kind in result.acquired_at
                    trial(writer, metadata, "p_acq", success, kind=kind, pass_condition=condition)
                    if success:
                        trial(writer, metadata, "F_pass", True, kind=kind,
                              duration=result.acquired_at[kind], deadline=duration, pass_condition=condition)
                trial(writer, metadata, "p_recv", delivered, opportunity="packet", deadline=config["ack_deadline_seconds"])
                if delivered:
                    trial(writer, metadata, "p_ack", True, opportunity="packet", deadline=config["ack_deadline_seconds"])
                    trial(writer, metadata, "F_recv", True, opportunity="packet",
                          duration=config["one_way_delay_seconds"], deadline=config["ack_deadline_seconds"])
                    # Content was generated from exact world state, not a recognizer.
                    for name, kind in (("r_crop", "I"), ("r_current", "C")):
                        trial(writer, metadata, name, True, kind=kind, opportunity="packet")
                continue

            for item in content.values():
                writer.observe(Observation("version", 0., {"item": item_to_dict(item)}))

            def decision(phase, time, selected, state, state_epoch):
                counts["decisions"] += 1
                key = metadata["episode_id"]+"/"+phase
                before = snapshot(state, time, state_epoch)
                writer.record_decision(decision_id=key, meta=metadata, time=time, object_ref="Q", site_id="S2",
                    event_id=key, task_version=1, attempt_id=1, selected_items=selected,
                    required_items=list("ENCI") if selected else [], before_snapshot=before, world_before=world(content, time))
                return key

            def finish(key, time):
                writer.record_after_receipt(key, snapshot=snapshot(memory, time, epoch), world=world(content, time))

            # Initial memory is explicitly empty in this experimental protocol.
            key = decision("initial", 0., [], {}, 0)
            writer.record_after_receipt(key, snapshot=snapshot({}, 0., 0), world=world(content, 0.))
            writer.observe(Observation("pass_by", duration, {"pass_id": "p", "items": [item_to_dict(content[k]) for k in "NI"],
                "normal_before": True, "complete": True, "route_id": "table-route", "site_id": "S2",
                "condition_id": condition}))
            writer.observe(Observation("pass_confirmation", duration, {"pass_id": "p", "normal_after": True}))
            key = decision("after_pass", duration, [], memory, epoch)
            finish(key, duration)
            sent_at = round(duration+config["dt_seconds"], 10)
            content = items(config, current_time=sent_at, current_version=2)
            writer.observe(Observation("version", sent_at, {"item": item_to_dict(content["C"])}))
            key = decision("send", sent_at, list("ENCI"), memory, epoch)
            payload = [item_to_dict(content[k]) for k in "NCI"]
            writer.observe(Observation("send", sent_at, {"msg_id": "m", "items": payload,
                "channel_condition": "los-channel", "content_condition": "ideal-content"}))
            encoded = json.dumps({"type": "TASK", "event_id": key, "items": payload}, sort_keys=True).encode()
            writer.record_transport(direction="A_to_B", message_kind="TASK", encoded_payload=encoded,
                decision_id=key, category="initial", time=sent_at, is_retry=False)
            if delivered:
                received_at = round(sent_at+config["one_way_delay_seconds"], 10)
                ack_at = round(sent_at+2*config["one_way_delay_seconds"], 10)
                for item in content.values():
                    store(memory, item, received_at)
                ack = {"msg_id": "m", "stored_items": payload, "memory_epoch": epoch, "received_at": received_at}
                writer.observe(Observation("ack", ack_at, ack))
                writer.record_transport(direction="B_to_A", message_kind="ACK", encoded_payload=json.dumps(ack, sort_keys=True).encode(),
                    decision_id=key, category="roundtrip", time=ack_at, is_retry=False)
            else:
                writer.observe(Observation("timeout", round(sent_at+config["ack_deadline_seconds"], 10), {"msg_id": "m"}))
            settled_at = round(sent_at+config["ack_deadline_seconds"], 10)
            finish(key, settled_at)
            key = decision("after_channel", settled_at, [], memory, epoch)
            finish(key, settled_at)
            if hidden_reset:
                memory.clear()
                epoch += 1  # No STATUS emitted: A cannot read this truth change.
            reset_at = round(settled_at+config["dt_seconds"], 10)
            key = decision("after_possible_reset", reset_at, [], memory, epoch)
            finish(key, reset_at)
            expiry = round(sent_at+config["current_ttl_seconds"], 10)
            key = decision("expired", expiry, [], memory, epoch)
            finish(key, expiry)
    writer.finalize()
    return counts


def run(output, config):
    validate_config(config)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"scenario.json", config)
    source_counts = capture(root/"source_runs", config, ("source",), sources_only=True)
    artifact = fit_sources(root/"source_runs", allow_simulation=True)
    write_json(root/"simulation_params.yaml", artifact)
    raw_counts = capture(root/"decision_runs", config, ("calibration", "test"))
    replay(root/"decision_runs", root/"replayed", ParameterStore(artifact, allow_simulation=True), allow_simulation=True)
    model = fit_scores(root/"replayed", allow_simulation=True)
    write_json(root/"calibration.json", model)
    report = evaluate(root/"replayed", model, config["evaluation_protocol"], allow_simulation=True)
    test_pairs = list(decision_pairs(root/"replayed", split="test"))
    report["constant_baseline_brier"] = {name: {k: sum((p-truth[k]["K"])**2 for _, _, _, truth in test_pairs)/len(test_pairs)
        for k in "NCI"} for name, p in (("always_unknown", 0.), ("always_known", 1.))}
    report["scope"] = "Fixed send-all collection policy. No rule/PPO, detector accuracy, manipulation, or physical communication claims."
    report["uncertainty_scope"] = "Finite deterministic sweep; bootstrap/binomial confidence intervals suppressed because independent random sampling is not established."
    write_json(root/"report.json", report)
    summary = {"origin": "simulation", "source_runs": source_counts, "decision_runs": raw_counts,
        "test_decisions": report["decisions"], "test_episodes": report["episodes"],
        "brier": {level: {k: report["metrics"][level][k]["brier"] for k in "NCI"} for level in ("raw", "cal")},
        "report": str(root/"report.json"), "not_hardware_performance": True}
    write_json(root/"summary.json", summary)
    from .analyze import analyze
    analyze(root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("scenario.json"))
    parser.add_argument("--speeds-mps", type=float, nargs="+",
                        help="Explicit designed speeds, not measured hardware speeds")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.speeds_mps is not None:
            if len(set(args.speeds_mps)) != len(args.speeds_mps):
                raise ValueError("duplicate speed conditions do not add experiments")
            for sweep in config["sweeps"].values():
                sweep["speeds_mps"] = args.speeds_mps
            config["protocol_id"] += "-planned-speed-conditions"
        result = run(args.output, config)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Simulation stopped: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
