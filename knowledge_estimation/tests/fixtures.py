"""Small deterministic CSV fixtures for Notion section 12's plumbing test.

Every number in this file is test input. None is an operational default or a
measurement of a camera, robot, channel or acquisition probability.
"""

from copy import deepcopy
from pathlib import Path

from knowledge_estimation.collect import MeasurementRecorder
from knowledge_estimation.measurement import TRIAL_FIELDS
from knowledge_estimation.parameters import ParameterStore
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.protocol import Observation, item_to_dict
from knowledge_estimation.types import Information, KnowledgeItem, KnowledgeKey


def item(kind, *, version=1, observed_at=0., ttl=7.):
    return KnowledgeItem(KnowledgeKey(Information(kind), "Q", "S2" if kind == "N" else None),
        version, frame_id="tag:S2", observed_at=observed_at,
        ttl_seconds=ttl if kind == "C" else None, content_id="fixture-crop" if kind == "I" else None,
        value="fixture-template" if kind == "I" else (0., 0., 0., 0., 0., 0.),
        uncertainty=None if kind == "I" else (.01, .01, .01, .01, .01, .01),
        registration_evidence_id="fixture-evidence" if kind == "N" else None)


def metadata(split, episode=None):
    return {"split": split, "episode_id": episode or f"{split}-episode",
            "session_id": f"{split}-session", "object_instance": f"{split}-object",
            "route_group_id": f"{split}-route-session", "origin": "fixture", "protocol_id": "fixture-contract-v1"}


def snapshots(time, *, stored=False, epoch=0):
    entries = {}
    if stored:
        for kind in "NCI":
            entries[kind] = {"version": 1, "physical_object_id": "fixture-physical-Q", "valid_content": True,
                             "frame_id": "tag:S2", "observed_at": 0., "received_at": .5,
                             "evidence_ids": ["fixture-evidence"]}
            if kind == "C":
                entries[kind]["ttl_seconds"] = 7.
    snapshot = {"time": time, "memory_epoch": epoch, "items": entries}
    world = {"time": time, "assessment_id": "fixture-evidence", "assessment_method": "fixture",
             "physical_object_id": "fixture-physical-Q", "frame_id": "tag:S2",
             "versions": {k: 1 for k in "NCI"}}
    return snapshot, world


def add_evidence(writer):
    (writer.root / "raw_fixture.txt").write_text("TEST INPUT ONLY: scripted storage/identity outcomes; not measured data.", encoding="utf-8")
    writer.add_evidence("fixture-evidence", "raw_fixture.txt", description="explicit contract-test fixture")


def trial_rows():
    rows = []
    def add(source, success, opportunity, *, kind="", site="", route="", condition="fixture-channel", duration=""):
        row = dict.fromkeys(TRIAL_FIELDS, "")
        row.update(metadata("calibration", "source-episode"))
        row.update(receiver_id="B", source=source, success=str(success), opportunity_id=opportunity,
                   evidence_id="fixture-evidence", kind=kind, site_id=site, route_id=route,
                   condition_id=condition, duration_seconds=str(duration),
                   deadline_seconds="1" if source in {"p_recv", "p_ack", "F_recv", "F_pass", "F_task"} else "")
        rows.append(row)
    for j, (d, a) in enumerate(((1, 1), (1, 1), (1, 0), (0, None))):
        add("p_recv", d, f"packet-{j}")
        if d:
            add("p_ack", a, f"packet-{j}")
            add("F_recv", 1, f"packet-{j}", duration=(.2, .4, .8)[j])
    for kind in "NI":
        for j, y in enumerate((1, 1, 0, 0)):
            add("p_acq", y, f"pass-{j}", kind=kind, site="S2", route="route-S2", condition="fixture-pass")
            if y:
                add("F_pass", 1, f"pass-{j}", kind=kind, site="S2", route="route-S2", condition="fixture-pass", duration=(.4, .6)[j])
    for source, kind in (("r_crop", "I"), ("r_current", "C"), ("q_learn", "I")):
        for j, y in enumerate((1, 0)):
            add(source, y, f"{source}-{j}", kind=kind, condition="fixture-content")
    add("F_task", 1, "task-0", condition="fixture-task", duration=.8)
    add("F_task", 0, "task-1", condition="fixture-task")
    return rows


def source_dataset(output):
    writer = MeasurementRecorder(output, protocol_id="fixture-contract-v1", episode_id="source-episode", origin="fixture")
    add_evidence(writer)
    for row in trial_rows():
        writer.record_trial(row)
    writer.finalize()
    return writer.root


def decision_dataset(output, *, parameters=None, include_trials=True):
    parameters = parameters or ParameterStore()
    writer = MeasurementRecorder(output, protocol_id="fixture-contract-v1", episode_id="calibration-episode",
        origin="fixture", estimator=KnowledgeEstimator(parameters=parameters))
    add_evidence(writer)
    if include_trials:
        for row in trial_rows():
            writer.record_trial(row)
    items = [item(k) for k in "NCI"]
    for split in ("calibration", "test"):
        if split != "calibration":
            writer.begin_episode(f"{split}-episode", estimator=KnowledgeEstimator(parameters=parameters))
        for content in items:
            writer.observe(Observation("version", 0., {"item": item_to_dict(content)}))
        before, world = snapshots(0.)
        writer.record_decision(decision_id=f"{split}-d0", meta=metadata(split), time=0., object_ref="Q", site_id="S2",
            event_id="event-1", task_version=1, attempt_id=1, selected_items=list("ENCI"), required_items=list("ENCI"),
            before_snapshot=before, world_before=world)
        writer.observe(Observation("send", .1, {"msg_id": "m", "items": [item_to_dict(i) for i in items],
            "channel_condition": "fixture-channel", "content_condition": "fixture-content"}))
        writer.observe(Observation("ack", 1., {"msg_id": "m", "stored_items": [item_to_dict(i) for i in items],
                                                "memory_epoch": 0, "received_at": .5}))
        after, world_after = snapshots(1., stored=True)
        writer.record_after_receipt(f"{split}-d0", snapshot=after, world=world_after)
        writer.record_transport(direction="A_to_B", message_kind="TASK", encoded_payload=b'{"fixture":"TASK"}',
                                decision_id=f"{split}-d0", category="initial", time=.1, is_retry=False)
        for index, time in ((1, 2.), (2, 3.)):
            # A receives no STATUS: final-test d2 has a hidden epoch change.
            has_memory = not (split == "test" and index == 2)
            before, world = snapshots(time, stored=has_memory, epoch=0 if has_memory else 1)
            writer.record_decision(decision_id=f"{split}-d{index}", meta=metadata(split), time=time,
                object_ref="Q", site_id="S2", event_id=f"event-{index+1}", task_version=1, attempt_id=1,
                selected_items=["E"], required_items=list("ENCI"), before_snapshot=before, world_before=world)
            after, world_after = snapshots(time+.1, stored=has_memory, epoch=0 if has_memory else 1)
            writer.record_after_receipt(f"{split}-d{index}", snapshot=after, world=world_after)
    writer.finalize()
    return writer.root
