"""Learning input features from a causal A-side prefix, without world access.

Identifiers, scenario names, seeds, physical configurations and truth labels
are deliberately absent from FEATURE_NAMES. A baseline probability is saved
separately for comparison, not used as a learned model's input.
"""

from math import cos, isfinite


_KINDS = "NCI"
_COUNTS = ("send", "ack", "timeout", "pass", "pass_confirmed", "pass_rejected",
           "result", "verified", "missing", "not_found", "revoke")
_AGES = ("send", "ack", "timeout", "pass", "verified", "missing", "version")
FEATURE_NAMES = (
    "epoch_changes", "epoch_age", "planned_distance", "planned_bearing_cos",
    "camera_range", "camera_fov", "planned_in_range", "planned_in_fov",
) + tuple(f"{kind}_{name}_count" for kind in _KINDS for name in _COUNTS) + tuple(
    f"{kind}_{name}_age" for kind in _KINDS for name in _AGES
) + tuple(f"{kind}_pending_count" for kind in _KINDS)


def extract_features(history, view):
    """history must end at view.history_end, even for events at the same time.

    Counts describe current known epoch/version only. Ages are seconds; -1
    means no matching observation. Unknown memory resets/physical changes must
    not clear them. 'pass' describes an opportunity, never actual acquisition.
    """
    if len(history) != view["history_end"]:
        raise ValueError("history must be exactly the sampled observation prefix")
    now, target = view["time"], view["target"]
    versions, epoch = view["known_versions"], view["known_epoch"]
    counters = {kind: {name: [] for name in _COUNTS + ("version",)} for kind in _KINDS}
    sends, passes, tasks = {}, {}, {}
    seen_acks, timed_out = set(), set()
    active_epoch, epoch_times, previous_time = 0, [], -1.

    def add(name, items, time, item_epoch):
        if item_epoch != epoch:
            return
        for item in items:
            kind = item["kind"]
            if item["object_id"] == target and item["version"] == versions[kind]:
                counters[kind][name].append(time)

    for event in history:
        time, kind, data = event["timestamp"], event["kind"], event["data"]
        if time < previous_time or time > now:
            raise ValueError("history must be chronological and contain no future events")
        previous_time = time
        announced_epoch = data.get("memory_epoch", active_epoch)
        if announced_epoch > active_epoch:
            active_epoch = announced_epoch
            epoch_times.append(time)
        if announced_epoch < active_epoch:
            continue
        if kind == "version":
            # Version registration does not imply B knows the item. A version
            # can remain known across epoch changes, unlike acquisition events.
            add("version", [data["item"]], time, epoch)
        elif kind == "send":
            sends[data["msg_id"]] = (data["items"], active_epoch, data.get("attempt_id", 1))
            add("send", data["items"], time, active_epoch)
        elif kind == "ack":
            seen_acks.add(data["msg_id"])
            add("ack", data["stored_items"], time, active_epoch)
            add("missing", data.get("needs_info", []), time, active_epoch)
        elif kind == "timeout":
            items, sent_epoch, attempt = sends.get(data["msg_id"], ([], -1, 0))
            timed_out.add((data["msg_id"], data.get("attempt_id", attempt)))
            add("timeout", items, time, sent_epoch)
        elif kind == "pass_by":
            passes[data["pass_id"]] = (data["items"], active_epoch)
            add("pass", data["items"], time, active_epoch)
        elif kind == "pass_confirmation":
            items, pass_epoch = passes.get(data["pass_id"], ([], -1))
            add("pass_confirmed" if data["normal_after"] else "pass_rejected", items, time, pass_epoch)
        elif kind == "result":
            items = data.get("learned_items", [])
            tasks[data["task_id"]] = (items, active_epoch)
            add("result", items, time, active_epoch)
            if data.get("status") == "not_found":
                add("not_found", [data["current_item"]], time, active_epoch)
        elif kind == "verification":
            items, task_epoch = tasks.get(data["task_id"], ([], -1))
            add("verified" if data["success"] else "revoke", items, time, task_epoch)
        elif kind == "missing":
            add("missing", [data["item"]], time, active_epoch)
        elif kind == "revoke":
            reference = data["evidence_id"]
            items, source_epoch = tasks.get(reference, passes.get(reference, ([], -1)))
            if reference in sends:
                items, source_epoch, _ = sends[reference]
            add("revoke", items, time, source_epoch)

    geometry = view["planned_pass"]
    values = {
        "epoch_changes": len(epoch_times), "epoch_age": now - epoch_times[-1] if epoch_times else -1.,
        "planned_distance": geometry["distance"], "planned_bearing_cos": geometry["bearing_cos"],
        "camera_range": geometry["camera_range"], "camera_fov": geometry["camera_fov"],
        "planned_in_range": int(geometry["distance"] <= geometry["camera_range"]),
        "planned_in_fov": int(geometry["bearing_cos"] >= cos(geometry["camera_fov"]/2)),
    }
    for item_kind in _KINDS:
        for name in _COUNTS:
            values[f"{item_kind}_{name}_count"] = len(counters[item_kind][name])
        for name in _AGES:
            times = counters[item_kind][name]
            values[f"{item_kind}_{name}_age"] = now - times[-1] if times else -1.
        values[f"{item_kind}_pending_count"] = sum(
            1 for msg_id, (items, sent_epoch, attempt) in sends.items()
            if sent_epoch == epoch and msg_id not in seen_acks and (msg_id, attempt) not in timed_out
            and any(item["kind"] == item_kind and item["object_id"] == target
                    and item["version"] == versions[item_kind] for item in items)
        )
    result = {name: float(values[name]) for name in FEATURE_NAMES}
    if not all(isfinite(value) for value in result.values()):
        raise ValueError("all feature values must be finite")
    return result
