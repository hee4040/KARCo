"""Choose raw vs existing calibration using separate validation evidence only."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json

from ..measurement import load_dataset, decision_pairs
from ..parameters import MissingMeasurement
from .projection import calibration_context, calibrated_value, validate_model


def fit_application_policy(base_model, training, validation, *, allow_fixture=False, allow_simulation=False):
    validate_model(base_model)
    if base_model.get("application_policy") is not None:
        raise ValueError("gate an unmodified base calibration model")
    flags = dict(allow_fixture=allow_fixture, allow_simulation=allow_simulation)
    train_manifest, _, train_rows = load_dataset(training, **flags)
    val_manifest, _, val_rows = load_dataset(validation, **flags)
    if any(m["origin"] != base_model["origin"] or m["protocol_id"] != base_model["protocol_id"] for m in (train_manifest, val_manifest)):
        raise ValueError("mixed origins/protocols")
    if not train_rows or any(r["split"] != "calibration" for r in train_rows):
        raise ValueError("training support must contain calibration rows only")
    if not val_rows or any(r["split"] != "validation" for r in val_rows):
        raise ValueError("application gates use a separate validation dataset only")
    fields = ("episode_id", "session_id", "object_instance", "route_group_id")
    for field in fields:
        if {r[field] for r in train_rows} & {r[field] for r in val_rows}:
            raise ValueError("training/validation groups overlap")
        prior = set(base_model["groups"].get(field, [])) | set(base_model["source_fit_groups"].get(field, []))
        if prior & {r[field] for r in val_rows}:
            raise ValueError("validation overlaps base fitting data")
    support, groups = defaultdict(list), defaultdict(list)
    for row in train_rows+val_rows:
        if row["estimator_fingerprint"] != base_model["estimator_fingerprint"] or row["receiver_id"] not in base_model["receiver_ids"]:
            raise ValueError("gate data uses another raw estimator/receiver")
    for row in train_rows:
        for kind in "NCI":
            if row["hard_invalid"][kind] or row["b_raw"]["b_"+kind] is None:
                continue
            evidence = row["evidence"][kind]
            key = calibration_context([e["source"] for e in evidence], [e["timestamp"] for e in evidence], row["time"])
            support[(kind, key)].append(row["episode_id"])
    for row, _, _, truth in decision_pairs(validation, split="validation"):
        for kind in "NCI":
            raw = row["b_raw"]["b_"+kind]
            if raw is None or row["hard_invalid"][kind]:
                continue
            evidence = row["evidence"][kind]
            sources, times = [e["source"] for e in evidence], [e["timestamp"] for e in evidence]
            key = calibration_context(sources, times, row["time"])
            try:
                candidate = calibrated_value(base_model, kind, raw, hard_invalid=False, sources=sources)
            except MissingMeasurement:
                candidate = None
            groups[(kind, key)].append({"raw": raw, "candidate": candidate, "K": truth[kind]["K"],
                "geometry": row["episode_id"].rsplit("/", 1)[-1], "episode": row["episode_id"]})
    gates = {k: {} for k in "NCI"}
    for (kind, key), rows in groups.items():
        train = support[(kind, key)]
        paired = [r for r in rows if r["candidate"] is not None]
        # Different intervention cases share one geometry. Weight each geometry
        # equally rather than claiming every repeated checkpoint is independent.
        geometry = defaultdict(list)
        for row in paired:
            geometry[row["geometry"]].append(row)
        raw_brier = sum(sum((r["raw"]-r["K"])**2 for r in rs)/len(rs) for rs in geometry.values())/len(geometry) if geometry else None
        cal_brier = sum(sum((r["candidate"]-r["K"])**2 for r in rs)/len(rs) for rs in geometry.values())/len(geometry) if geometry else None
        reason = "validation_not_improved_or_tie"
        if not train:
            reason = "training_context_absent"
        elif len(paired) != len(rows):
            reason = "incomplete_calibration_support"
        elif len(geometry) < 2:
            reason = "single_geometry_cannot_validate_across_conditions"
        elif cal_brier < raw_brier:
            reason = "validation_improved"
        gates[kind][key] = {"use_calibration": reason == "validation_improved", "reason": reason,
            "train_rows": len(train), "train_episodes": len(set(train)), "validation_rows": len(rows),
            "paired_rows": len(paired), "validation_geometry_groups": len(geometry),
            "raw_brier_geometry_weighted": raw_brier, "cal_brier_geometry_weighted": cal_brier}
    model = deepcopy(base_model)
    model["application_policy"] = {"name": "validation_gated_raw_fallback_v1", "gates": gates,
        "base_model_id": base_model["model_id"],
        "context": "per A evidence-source type: processed at decision time or earlier; no stale-time cutoff",
        "selection": "strictly lower validation Brier with complete calibration support and >1 geometry; tie/missing support keeps raw",
        "sample_scope": "minimum >1 geometry is a diversity guard, not a statistical adequacy claim",
        "validation_input_sha256": val_manifest["sha256"], "training_support_sha256": train_manifest["sha256"]}
    model["application_validation_groups"] = {field: sorted({r[field] for r in val_rows}) for field in fields}
    model.pop("model_id")
    model["model_id"] = hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()
    validate_model(model)
    return model
