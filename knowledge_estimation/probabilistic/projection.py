"""b_raw -> b_cal: a fixed isotonic (PAV) score map per item, fitted on a separate calibration split.

The map is bound to one raw estimator (its fingerprint) and is never extrapolated: a score outside the scores
seen while fitting is MissingMeasurement, so b_cal stays None. Policy caps (b_safe) are applied by the caller.
"""

from collections import defaultdict
import hashlib
import json

from ..types import probability
from ..parameters import MissingMeasurement


def model_id(model):
    """Content hash of a calibration artifact (without its own model_id)."""
    payload = {key: value for key, value in model.items() if key != "model_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def monotone_map(pairs):
    """Pool-adjacent-violators on (raw score, 0/1 label) pairs; no prior, no pseudo-counts."""
    grouped = defaultdict(lambda: [0, 0])
    for score, label in pairs:
        grouped[score][0] += 1
        grouped[score][1] += label
    blocks = []
    for score, (n, s) in sorted(grouped.items()):
        blocks.append({"upper": score, "count": n, "positive": s})
        while len(blocks) > 1 and blocks[-2]["positive"]/blocks[-2]["count"] > blocks[-1]["positive"]/blocks[-1]["count"]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append({"upper": right["upper"], "count": left["count"]+right["count"],
                           "positive": left["positive"]+right["positive"]})
    if not blocks:
        raise MissingMeasurement("no known raw calibration scores for one item")
    for block in blocks:
        block["value"] = block["positive"]/block["count"]
    return {"support": [min(grouped), max(grouped)], "blocks": blocks}


def validate_model(model):
    if not isinstance(model, dict) or not isinstance(model.get("maps"), dict):
        raise ValueError("calibration model requires per-item maps")
    if model.get("model_id") != model_id(model):
        raise ValueError("fixed calibration artifact hash mismatch")
    for kind in "NCI":
        mapping = model["maps"].get(kind)
        if not isinstance(mapping, dict) or set(mapping) != {"support", "blocks"}:
            raise ValueError("calibration map requires its measured score support")
        blocks = mapping["blocks"]
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("calibration map is empty")
        previous_upper, previous_value = -1., -1.
        for block in blocks:
            if not isinstance(block, dict) or not {"upper", "value"} <= set(block):
                raise ValueError("invalid calibration block")
            probability(block["upper"], "calibration upper")
            probability(block["value"], "calibration value")
            if block["upper"] <= previous_upper or block["value"] < previous_value:
                raise ValueError("calibration blocks must have increasing bounds and monotone values")
            previous_upper, previous_value = block["upper"], block["value"]
        support = mapping["support"]
        if len(support) != 2 or not 0 <= support[0] <= support[1] <= 1 or previous_upper != support[1]:
            raise ValueError("calibration support mismatch")
    if model.get("origin") not in {"measured", "fixture", "simulation"} or not model.get("estimator_fingerprint"):
        raise ValueError("calibration requires data origin and exact raw estimator fingerprint")


def calibration_result(model, kind, score, *, hard_invalid):
    """(b_cal, reason). An explicitly invalidated item is 0; an unknown raw score stays None."""
    if hard_invalid:
        return 0., "hard_invalid_zero"
    if score is None:
        return None, "unmeasured_raw_parameters"
    probability(score, "raw score")
    mapping = model["maps"][kind]
    if not mapping["support"][0] <= score <= mapping["support"][1]:
        raise MissingMeasurement(f"{kind} score is outside measured calibration support")
    return next(b["value"] for b in mapping["blocks"] if score <= b["upper"]), "calibrated"
