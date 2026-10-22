"""Keep model probabilities, offline calibration, and policy caps distinct."""

from dataclasses import dataclass, field
from math import isfinite
from ..types import BeliefSnapshot, probability
from ..parameters import MissingMeasurement
import hashlib
import json


def source_key(sources):
    """Only A-observed evidence types; no phase, B cache, reset or geometry."""
    values = sorted(set(sources))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("evidence sources must be nonempty strings")
    return json.dumps(values, separators=(",", ":"))


def calibration_context(sources, timestamps, now):
    """Exact event-time freshness, without guessed stale-time cutoffs.

    This only distinguishes evidence processed NOW from earlier evidence. It
    does not claim that any positive age makes content invalid.
    """
    sources, timestamps = list(sources), list(timestamps)
    if len(sources) != len(timestamps) or type(now) not in (int, float) or not isfinite(now):
        raise ValueError("calibration context needs aligned sources/timestamps and finite clock")
    latest = {}
    for source, timestamp in zip(sources, timestamps):
        if not isinstance(source, str) or not source or type(timestamp) not in (int, float) or not isfinite(timestamp) or not 0 <= timestamp <= now:
            raise ValueError("invalid A evidence timestamp/source")
        latest[source] = max(latest.get(source, timestamp), timestamp)
    return json.dumps([[s, "now" if t == now else "earlier"] for s, t in sorted(latest.items())], separators=(",", ":"))


def validate_model(model):
    if not isinstance(model, dict) or not isinstance(model.get("maps"), dict):
        raise ValueError("calibration model requires per-item maps")
    payload = {key: value for key, value in model.items() if key != "model_id"}
    actual_id = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if model.get("model_id") != actual_id:
        raise ValueError("fixed calibration artifact hash mismatch")
    mode = model.get("conditioning", "pooled")
    if mode not in {"pooled", "evidence_sources_v1"}:
        raise ValueError("unknown calibration conditioning")
    mappings = [model["maps"].get(kind) for kind in "NCI"]
    if mode == "evidence_sources_v1":
        contexts = model.get("context_maps")
        if not isinstance(contexts, dict) or set(contexts) != set("NCI"):
            raise ValueError("source calibration requires N/C/I context maps")
        for kind in "NCI":
            if not isinstance(contexts[kind], dict) or not contexts[kind]:
                raise ValueError("empty calibration contexts")
            for key, mapping in contexts[kind].items():
                if source_key(json.loads(key)) != key:
                    raise ValueError("noncanonical calibration source key")
                mappings.append(mapping)
    for mapping in mappings:
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
    policy = model.get("application_policy")
    if policy is not None:
        if policy.get("name") != "validation_gated_raw_fallback_v1" or set(policy.get("gates", {})) != set("NCI"):
            raise ValueError("invalid selective calibration policy")
        for group in policy["gates"].values():
            for key, gate in group.items():
                if not isinstance(json.loads(key), list) or type(gate.get("use_calibration")) is not bool:
                    raise ValueError("invalid validation gate")
                if gate["use_calibration"] and (gate.get("train_rows", 0) <= 0 or gate.get("validation_rows", 0) <= 0):
                    raise ValueError("enabled gate needs training and validation evidence")


def _mapped_value(model, kind, score, sources):
    if model.get("conditioning", "pooled") == "evidence_sources_v1":
        key = source_key(sources)
        try:
            mapping = model["context_maps"][kind][key]
        except KeyError:
            raise MissingMeasurement(f"unseen calibration evidence sources for {kind}: {key}") from None
    else:
        mapping = model["maps"][kind]
    if not mapping["support"][0] <= score <= mapping["support"][1]:
        raise MissingMeasurement(f"{kind} score is outside measured calibration support")
    return next(b["value"] for b in mapping["blocks"] if score <= b["upper"])


def calibration_result(model, kind, score, *, hard_invalid, sources, timestamps=None, now=None):
    """Return prediction + explicit application reason; preserve unknown raw."""
    if hard_invalid:
        return 0., "hard_invalid_zero"
    if score is None:
        return None, "unmeasured_raw_parameters"
    probability(score, "raw score")
    policy = model.get("application_policy")
    if policy is not None:
        if timestamps is None or now is None:
            raise ValueError("selective calibration needs A evidence timestamps and decision time")
        sources, timestamps = list(sources), list(timestamps)
        key = calibration_context(sources, timestamps, now)
        if any(s == "ack" and t == now for s, t in zip(sources, timestamps)):
            return score, "raw_fresh_ACK_preserves_storage_times_content_validity"
        gate = policy["gates"][kind].get(key)
        if gate is None:
            return score, "raw_no_validated_context"
        if not gate["use_calibration"]:
            return score, "raw_"+gate["reason"]
        try:
            return _mapped_value(model, kind, score, sources), "calibration_validation_improved"
        except MissingMeasurement:
            return score, "raw_calibration_support_missing"
    return _mapped_value(model, kind, score, sources), "legacy_calibration"


def calibrated_value(model, kind, score, *, hard_invalid, sources, timestamps=None, now=None):
    return calibration_result(model, kind, score, hard_invalid=hard_invalid, sources=sources,
                              timestamps=timestamps, now=now)[0]


@dataclass(frozen=True)
class BeliefLevels:
    b_raw: dict[str, float | None]
    b_cal: dict[str, float | None]
    b_safe: dict[str, float | None]
    calibrated: bool
    model_id: str | None
    application_reasons: dict[str, str] = field(default_factory=dict)

    def as_policy_input(self, level="raw"):
        if level not in {"raw", "cal", "safe"}:
            raise ValueError("belief level must be raw, cal or safe")
        values = getattr(self, "b_" + level)
        if any(v is None for v in values.values()):
            raise MissingMeasurement(f"{level} contains unmeasured parameters/calibration; cannot pass guessed values to a policy")
        return dict(values)


def project(snapshot: BeliefSnapshot, model=None, caps=None) -> BeliefLevels:
    """A fixed map may calibrate raw scores; caps are a separate policy choice.

    No map means identity, explicitly marked uncalibrated. Expired C evidence
    remains invalid even when the fitted map would raise a zero score.
    """
    raw = snapshot.as_policy_input()
    calibrated = {key: None for key in raw}
    reasons = {}
    if model is not None:
        validate_model(model)
        if model["estimator_fingerprint"] != snapshot.estimator_fingerprint:
            raise ValueError("calibration belongs to a different raw estimator/parameter set")
        if snapshot.receiver_id not in model["receiver_ids"]:
            raise ValueError("calibration belongs to a different receiver")
        for kind in "NCI":
            belief = {"N": snapshot.normal, "C": snapshot.current, "I": snapshot.appearance}[kind]
            calibrated[f"b_{kind}"], reasons[f"b_{kind}"] = calibration_result(model, kind, raw[f"b_{kind}"],
                hard_invalid=belief.hard_invalid, sources=[e.source for e in belief.evidence],
                timestamps=[e.timestamp for e in belief.evidence], now=snapshot.timestamp)
    if snapshot.current.hard_invalid:
        calibrated["b_C"] = 0.
    caps = {} if caps is None else caps
    if set(caps) - set(raw):
        raise ValueError("caps require b_N, b_C or b_I keys")
    for value in caps.values():
        probability(value, "policy cap")
    safe = {key: min(value, caps.get(key, 1.)) if value is not None else None for key, value in calibrated.items()}
    return BeliefLevels(raw, calibrated, safe, model is not None,
                        model.get("model_id") if model else None, reasons)
