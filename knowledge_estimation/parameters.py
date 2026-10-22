"""Measured parameter lookup. There are no numerical probability fallbacks."""

import hashlib
import json
from pathlib import Path
from copy import deepcopy

from .types import identifier, probability


class MissingMeasurement(ValueError):
    pass


CONTEXT_FIELDS = ("receiver_id", "kind", "site_id", "route_id", "condition_id")
PROBABILITIES = ("p_acq", "q_learn", "r_crop", "r_current", "p_recv", "p_ack")
DISTRIBUTIONS = ("F_pass", "F_task", "F_recv")


def context_key(context):
    if set(context) != set(CONTEXT_FIELDS):
        raise ValueError(f"parameter context requires exactly {CONTEXT_FIELDS}")
    for key, value in context.items():
        if not isinstance(value, str) or (key in {"receiver_id", "condition_id"} and not value.strip()):
            raise ValueError(f"invalid parameter context: {key}")
    return tuple(context[key] for key in CONTEXT_FIELDS)


class ParameterStore:
    def __init__(self, artifact=None, *, allow_fixture=False, allow_simulation=False):
        self._artifact = deepcopy(artifact) if artifact is not None else {"schema_version": 1, "origin": "unmeasured", "parameters": []}
        if self.artifact.get("schema_version") != 1:
            raise ValueError("unsupported parameter schema")
        self._origin = self.artifact.get("origin")
        if self.origin not in {"unmeasured", "measured", "fixture", "simulation"}:
            raise ValueError("parameter origin must be measured, fixture or unmeasured")
        if self.origin == "fixture" and not allow_fixture:
            raise MissingMeasurement("test fixture parameters cannot be used as measured parameters")
        if self.origin == "simulation" and not allow_simulation:
            raise MissingMeasurement("simulation parameters require explicit allow_simulation=True")
        if self.origin == "unmeasured" and self.artifact["parameters"]:
            raise ValueError("unmeasured parameter store cannot contain numerical entries")
        self._table = {}
        for entry in self.artifact["parameters"]:
            name = entry["name"]
            if name not in PROBABILITIES + DISTRIBUTIONS:
                raise ValueError(f"unsupported parameter: {name}")
            key = (name, context_key(entry["context"]))
            if key in self._table:
                raise ValueError("duplicate parameter context")
            identifier(entry["parameter_id"], "parameter_id")
            if not entry.get("evidence_ids") or not entry.get("input_sha256"):
                raise ValueError("parameters require measurement evidence and input hashes")
            digest = entry["input_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("parameter input_sha256 must identify the measured CSV bytes")
            n = entry["trials"]
            if type(n) is not int or n <= 0:
                raise ValueError("parameter requires positive trial count")
            if name in PROBABILITIES:
                probability(entry["value"], name)
                s = entry["successes"]
                if type(s) is not int or not 0 <= s <= n or entry["value"] != s/n:
                    raise ValueError("source rates must equal measured successes/trials; no pseudo-counts")
            else:
                samples = entry["samples_seconds"]
                if (len(samples) > n or (name != "F_task" and len(samples) != n)
                        or any(type(v) not in (int, float) or not 0 <= v < float("inf") for v in samples)):
                    raise ValueError("CDF needs one non-negative measured duration per trial")
                horizon = entry.get("observation_horizon_seconds")
                if type(horizon) not in (int, float) or not 0 < horizon < float("inf") or any(t > horizon for t in samples):
                    raise ValueError("CDF requires its observed finite horizon")
            self._table[key] = entry

    @classmethod
    def load(cls, path, *, allow_fixture=False, allow_simulation=False):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), allow_fixture=allow_fixture,
                   allow_simulation=allow_simulation)

    def entry(self, name, context):
        try:
            return deepcopy(self._table[(name, context_key(context))])
        except KeyError:
            raise MissingMeasurement(f"unmeasured {name} for {context}; no global fallback") from None

    def value(self, name, context):
        return self.entry(name, context)["value"]

    def cdf(self, name, context, elapsed):
        if type(elapsed) not in (int, float) or not 0 <= elapsed < float("inf"):
            raise ValueError("elapsed must be finite and non-negative")
        entry = self.entry(name, context)
        if elapsed > entry["observation_horizon_seconds"]:
            raise MissingMeasurement("elapsed exceeds measured CDF horizon; no tail extrapolation")
        samples = entry["samples_seconds"]
        return sum(t <= elapsed for t in samples)/entry["trials"]

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.artifact, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def artifact(self):
        return deepcopy(self._artifact)

    @property
    def origin(self):
        return self._origin
