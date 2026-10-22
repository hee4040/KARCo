"""Thin adapter for heewoo's rule_based_comm; never implements its cost model."""

import importlib
from pathlib import Path
import sys
from dataclasses import asdict, is_dataclass
from copy import deepcopy

from ..probabilistic.projection import project, validate_model
from ..parameters import MissingMeasurement


_ACTION_IDS = {frozenset(items): index for index, items in enumerate(
    ("E", "EN", "EC", "EI", "ENC", "ENI", "ECI", "ENCI"))}


class RuleBasedSelector:
    def __init__(self, *, source=None, adapter=None, policy_config=None, policy_manifest=None,
                 model=None, level="cal", caps=None, allow_fixture=False):
        if level not in {"raw", "cal", "safe"}:
            raise ValueError("level must be raw, cal or safe")
        if level == "cal" and model is None:
            raise ValueError("cal requires a fitted calibration model")
        if model is not None:
            validate_model(model)
            if model["origin"] == "fixture" and not allow_fixture:
                raise ValueError("test fixture calibration cannot drive a real policy")
        if not isinstance(policy_manifest, dict) or policy_manifest.get("origin") not in {"measured", "fixture"}:
            raise MissingMeasurement("rule policy requires an explicit parameter/provenance manifest; no teammate default values")
        if policy_manifest["origin"] == "fixture" and not allow_fixture:
            raise MissingMeasurement("fixture policy is restricted to contract tests")
        if not policy_manifest.get("evidence_ids") or not policy_manifest.get("input_sha256"):
            raise MissingMeasurement("policy parameter evidence and raw log hashes are required")
        for digest in policy_manifest["input_sha256"].values():
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("policy evidence hashes must be SHA256 digests")
        if model is not None and model["origin"] != policy_manifest["origin"]:
            raise ValueError("policy and calibration data origins must match")
        if adapter is None:
            if policy_config is None:
                raise MissingMeasurement("supply the teammate's explicitly configured policy_config")
            if source is not None:
                root = Path(source).resolve()
                if not (root / "rule_based_comm" / "policy_adapter.py").is_file():
                    raise ValueError("source must contain rule_based_comm/policy_adapter.py")
                loaded = sys.modules.get("rule_based_comm")
                if loaded is not None and Path(loaded.__file__).resolve().parent != root / "rule_based_comm":
                    raise ValueError("a different rule_based_comm is already loaded; start a new process")
                sys.path.insert(0, str(root))
            try:
                module = importlib.import_module("rule_based_comm.policy_adapter")
            except ModuleNotFoundError as error:
                if error.name and error.name.startswith("rule_based_comm"):
                    raise RuntimeError("rule_based_comm is unavailable: merge the rule branch or use --rule-source") from error
                raise
            from importlib import import_module
            policy_type = import_module("rule_based_comm.rule_policy").RuleBasedCommunicationPolicy
            adapter = module.RulePolicyAdapter(policy=policy_type(config=policy_config))
        actual_config = getattr(getattr(adapter, "policy", None), "config", None)
        if actual_config is None or not is_dataclass(actual_config) or asdict(actual_config) != policy_manifest.get("config"):
            raise ValueError("actual rule policy config differs from its recorded parameter evidence")
        self.adapter, self.model, self.level, self.caps = adapter, deepcopy(model), level, deepcopy(caps)
        self.policy_manifest = deepcopy(policy_manifest)

    def select(self, event, snapshot):
        """Use named N/I/C keys; scene identification stays in the rule model."""
        required = {"event_id", "spot_id", "object_id", "target_confirmed",
                    "has_similar_objects", "similar_object_count", "already_processing",
                    "scene_changed", "object_distance_cm", "position_uncertainty_cm",
                    "anomaly_type"}
        if set(event) - required:
            raise ValueError("event contains unsupported or privileged fields")
        if "object_id" in event and event["object_id"] != snapshot.normal.item.key.object_id:
            raise ValueError("event and belief refer to different objects")
        if "spot_id" in event and event["spot_id"] != snapshot.normal.item.key.site_id:
            raise ValueError("event and N belief refer to different sites")
        levels = project(snapshot, self.model, self.caps)
        observation = {**event, **levels.as_policy_input(self.level)}
        action_id, decision = self.adapter.predict(observation)
        result = decision.as_dict()
        if result["action"] not in {"SEND", "NO_COMM", "REOBSERVE"}:
            raise ValueError("unknown rule action")
        selected = result["selected_items"]
        if len(set(selected)) != len(selected) or set(selected) - set("ENCI"):
            raise ValueError("invalid selected_items")
        if result["action"] == "SEND" and "E" not in selected:
            raise ValueError("a task SEND must include E")
        if result["action"] == "SEND" and (type(action_id) is not int or _ACTION_IDS[frozenset(selected)] != action_id):
            raise ValueError("action_id does not match the selected payload")
        if result["action"] != "SEND" and (selected or action_id is not None):
            raise ValueError("non-SEND must not contain payload or action id")
        return {"action_id": action_id, **result, "observation": observation,
                "belief_level": self.level, "b_raw": levels.b_raw,
                "b_cal": levels.b_cal, "b_safe": levels.b_safe,
                "calibrated": levels.calibrated, "model_id": levels.model_id}
