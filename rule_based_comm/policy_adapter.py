"""Shared discrete action adapter for rule-based and PPO policies."""

from collections.abc import Mapping
from typing import Any

from .rule_policy import RuleBasedCommunicationPolicy
from .schemas import Action, MESSAGE_ORDER, PolicyDecision


ACTIONS: dict[int, tuple[str, ...]] = {
    0: ("E",),
    1: ("E", "N"),
    2: ("E", "C"),
    3: ("E", "I"),
    4: ("E", "N", "C"),
    5: ("E", "N", "I"),
    6: ("E", "C", "I"),
    7: ("E", "N", "C", "I"),
}
ACTION_IDS = {items: action_id for action_id, items in ACTIONS.items()}


class RulePolicyAdapter:
    """Returns a PPO-compatible action id while preserving diagnostics."""

    def __init__(self, policy: RuleBasedCommunicationPolicy | None = None) -> None:
        self.policy = policy or RuleBasedCommunicationPolicy()

    def predict(self, observation: Mapping[str, Any]) -> tuple[int | None, PolicyDecision]:
        decision = self.policy.select(observation, observation)
        if decision.action is not Action.SEND:
            return None, decision
        ordered = tuple(item for item in MESSAGE_ORDER if item in decision.selected_items)
        return ACTION_IDS[ordered], decision

