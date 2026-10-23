"""Minimum expected-cost communication policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .candidates import generate_candidates
from .config import PolicyConfig
from .cost_model import ExpectedCostModel
from .requirements import determine_requirements
from .schemas import Action, BeliefState, Event, PolicyDecision


class RuleBasedCommunicationPolicy:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self.cost_model = ExpectedCostModel(self.config)

    def select(
        self,
        event: Event | Mapping[str, Any],
        belief: BeliefState | Mapping[str, Any],
    ) -> PolicyDecision:
        event_obj = event if isinstance(event, Event) else Event.from_mapping(event)
        belief_obj = belief if isinstance(belief, BeliefState) else BeliefState.from_mapping(belief)
        requirements = determine_requirements(event_obj)

        if requirements.status is Action.NO_COMM:
            return PolicyDecision(action=Action.NO_COMM, reason="; ".join(requirements.reasons))
        if requirements.status is Action.REOBSERVE:
            return PolicyDecision(action=Action.REOBSERVE, reason="; ".join(requirements.reasons))

        costs = tuple(
            self.cost_model.evaluate(candidate, event_obj, belief_obj, requirements)
            for candidate in generate_candidates()
        )
        # Stable tie-break: lower cost, then fewer bytes, then action-space order.
        best = min(enumerate(costs), key=lambda pair: (pair[1].total_cost, pair[1].initial_bytes, pair[0]))[1]
        ranked = tuple(sorted(costs, key=lambda cost: (cost.total_cost, cost.initial_bytes)))
        return PolicyDecision(
            action=Action.SEND,
            selected_items=best.items,
            estimated_bytes=best.initial_bytes,
            expected_cost=best.total_cost,
            reason="minimum expected cost among 8 E/N/C/I candidates",
            candidate_costs=ranked,
        )

