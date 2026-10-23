"""Explainable expected-cost model for each candidate message."""

from __future__ import annotations

from .config import PolicyConfig
from .schemas import BeliefState, CandidateCost, Event, OutcomeEstimate, TaskRequirements
from .candidates import ordered_items


class ExpectedCostModel:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def message_size(self, candidate: frozenset[str]) -> int:
        return sum(self.config.message_sizes.size_of(item) for item in candidate)

    def transmission_cost(self, byte_count: float) -> float:
        return self.config.costs.transmission_per_kib * byte_count / 1024.0

    def evaluate(
        self,
        candidate: frozenset[str],
        event: Event,
        belief: BeliefState,
        requirements: TaskRequirements,
    ) -> CandidateCost:
        initial_bytes = self.message_size(candidate)
        outcome = self.predict_outcome(candidate, event, belief, requirements)
        weights = self.config.costs

        initial_tx = self.transmission_cost(initial_bytes)
        followup_tx = self.transmission_cost(outcome.expected_followup_bytes)
        request_cost = outcome.expected_requests * weights.request_roundtrip
        wrong_cost = outcome.p_wrong * weights.wrong_action
        reobserve_cost = outcome.expected_reobservations * weights.reobservation
        unresolved_cost = outcome.p_unresolved * weights.unresolved
        total = initial_tx + followup_tx + request_cost + wrong_cost + reobserve_cost + unresolved_cost

        return CandidateCost(
            items=ordered_items(candidate),
            initial_bytes=initial_bytes,
            transmission_cost=initial_tx,
            request_cost=request_cost,
            wrong_cost=wrong_cost,
            reobservation_cost=reobserve_cost,
            unresolved_cost=unresolved_cost,
            followup_transmission_cost=followup_tx,
            total_cost=total,
            outcome=outcome,
        )

    def predict_outcome(
        self,
        candidate: frozenset[str],
        event: Event,
        belief: BeliefState,
        requirements: TaskRequirements,
    ) -> OutcomeEstimate:
        requests = 0.0
        reobservations = 0.0
        p_wrong = 0.0
        p_unresolved = 0.0
        followup_bytes = 0.0

        # Missing required state is recovered by an explicit pull. The belief is
        # the chance that the pull is unnecessary, not an oracle truth value.
        for item, probability in (("N", belief.b_N), ("C", belief.b_C)):
            if item in requirements.required and item not in candidate:
                p_missing = 1.0 - probability
                requests += p_missing
                followup_bytes += p_missing * self.config.message_sizes.size_of(item)

        if "I" in requirements.required:
            appearance = self._identification_outcome(candidate, event, belief)
            requests += appearance.expected_requests
            reobservations += appearance.expected_reobservations
            p_wrong += appearance.p_wrong
            p_unresolved += appearance.p_unresolved
            followup_bytes += appearance.expected_followup_bytes

        return OutcomeEstimate(
            expected_requests=requests,
            expected_reobservations=reobservations,
            p_wrong=min(1.0, p_wrong),
            p_unresolved=min(1.0, p_unresolved),
            expected_followup_bytes=followup_bytes,
        )

    def _identification_outcome(
        self,
        candidate: frozenset[str],
        event: Event,
        belief: BeliefState,
    ) -> OutcomeEstimate:
        params = self.config.outcome
        if "I" in candidate:
            return self._crop_failure_outcome(1.0 - params.crop_match_success)

        known_path = self._match_failure_outcome(1.0 - params.appearance_match_success)
        missing_path = self._missing_appearance_outcome(event)
        return self._mix(known_path, missing_path, belief.b_I)

    def _missing_appearance_outcome(self, event: Event) -> OutcomeEstimate:
        """Model request-vs-position-only behavior when B has no appearance.

        sigma / distance is interpreted as the probability that B notices the
        scene is ambiguous and asks for a crop. Otherwise B attempts a spatial
        match. This is deliberately simple and must be calibrated with trials.
        """
        params = self.config.outcome
        sigma = event.position_uncertainty_cm
        distance = event.object_distance_cm
        if sigma is None or distance is None:
            request_probability = 1.0
            position_success = params.default_position_success
        else:
            request_probability = min(1.0, sigma / distance)
            position_success = distance / (distance + 2.0 * sigma)

        crop_failure = self._crop_failure_outcome(1.0 - params.crop_match_success)
        position_failure = self._match_failure_outcome(1.0 - position_success)
        no_request_probability = 1.0 - request_probability

        return OutcomeEstimate(
            expected_requests=request_probability,
            expected_reobservations=(
                request_probability * crop_failure.expected_reobservations
                + no_request_probability * position_failure.expected_reobservations
            ),
            p_wrong=(
                request_probability * crop_failure.p_wrong
                + no_request_probability * position_failure.p_wrong
            ),
            p_unresolved=(
                request_probability * crop_failure.p_unresolved
                + no_request_probability * position_failure.p_unresolved
            ),
            expected_followup_bytes=request_probability * self.config.message_sizes.I,
        )

    def _match_failure_outcome(self, failure_probability: float) -> OutcomeEstimate:
        params = self.config.outcome
        p_wrong = failure_probability * params.wrong_fraction_on_match_failure
        safe_failure = failure_probability - p_wrong
        return OutcomeEstimate(
            expected_reobservations=safe_failure,
            p_wrong=p_wrong,
            p_unresolved=safe_failure * params.unresolved_after_reobservation,
        )

    def _crop_failure_outcome(self, failure_probability: float) -> OutcomeEstimate:
        """Residual crop risk using scenario ④'s 2.3%/2.7% split."""
        params = self.config.outcome
        p_wrong = failure_probability * params.crop_wrong_fraction_on_failure
        return OutcomeEstimate(
            expected_reobservations=failure_probability,
            p_wrong=p_wrong,
            p_unresolved=failure_probability - p_wrong,
        )

    @staticmethod
    def _mix(
        known: OutcomeEstimate,
        missing: OutcomeEstimate,
        p_known: float,
    ) -> OutcomeEstimate:
        p_missing = 1.0 - p_known
        return OutcomeEstimate(
            expected_requests=p_known * known.expected_requests + p_missing * missing.expected_requests,
            expected_reobservations=(
                p_known * known.expected_reobservations
                + p_missing * missing.expected_reobservations
            ),
            p_wrong=p_known * known.p_wrong + p_missing * missing.p_wrong,
            p_unresolved=p_known * known.p_unresolved + p_missing * missing.p_unresolved,
            expected_followup_bytes=(
                p_known * known.expected_followup_bytes
                + p_missing * missing.expected_followup_bytes
            ),
        )

