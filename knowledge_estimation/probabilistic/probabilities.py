"""Probability identities; see ../docs/RESEARCH_BASIS.md for assumptions.

These are not the second-order belief filters of Buehler et al.
"""

from math import expm1, fsum, log1p

from ..types import probability


def receipt_given_no_ack(p_delivery: float, p_ack_given_delivery: float) -> float:
    """P(D | no A) = d(1-a)/(1-da), after a complete ACK deadline.

    D means validated and stored delivery. ACKs are truthful and generated only
    after D. a includes ACK generation and timely return, conditional on D.
    No separate receiver storage-failure model is included.
    """
    probability(p_delivery, "p_delivery")
    probability(p_ack_given_delivery, "p_ack_given_delivery")
    if p_delivery * p_ack_given_delivery == 1:
        raise ValueError("no ACK is impossible when delivery and ACK are certain")
    return p_delivery * (1-p_ack_given_delivery) / (1-p_delivery*p_ack_given_delivery)


def independent_union(probabilities) -> float:
    """P(any acquisition succeeds) = 1-product(1-p), ONLY if independent.

    These are acquisition events, not generic posterior opinions. The estimator
    first collapses dependent sources to a maximum, a union lower bound rather
    than an exact posterior. Remaining groups must be independent for this step.
    """
    values = tuple(probabilities)
    for value in values:
        probability(value, "acquisition probability")
    if 1. in values:
        return 1.
    return -expm1(fsum(log1p(-value) for value in values)) if values else 0.
