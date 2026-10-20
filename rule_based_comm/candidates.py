"""Generate the common eight-action E/N/C/I action space."""

from itertools import combinations

from .schemas import MESSAGE_ORDER


def generate_candidates() -> tuple[frozenset[str], ...]:
    optional = MESSAGE_ORDER[1:]
    result: list[frozenset[str]] = []
    for count in range(len(optional) + 1):
        for combo in combinations(optional, count):
            result.append(frozenset(("E", *combo)))
    return tuple(result)


def ordered_items(candidate: frozenset[str]) -> tuple[str, ...]:
    return tuple(item for item in MESSAGE_ORDER if item in candidate)

