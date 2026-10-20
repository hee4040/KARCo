"""Small CLI demo that prints the selected message and all candidate costs."""

import argparse
import json

from rule_based_comm import RuleBasedCommunicationPolicy


SCENARIOS = {
    "first": (
        {"event_id": "e1", "target_confirmed": True, "has_similar_objects": False},
        {"b_N": 0.0, "b_I": 0.0},
    ),
    "reuse": (
        {"event_id": "e2", "target_confirmed": True, "has_similar_objects": False},
        {"b_N": 1.0, "b_I": 0.9},
    ),
    "observed": (
        {
            "event_id": "e3",
            "target_confirmed": True,
            "has_similar_objects": True,
            "object_distance_cm": 4.0,
            "position_uncertainty_cm": 3.0,
        },
        {"b_N": 0.72, "b_I": 0.72},
    ),
    "ambiguous": (
        {
            "event_id": "e4",
            "target_confirmed": True,
            "has_similar_objects": True,
            "object_distance_cm": 4.0,
            "position_uncertainty_cm": 3.0,
        },
        {"b_N": 0.0, "b_I": 0.0},
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=SCENARIOS, nargs="?", default="observed")
    args = parser.parse_args()
    event, belief = SCENARIOS[args.scenario]
    decision = RuleBasedCommunicationPolicy().select(event, belief)
    print(json.dumps(decision.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
