"""Deterministic fake-CSV plumbing test; produces no performance claim."""

import argparse
import json
from pathlib import Path

from knowledge_estimation.calibrate import fit_sources, fit_scores
from knowledge_estimation.collect import replay
from knowledge_estimation.evaluate import evaluate
from knowledge_estimation.measurement import write_json
from knowledge_estimation.parameters import ParameterStore
from knowledge_estimation.tests.fixtures import source_dataset, decision_dataset


# These are declared test-analysis choices, not robot parameter estimates.
FIXTURE_EVALUATION_PROTOCOL = {"high_confidence_threshold": .9, "reliability_bin_edges": [0., .5, 1.],
    "confidence_level": .95, "bootstrap_repetitions": 20, "bootstrap_seed": 1, "bootstrap_unit": "episode_id"}


def run(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    source_dataset(root / "source_csv_fixture")
    params = fit_sources(root / "source_csv_fixture", allow_fixture=True)
    write_json(root / "fixture_params.yaml", params)
    store = ParameterStore(params, allow_fixture=True)
    decision_dataset(root / "unmeasured_log_fixture")
    replay(root / "unmeasured_log_fixture", root / "replayed_fixture", store, allow_fixture=True)
    model = fit_scores(root / "replayed_fixture", allow_fixture=True)
    write_json(root / "fixture_calibration.json", model)
    write_json(root / "fixture_evaluation_protocol.json", FIXTURE_EVALUATION_PROTOCOL)
    result = evaluate(root / "replayed_fixture", model, FIXTURE_EVALUATION_PROTOCOL, allow_fixture=True)
    write_json(root / "fixture_contract_report.json", result)
    return {"passed": True, "origin": "fixture", "purpose": "CSV -> params -> A-only replay -> fixed score calibration -> evaluation contract only",
            "not_robot_performance": True, "output": str(root)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output)))


if __name__ == "__main__":
    main()
