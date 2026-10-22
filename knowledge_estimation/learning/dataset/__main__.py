"""python -m knowledge_estimation.learning.dataset --output knowledge_estimation/data/run_001"""

import argparse
import json
from pathlib import Path

from .collection import CollectionConfig, collect
from .audit import audit_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect A-only histories and separate B validity labels")
    parser.add_argument("--output", type=Path, required=True, help="a new directory; existing paths are never overwritten")
    parser.add_argument("--families", type=int, default=30, help="four scenarios per family, minimum 5 (default: 30)")
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--sample-interval", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args(argv)
    try:
        options = CollectionConfig(args.families, args.seconds, args.sample_interval, args.seed)
        result = collect(args.output, options,
                         progress=lambda done, total: print(f"Collected {done}/{total} families", flush=True))
        report = audit_dataset(args.output)
        (args.output / "audit_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Collection failed: {error}\n")
    print(json.dumps({"output": str(args.output), "audit_passed": report["passed"],
                      "statistics": result["statistics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
