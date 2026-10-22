"""Run the native simulator, or deterministic headless episodes without Tk."""

import argparse
import json
import math
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KARCo native 2D knowledge-estimation simulator")
    parser.add_argument("--scenario", choices=("first_event", "pass_clear", "pass_occluded", "ack_loss"), default="pass_clear")
    parser.add_argument("--seed", type=int, default=7)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="run without importing Tk or opening a window")
    mode.add_argument("--smoke-test", action="store_true", help="open the native window briefly and exit")
    parser.add_argument("--seconds", type=float, default=30, help="headless episode duration (default: 30)")
    parser.add_argument("--steps", type=int, help="headless fixed 1/60-second steps; overrides --seconds")
    parser.add_argument("--output", type=Path, help="write headless snapshot and decision_records as JSON")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and non-negative")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if not args.headless and (args.output is not None or args.steps is not None):
        parser.error("--output and --steps require --headless")
    from .engine import Simulator

    simulator = Simulator(scenario=args.scenario, seed=args.seed)
    if args.headless:
        if args.steps is not None:
            for _ in range(args.steps):
                simulator.step(1 / 60)
        else:
            count = int(args.seconds * 60)
            for _ in range(count):
                simulator.step(1 / 60)
            remainder = args.seconds - count / 60
            if remainder > 1e-12:
                simulator.step(remainder)
        snapshot, records = simulator.snapshot(), simulator.export_records()
        summary = {"scenario": args.scenario, "seed": args.seed, "time": snapshot["time"],
                   "event": snapshot.get("event"), "stats": snapshot.get("stats"),
                   "decision_records": len(records)}
        if args.output is not None:
            payload = {"summary": summary, "snapshot": snapshot, "decision_records": records}
            try:
                args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError, TypeError) as error:
                parser.exit(2, f"Could not save output: {error}\n")
            summary["output"] = str(args.output)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    try:
        from .desktop import run_desktop
    except ImportError as error:
        parser.exit(2, f"Native GUI requires Python with Tkinter. Headless mode remains available: {error}\n")
    run_desktop(simulator, seed=args.seed, smoke_test=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
