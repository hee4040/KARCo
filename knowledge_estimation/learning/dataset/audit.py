"""Audit learning inputs, grouped splits and privileged label evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from .features import FEATURE_NAMES, extract_features


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def audit_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _check(manifest.get("schema_version") == 1 and manifest.get("complete"), "incomplete or unsupported dataset")
    _check(manifest["feature_names"] == list(FEATURE_NAMES), "unexpected feature schema")
    for relative, digest in manifest["sha256"].items():
        path = (root / relative).resolve()
        _check(path.is_relative_to(root.resolve()), "checksum path escapes dataset")
        _check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"checksum mismatch: {relative}")
    episodes = {entry["episode_id"]: entry for entry in manifest["episodes"]}
    _check(len(episodes) == len(manifest["episodes"]), "duplicate episode IDs")
    family_split = {}
    for entry in episodes.values():
        previous = family_split.setdefault(entry["family_id"], entry["split"])
        _check(previous == entry["split"], "a layout/seed family leaks across splits")
    seen_ids, used_episodes, report = set(), set(), {}
    for split in ("train", "validation", "test"):
        label_rows = {row["sample_id"]: row for row in read_jsonl(root / "labels" / f"{split}.jsonl")}
        _check(len(label_rows) == manifest["statistics"][split]["samples"], "missing or duplicate label rows")
        count, positive = 0, {k: 0 for k in "NCI"}
        brier, reliability = {k: 0. for k in "NCI"}, {k: [[0, 0., 0] for _ in range(10)] for k in "NCI"}
        by_reason = {}
        history_cache = {}
        for row in read_jsonl(root / "inputs" / f"{split}.jsonl"):
            identifier, episode = row["sample_id"], row["episode_id"]
            _check(identifier not in seen_ids, "sample ID repeats or leaks across splits")
            seen_ids.add(identifier)
            used_episodes.add(episode)
            _check(episode in episodes and episodes[episode]["split"] == split, "episode belongs to the wrong split")
            label = label_rows.pop(identifier, None)
            _check(label is not None and label["episode_id"] == episode, "input/label ID mismatch")
            _check(row["time"] == row["a_view"]["time"] and row["target"] == row["a_view"]["target"], "sample metadata mismatch")
            if episode not in history_cache:
                history_cache[episode] = list(read_jsonl(root / "observations" / f"{episode}.jsonl"))
            prefix = history_cache[episode][:row["a_view"]["history_end"]]
            expected = extract_features(prefix, row["a_view"])
            _check(list(row["features"]) == list(FEATURE_NAMES) and row["features"] == expected,
                   "features do not match the A-only causal prefix")
            count += 1
            reason = by_reason.setdefault(row["reason"], {"samples": 0, "brier_sum": {k: 0. for k in "NCI"}})
            reason["samples"] += 1
            for kind in "NCI":
                y = label["labels"][kind]
                evidence = label["audit"]["items"][kind]
                valid = int(evidence["stored"] and evidence["usable"]
                            and evidence["stored_version"] == evidence["actual_version"])
                _check(type(y) is int and y in (0, 1) and y == valid, "label disagrees with privileged cache/version audit")
                probability = row["a_view"]["baseline"][f"b_{kind}"]
                _check(0 <= probability <= 1, "invalid baseline probability")
                positive[kind] += y
                squared_error = (probability-y)**2
                brier[kind] += squared_error
                reason["brier_sum"][kind] += squared_error
                bucket = reliability[kind][min(9, int(probability*10))]
                bucket[0] += 1
                bucket[1] += probability
                bucket[2] += y
        _check(not label_rows and count == manifest["statistics"][split]["samples"], "unpaired samples")
        _check(positive == manifest["statistics"][split]["positive"], "class counts disagree with manifest")
        for reason in by_reason.values():
            reason["baseline_brier"] = {k: reason["brier_sum"][k]/reason["samples"] for k in "NCI"}
            del reason["brier_sum"]
        report[split] = {
            "samples": count, "positive": positive,
            "baseline_brier": {k: brier[k]/count for k in "NCI"}, "by_reason": by_reason,
            "reliability": {k: [{"bin": i, "samples": n, "mean_prediction": p/n if n else None,
                                 "positive_rate": y/n if n else None}
                                for i, (n, p, y) in enumerate(reliability[k])] for k in "NCI"},
            "single_class_warning": [k for k in "NCI" if positive[k] in (0, count)],
        }
    _check(used_episodes == set(episodes), "manifest includes unused episodes")
    return {"passed": True, "generator_version": manifest.get("generator", {}).get("version", 1),
            "families": len(family_split), "episodes": len(episodes),
            "features": len(FEATURE_NAMES), "splits": report,
            "interpretation": "Audits export integrity and observable-input construction; does not prove realism or calibration."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.dataset)
        (args.dataset / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    print(json.dumps({key: report[key] for key in ("passed", "families", "episodes", "features")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
