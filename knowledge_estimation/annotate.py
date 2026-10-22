"""Attach external identity/validity annotations to captured B DB snapshots.

Raw DB membership, versions and timestamps cannot be rewritten by annotations.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .measurement import load_dataset, read_jsonl, decision_pairs, label_before, write_json


def attach(root, annotations, output, *, allow_fixture=False):
    root, annotations, output = Path(root), Path(annotations), Path(output)
    manifest, _, _ = load_dataset(root, allow_fixture=allow_fixture)
    labels = {(r["decision_id"], r["phase"]): r for r in read_jsonl(root / "labels.jsonl")}
    additions, assessment_ids = {}, set()
    for annotation in read_jsonl(annotations):
        if set(annotation) != {"decision_id", "phase", "world", "item_assessments"}:
            raise ValueError("annotation has unexpected fields")
        key = (annotation["decision_id"], annotation["phase"])
        if key not in labels or key in additions:
            raise ValueError("unknown or repeated snapshot annotation")
        row = deepcopy(labels[key])
        assessments = annotation["item_assessments"]
        if set(assessments) != set(row["snapshot"]["items"]):
            raise ValueError("annotate only the items actually present in the captured DB")
        for kind, assessment in assessments.items():
            if set(assessment) != {"physical_object_id", "valid_content"}:
                raise ValueError("annotations may assess identity/content, not rewrite DB state")
            row["snapshot"]["items"][kind].update(assessment)
        row["world"] = annotation["world"]
        if manifest["origin"] == "measured" and row["world"]["assessment_method"] not in {"external_camera", "manual_annotation"}:
            raise ValueError("synthetic truth cannot label measured data")
        label_before(row["snapshot"], row["world"], row["snapshot"]["time"])
        assessment_ids.add(row["world"]["assessment_id"])
        additions[key] = row
    if not additions:
        raise ValueError("no annotations supplied")
    # Validate everything before creating the immutable derivative dataset.
    output.mkdir(parents=True, exist_ok=False)
    for relative in manifest["sha256"]:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    raw_annotation = output / "external_annotations.jsonl"
    if raw_annotation.exists():
        raise ValueError("existing annotation derivative; use original raw data as input")
    shutil.copyfile(annotations, raw_annotation)
    manifest = deepcopy(manifest)
    for key, row in additions.items():
        labels[key] = row
    with (output / "labels.jsonl").open("w", encoding="utf-8") as stream:
        for row in labels.values():
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    for evidence_id in assessment_ids:
        if evidence_id not in manifest["evidence"]:
            manifest["evidence"][evidence_id] = {"path": "external_annotations.jsonl",
                "description": "external-camera/manual identity and content assessment supplied offline"}
    manifest["annotations_complete"] = all(r["world"] is not None for r in labels.values())
    manifest["parent_manifest_sha256"] = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    manifest["sha256"] = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.rglob("*") if p.is_file()}
    write_json(output / "manifest.json", manifest)
    load_dataset(output, allow_fixture=allow_fixture)
    list(decision_pairs(output, require_labels=False))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = attach(args.dataset, args.annotations, args.output, allow_fixture=args.allow_fixture)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Annotation stopped: {error}\n")
    print(json.dumps({"output": str(args.output), "annotations_complete": manifest["annotations_complete"]}))


if __name__ == "__main__":
    main()
