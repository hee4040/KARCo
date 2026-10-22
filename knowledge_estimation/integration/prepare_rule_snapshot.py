"""Export the locally fetched rule branch for integration without switching it.

The ignored copy is byte-for-byte teammate code. After merging that branch,
the normal root-level rule_based_comm import works without this helper.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/heewoo/rule-based")
    parser.add_argument("--output", type=Path, default=Path("knowledge_estimation/data/rule_snapshot"))
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    def git(*values):
        return subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", *values], cwd=repo)
    # Resolve a commit before exporting, so the ref cannot change mid-export.
    commit = git("rev-parse", "--verify", f"{args.ref}^{{commit}}").decode().strip()
    paths = git("ls-tree", "-r", "--name-only", commit, "rule_based_comm").decode().splitlines()
    if not paths:
        parser.error("the selected commit has no rule_based_comm package")
    args.output.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for relative in paths:
        path = args.output / relative
        if not path.resolve().is_relative_to(args.output.resolve()) or not relative.endswith(".py"):
            parser.error("unexpected package path")
        content = git("show", f"{commit}:{relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        checksums[relative] = hashlib.sha256(content).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps({"ref": args.ref, "commit": commit,
        "sha256": checksums}, indent=2), encoding="utf-8")
    print(json.dumps({"commit": commit, "output": str(args.output), "files": len(paths)}))


if __name__ == "__main__":
    main()
