"""Compatibility entry point for learning/dataset/audit.py."""

from ..learning.dataset.audit import audit_dataset, main, read_jsonl

__all__ = ["audit_dataset", "read_jsonl"]

if __name__ == "__main__":
    raise SystemExit(main())
