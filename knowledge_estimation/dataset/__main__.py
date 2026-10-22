"""Compatibility entry point; prefer knowledge_estimation.learning.dataset."""

from ..learning.dataset.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
