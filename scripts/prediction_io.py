"""Small, testable I/O helpers shared by model-inference scripts."""

from __future__ import annotations

from pathlib import Path


def load_corpus_lines(path: str | Path) -> list[str]:
    """Read one sentence per line without discarding a final unterminated line."""
    with Path(path).open(encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]
