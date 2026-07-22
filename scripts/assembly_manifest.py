"""Validation helpers for strain-to-assembly manifests."""

from __future__ import annotations

from pathlib import Path


def load_assembly_manifest(
    path: str | Path, *, require_nonempty: bool = True
) -> list[tuple[str, str]]:
    """Load unique ``strain/assembly`` records and reject malformed input."""
    records: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("/")
            if len(fields) != 2 or not all(fields):
                raise ValueError(
                    f"Malformed assembly manifest line {line_number} in {path}: {line!r}"
                )
            record = (fields[0], fields[1])
            if record not in seen:
                seen.add(record)
                records.append(record)
    if require_nonempty and not records:
        raise ValueError(f"Assembly manifest is empty: {path}")
    return records
