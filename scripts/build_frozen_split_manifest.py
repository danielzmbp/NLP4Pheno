#!/usr/bin/env python3
"""Record task IDs from an existing NER/RE run as immutable eval splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_ids(values: set[str]) -> list[int | str]:
    def key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return [int(value) if value.isdigit() else value for value in sorted(values, key=key)]


def ner_task_ids(path: Path) -> set[str]:
    with path.open() as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a JSON task list: {path}")
    return {str(task["id"]) for task in tasks}


def relation_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" not in row:
                raise ValueError(f"Missing task_id in {path}:{line_number}")
            task_ids.add(str(row["task_id"]))
    return task_ids


def collect_section(root: Path, loader: Any) -> dict[str, dict[str, Any]]:
    section: dict[str, dict[str, Any]] = {}
    suffix = "jsonls" if root.name == "NER" else "json"
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        dev_path = directory / f"dev.{suffix}"
        test_path = directory / f"test.{suffix}"
        if not dev_path.exists() or not test_path.exists():
            continue
        dev_ids = loader(dev_path)
        test_ids = loader(test_path)
        overlap = dev_ids & test_ids
        if overlap:
            raise ValueError(
                f"{directory.name} has task IDs in both dev and test: {sorted(overlap)[:10]}"
            )
        section[directory.name] = {
            "dev_task_ids": sorted_ids(dev_ids),
            "test_task_ids": sorted_ids(test_ids),
            "dev_file": str(dev_path),
            "dev_sha256": sha256(dev_path),
            "test_file": str(test_path),
            "test_sha256": sha256(test_path),
        }
    if not section:
        raise ValueError(f"No completed split directories found under {root}")
    return section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "schema_version": 1,
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(args.run.resolve()),
        "ner": collect_section(args.run / "NER", ner_task_ids),
        "rel": collect_section(args.run / "REL", relation_task_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ner_labels": len(manifest["ner"]),
                "relation_labels": len(manifest["rel"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
