#!/usr/bin/env python3
"""Compare generated evaluation files with a frozen split manifest by hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for section, suffix in (("ner", "jsonls"), ("rel", "json")):
        for label, frozen in sorted(manifest.get(section, {}).items()):
            for split in ("dev", "test"):
                path = run / section.upper() / label / f"{split}.{suffix}"
                expected = str(frozen[f"{split}_sha256"])
                actual = sha256(path) if path.is_file() else None
                rows.append(
                    {
                        "section": section,
                        "label": label,
                        "split": split,
                        "path": str(path),
                        "exists": path.is_file(),
                        "expected_sha256": expected,
                        "actual_sha256": actual,
                        "matches": actual == expected,
                    }
                )
    missing = [row for row in rows if not row["exists"]]
    drifted = [row for row in rows if row["exists"] and not row["matches"]]
    return {
        "manifest_name": manifest.get("name"),
        "run": str(run),
        "files": len(rows),
        "matching_files": sum(row["matches"] for row in rows),
        "missing_files": len(missing),
        "drifted_files": len(drifted),
        "drifted": drifted,
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Exit non-zero if a file is missing or its content has changed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = compare(args.run, manifest)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if args.require_match and (report["missing_files"] or report["drifted_files"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
