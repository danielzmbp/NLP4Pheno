#!/usr/bin/env python3
"""Benchmark conservative StrainInfo matching against annotated STRAIN spans."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from annotation_utils import load_annotations
from straininfo_matching import (
    resolve_mentions,
)


def strain_mentions(annotation_file: Path) -> list[str]:
    mentions: set[str] = set()
    for task in load_annotations(annotation_file):
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                value = result.get("value", {})
                if result.get("type") == "labels" and "STRAIN" in value.get("labels", []):
                    text = value.get("text")
                    if text:
                        mentions.add(str(text))
    return sorted(mentions)


def audit(annotation_file: Path, designations_file: Path) -> dict:
    mentions = strain_mentions(annotation_file)
    aliases = pl.read_parquet(designations_file).select(
        "designation_key", "designation", "si_id", "taxon", "type_strain"
    )
    resolved = resolve_mentions(mentions, aliases)
    resolutions = [
        {
            "mention": row["mention"],
            "status": row["straininfo_status"],
            "method": row["straininfo_method"],
            "si_id": row["straininfo_si_id"],
            "taxon": row["straininfo_taxon"],
            "si_ids": row["straininfo_si_ids"],
            "designation_keys": row["straininfo_designation_keys"],
        }
        for row in resolved.iter_rows(named=True)
    ]
    counts = Counter(row["straininfo_status"] for row in resolved.iter_rows(named=True))
    methods = Counter(
        row["straininfo_method"]
        for row in resolved.iter_rows(named=True)
        if row["straininfo_method"]
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotation_file": str(annotation_file),
        "designations_file": str(designations_file),
        "policy": {
            "exact": "uppercase alphanumeric equality",
            "contained": "complete token-bounded alias, alphanumeric length >=4, contains letters and digits",
            "taxonomy": "reject a candidate when an explicit binomial hint contradicts the catalog taxon",
            "fuzzy_fallback": False,
        },
        "unique_mentions": len(mentions),
        "status_counts": dict(sorted(counts.items())),
        "method_counts": dict(sorted(methods.items())),
        "matches": resolutions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_file", type=Path)
    parser.add_argument("designations_file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.annotation_file, args.designations_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "matches"}, indent=2))


if __name__ == "__main__":
    main()
