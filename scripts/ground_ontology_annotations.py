#!/usr/bin/env python3
"""Ground reviewed Label Studio entity mentions to local ontology snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from ontology_grounding import run_annotation_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=Path("resources/ontologies/runtime"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/ontologies/runtime/pilot/groundings.parquet"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("resources/ontologies/runtime/pilot/summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("label/ontology_grounding_pilot.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_annotation_pilot(
        args.annotations,
        args.ontology_dir / "aliases.parquet",
        args.ontology_dir / "manifest.json",
        args.output,
        args.summary,
        args.report,
    )
    for entity_type, values in summary["entity_types"].items():
        if entity_type in summary["supported_entity_types"]:
            print(
                f"{entity_type}: {values['matched']:,}/{values['mentions']:,} "
                f"({values['coverage']:.1%}) uniquely grounded"
            )


if __name__ == "__main__":
    main()
