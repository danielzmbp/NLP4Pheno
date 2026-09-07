#!/usr/bin/env python3
"""Append reviewed PMC prediction tasks to a versioned annotation export."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotation_utils import select_annotation
from merge_annotation_reviews import sha256, validate_results


def append_reviewed_tasks(
    source_tasks: list[dict[str, Any]],
    reviewed_groups: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = copy.deepcopy(source_tasks)
    source_texts = {
        str(task.get("data", {}).get("text") or "") for task in source_tasks
    }
    existing_ids = [int(task["id"]) for task in source_tasks if str(task.get("id", "")).isdigit()]
    next_task_id = max(existing_ids, default=0) + 1
    seen_candidates: set[str] = set()
    added_by_source: dict[str, int] = {}

    for provenance, tasks in reviewed_groups:
        added = 0
        for task in tasks:
            candidate_id = str(task.get("data", {}).get("candidate_id") or "")
            if not candidate_id:
                raise ValueError(f"Reviewed prediction task lacks candidate_id: {provenance}")
            if candidate_id in seen_candidates:
                raise ValueError(f"Duplicate reviewed candidate_id: {candidate_id}")
            seen_candidates.add(candidate_id)
            text = str(task.get("data", {}).get("text") or "")
            if not text:
                raise ValueError(f"Reviewed prediction task has empty text: {candidate_id}")
            if text in source_texts:
                raise ValueError(
                    f"Reviewed prediction text already exists in source: {candidate_id}"
                )
            annotation = select_annotation(task)
            if annotation is None:
                raise ValueError(f"Prediction review is not submitted: {candidate_id}")
            results = copy.deepcopy(annotation.get("result", []))
            validate_results(text, results, candidate_id)

            new_task = {
                "id": next_task_id,
                "data": copy.deepcopy(task["data"]),
                "meta": {
                    **copy.deepcopy(task.get("meta", {})),
                    "review_append_source": provenance,
                    "candidate_id": candidate_id,
                },
                "annotations": [
                    {
                        **copy.deepcopy(annotation),
                        "id": 2_000_000 + next_task_id,
                        "task": next_task_id,
                        "ground_truth": True,
                        "review_provenance": {
                            "source": provenance,
                            "candidate_id": candidate_id,
                            "label_studio_project_id": annotation.get("project"),
                            "label_studio_task_id": task.get("id"),
                            "label_studio_annotation_id": annotation.get("id"),
                        },
                    }
                ],
            }
            output.append(new_task)
            source_texts.add(text)
            next_task_id += 1
            added += 1
        added_by_source[provenance] = added

    return output, {
        "source_tasks": len(source_tasks),
        "added_tasks": sum(added_by_source.values()),
        "output_tasks": len(output),
        "added_by_source": added_by_source,
        "candidate_ids": sorted(seen_candidates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--reviewed",
        action="append",
        required=True,
        type=Path,
        help="Reviewed JSON file; repeat for multiple review sources",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_tasks = json.loads(args.source.read_text())
    reviewed_groups = [
        (str(path), json.loads(path.read_text())) for path in args.reviewed
    ]
    output, report = append_reviewed_tasks(source_tasks, reviewed_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "reviewed": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.reviewed
            ],
            "output": str(args.output),
            "output_sha256": sha256(args.output),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
