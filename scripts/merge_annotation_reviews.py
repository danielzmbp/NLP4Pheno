#!/usr/bin/env python3
"""Merge submitted Label Studio review annotations into a versioned export."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotation_utils import select_annotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_results(text: str, results: list[dict[str, Any]], task_id: str) -> None:
    entity_ids: set[str] = set()
    for result in results:
        result_type = result.get("type")
        if result_type == "labels":
            value = result.get("value", {})
            start = int(value.get("start", -1))
            end = int(value.get("end", -1))
            if start < 0 or end <= start or end > len(text):
                raise ValueError(f"Invalid reviewed span in task {task_id}: {start}:{end}")
            if text[start:end] != value.get("text"):
                raise ValueError(
                    f"Reviewed span text mismatch in task {task_id}: "
                    f"{text[start:end]!r} != {value.get('text')!r}"
                )
            result_id = str(result.get("id") or "")
            if not result_id or result_id in entity_ids:
                raise ValueError(f"Missing or duplicate entity ID in task {task_id}: {result_id!r}")
            entity_ids.add(result_id)
        elif result_type != "relation":
            raise ValueError(f"Unknown reviewed result type in task {task_id}: {result_type!r}")

    for result in results:
        if result.get("type") != "relation":
            continue
        source = str(result.get("from_id") or "")
        target = str(result.get("to_id") or "")
        if source not in entity_ids or target not in entity_ids:
            raise ValueError(
                f"Missing reviewed relation endpoint in task {task_id}: {source!r} -> {target!r}"
            )


def result_signature(results: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Return a semantic signature that ignores Label Studio UI metadata."""
    entities: dict[str, tuple[Any, ...]] = {}
    signatures: list[tuple[Any, ...]] = []
    for result in results:
        if result.get("type") != "labels":
            continue
        value = result.get("value", {})
        entity = (
            "labels",
            int(value.get("start", -1)),
            int(value.get("end", -1)),
            str(value.get("text") or ""),
            tuple(value.get("labels", [])),
        )
        entities[str(result.get("id"))] = entity
        signatures.append(entity)
    for result in results:
        if result.get("type") != "relation":
            continue
        signatures.append(
            (
                "relation",
                entities.get(str(result.get("from_id"))),
                entities.get(str(result.get("to_id"))),
                tuple(result.get("labels", [])),
            )
        )
    return tuple(sorted(signatures, key=repr))


def merged_annotation(
    review_annotation: dict[str, Any],
    *,
    original_task_id: Any,
    source_project_id: Any,
    review_task_id: Any,
    reviewer: str | None,
) -> dict[str, Any]:
    annotation = copy.deepcopy(review_annotation)
    review_annotation_id = annotation.get("id")
    annotation["id"] = 1_000_000 + int(review_annotation_id or 0)
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        annotation["completed_by"] = completed_by.get("email") or completed_by.get("id")
    if reviewer:
        annotation["completed_by"] = reviewer
    annotation["task"] = original_task_id
    annotation["project"] = source_project_id
    annotation["ground_truth"] = True
    annotation["review_provenance"] = {
        "label_studio_project_id": review_annotation.get("project"),
        "label_studio_task_id": review_task_id,
        "label_studio_annotation_id": review_annotation_id,
    }
    return annotation


def merge_reviews(
    source_tasks: list[dict[str, Any]],
    review_tasks: list[dict[str, Any]],
    *,
    reviewer: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged = copy.deepcopy(source_tasks)
    source_by_id: dict[str, dict[str, Any]] = {}
    for task in merged:
        task_id = str(task.get("id"))
        if task_id in source_by_id:
            raise ValueError(f"Duplicate source task ID: {task_id}")
        source_by_id[task_id] = task

    seen_review_ids: set[str] = set()
    changed = 0
    unchanged = 0
    for review_task in review_tasks:
        original_id = str(review_task.get("data", {}).get("original_task_id") or "")
        if not original_id or original_id not in source_by_id:
            raise ValueError(f"Unknown original task ID in review export: {original_id!r}")
        if original_id in seen_review_ids:
            raise ValueError(f"Duplicate review for original task ID: {original_id}")
        seen_review_ids.add(original_id)

        source_task = source_by_id[original_id]
        source_text = str(source_task.get("data", {}).get("text", ""))
        review_text = str(review_task.get("data", {}).get("text", ""))
        if source_text != review_text:
            raise ValueError(f"Review text differs from source task {original_id}")
        selected = select_annotation(review_task)
        if selected is None:
            raise ValueError(f"Review task {original_id} has no submitted annotation")
        results = copy.deepcopy(selected.get("result", []))
        validate_results(source_text, results, original_id)

        old = select_annotation(source_task)
        old_results = old.get("result", []) if old else []
        if result_signature(old_results) == result_signature(results):
            unchanged += 1
        else:
            changed += 1
        source_project_id = old.get("project") if old else None
        source_task["annotations"] = [
            merged_annotation(
                selected,
                original_task_id=source_task.get("id"),
                source_project_id=source_project_id,
                review_task_id=review_task.get("id"),
                reviewer=reviewer,
            )
        ]

    report = {
        "source_tasks": len(source_tasks),
        "review_tasks": len(review_tasks),
        "merged_reviews": len(seen_review_ids),
        "changed_tasks": changed,
        "unchanged_tasks": unchanged,
        "reviewed_original_task_ids": sorted(seen_review_ids, key=int),
    }
    return merged, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviewer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_tasks = json.loads(args.source.read_text())
    review_tasks = json.loads(args.reviews.read_text())
    merged, report = merge_reviews(source_tasks, review_tasks, reviewer=args.reviewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "reviews": str(args.reviews),
            "reviews_sha256": sha256(args.reviews),
            "output": str(args.output),
            "output_sha256": sha256(args.output),
            "reviewer": args.reviewer,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
