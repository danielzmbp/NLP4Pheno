#!/usr/bin/env python3
"""Apply a reproducible second-pass correction set to reviewed annotations."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotation_utils import select_annotation
from merge_annotation_reviews import sha256, validate_results


def locate(text: str, value: str, occurrence: int = 1) -> tuple[int, int]:
    if occurrence < 1:
        raise ValueError(f"Occurrence must be >= 1, received {occurrence}")
    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = text.find(value, cursor)
        if start < 0:
            raise ValueError(
                f"Could not find occurrence {occurrence} of {value!r} in {text!r}"
            )
        cursor = start + len(value)
    return start, start + len(value)


def relation_key(result: dict[str, Any]) -> tuple[str, str, str]:
    labels = result.get("labels") or []
    return (
        str(result.get("from_id") or ""),
        str(result.get("to_id") or ""),
        str(labels[0]) if labels else "",
    )


def requested_relation_key(values: list[Any]) -> tuple[str, str, str]:
    if len(values) != 3:
        raise ValueError(f"Relation key must have three fields: {values!r}")
    return tuple(str(value) for value in values)  # type: ignore[return-value]


def correction_locator(correction: dict[str, Any]) -> tuple[str, str]:
    """Return the single stable task locator supplied by a correction."""
    candidate_id = str(correction.get("candidate_id") or "")
    task_id = str(correction.get("task_id") or "")
    if bool(candidate_id) == bool(task_id):
        raise ValueError(
            "Each correction must provide exactly one of candidate_id or task_id"
        )
    return ("candidate_id", candidate_id) if candidate_id else ("task_id", task_id)


def locator_label(correction: dict[str, Any]) -> str:
    kind, value = correction_locator(correction)
    return f"{kind}={value}"


def make_entity(
    *,
    result_id: str,
    label: str,
    text: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    return {
        "id": result_id,
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "origin": "manual",
        "value": {
            "start": start,
            "end": end,
            "text": text[start:end],
            "labels": [label],
        },
    }


def apply_one(
    task: dict[str, Any],
    correction: dict[str, Any],
) -> tuple[dict[str, Any], Counter[str]]:
    corrected = copy.deepcopy(task)
    annotation = select_annotation(corrected)
    if annotation is None:
        raise ValueError(f"Task has no submitted annotation: {locator_label(correction)}")
    text = str(corrected.get("data", {}).get("text") or "")
    results = copy.deepcopy(annotation.get("result", []))
    counts: Counter[str] = Counter()

    drop_entities = {str(value) for value in correction.get("drop_entity_ids", [])}
    if drop_entities:
        present = {
            str(result.get("id"))
            for result in results
            if result.get("type") == "labels"
        }
        missing = drop_entities - present
        if missing:
            raise ValueError(f"Unknown entity IDs to drop: {sorted(missing)}")
        endpoint_relations_dropped = sum(
            result.get("type") == "relation"
            and (
                str(result.get("from_id")) in drop_entities
                or str(result.get("to_id")) in drop_entities
            )
            for result in results
        )
        results = [
            result
            for result in results
            if not (
                result.get("type") == "labels"
                and str(result.get("id")) in drop_entities
            )
            and not (
                result.get("type") == "relation"
                and (
                    str(result.get("from_id")) in drop_entities
                    or str(result.get("to_id")) in drop_entities
                )
            )
        ]
        counts["entities_dropped"] += len(drop_entities)
        counts["relations_dropped"] += endpoint_relations_dropped

    entity_by_id = {
        str(result.get("id")): result
        for result in results
        if result.get("type") == "labels"
    }
    for edit in correction.get("edit_entities", []):
        result_id = str(edit["id"])
        if result_id not in entity_by_id:
            raise ValueError(f"Unknown entity ID to edit: {result_id}")
        entity = entity_by_id[result_id]
        value = entity["value"]
        if "text" in edit:
            start, end = locate(
                text, str(edit["text"]), int(edit.get("occurrence", 1))
            )
        else:
            start = int(edit["start"])
            end = int(edit["end"])
        value["start"] = start
        value["end"] = end
        value["text"] = text[start:end]
        if "label" in edit:
            value["labels"] = [str(edit["label"])]
        counts["entities_edited"] += 1

    for addition in correction.get("add_entities", []):
        result_id = str(addition["id"])
        if result_id in entity_by_id:
            raise ValueError(f"Added entity ID already exists: {result_id}")
        start, end = locate(
            text, str(addition["text"]), int(addition.get("occurrence", 1))
        )
        entity = make_entity(
            result_id=result_id,
            label=str(addition["label"]),
            text=text,
            start=start,
            end=end,
        )
        results.append(entity)
        entity_by_id[result_id] = entity
        counts["entities_added"] += 1

    drop_relations = {
        requested_relation_key(values)
        for values in correction.get("drop_relations", [])
    }
    found_drop_relations = {
        relation_key(result)
        for result in results
        if result.get("type") == "relation"
        and relation_key(result) in drop_relations
    }
    missing_relations = drop_relations - found_drop_relations
    if missing_relations:
        raise ValueError(
            f"Unknown relations to drop for {locator_label(correction)}: "
            f"{sorted(missing_relations)}"
        )
    results = [
        result
        for result in results
        if not (
            result.get("type") == "relation"
            and relation_key(result) in drop_relations
        )
    ]
    counts["relations_dropped"] += len(drop_relations)

    entity_ids = set(entity_by_id)
    existing_relations = {
        relation_key(result)
        for result in results
        if result.get("type") == "relation"
    }
    for values in correction.get("add_relations", []):
        source, target, label = requested_relation_key(values)
        if source not in entity_ids or target not in entity_ids:
            raise ValueError(
                f"Added relation references missing entity: "
                f"{source} -> {target} [{label}]"
            )
        key = (source, target, label)
        if key in existing_relations:
            raise ValueError(f"Added relation already exists: {key}")
        results.append(
            {
                "from_id": source,
                "to_id": target,
                "type": "relation",
                "direction": "right",
                "labels": [label],
            }
        )
        existing_relations.add(key)
        counts["relations_added"] += 1

    validate_results(text, results, locator_label(correction))
    annotation["result"] = results
    annotation["ground_truth"] = True
    annotation["second_pass_review"] = {
        "reviewer": str(correction.get("reviewer") or "Codex"),
        "notes": str(correction.get("notes") or ""),
    }
    corrected.setdefault("meta", {})["second_pass_reviewed"] = True
    return corrected, counts


def apply_corrections(
    tasks: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corrected_tasks = copy.deepcopy(tasks)
    candidate_indexes: dict[str, int] = {}
    id_indexes: dict[str, int] = {}
    for index, task in enumerate(corrected_tasks):
        candidate_id = str(task.get("data", {}).get("candidate_id") or "")
        if candidate_id and candidate_id in candidate_indexes:
            raise ValueError(f"Duplicate candidate_id in tasks: {candidate_id}")
        if candidate_id:
            candidate_indexes[candidate_id] = index
        task_id = str(task.get("id") or "")
        if not task_id:
            raise ValueError(f"Task at index {index} lacks id")
        if task_id in id_indexes:
            raise ValueError(f"Duplicate task id in tasks: {task_id}")
        id_indexes[task_id] = index

    seen: set[tuple[str, str]] = set()
    totals: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for correction in corrections:
        kind, value = correction_locator(correction)
        key = (kind, value)
        if key in seen:
            raise ValueError(f"Duplicate correction locator: {kind}={value}")
        indexes = candidate_indexes if kind == "candidate_id" else id_indexes
        if value not in indexes:
            raise ValueError(f"Unknown correction locator: {kind}={value}")
        seen.add(key)
        index = indexes[value]
        corrected, counts = apply_one(corrected_tasks[index], correction)
        corrected_tasks[index] = corrected
        totals.update(counts)
        changed.append(
            {
                kind: value,
                "task_id": str(corrected.get("id") or ""),
                "candidate_id": str(
                    corrected.get("data", {}).get("candidate_id") or ""
                ),
                "human_review_rank": corrected.get("data", {}).get(
                    "human_review_rank"
                ),
                "notes": correction.get("notes", ""),
                **dict(counts),
            }
        )

    return corrected_tasks, {
        "source_tasks": len(tasks),
        "corrected_tasks": len(changed),
        **dict(totals),
        "changes": sorted(
            changed,
            key=lambda row: (
                int(row.get("human_review_rank") or 0),
                row["task_id"],
                row["candidate_id"],
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("corrections", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = json.loads(args.source.read_text())
    corrections = json.loads(args.corrections.read_text())
    corrected, report = apply_corrections(tasks, corrections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, indent=2, ensure_ascii=False) + "\n")
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "corrections": str(args.corrections),
            "corrections_sha256": sha256(args.corrections),
            "output": str(args.output),
            "output_sha256": sha256(args.output),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
