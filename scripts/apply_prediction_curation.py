#!/usr/bin/env python3
"""Apply auditable expert decisions to a model-prediction review queue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relation_key(result: dict[str, Any]) -> tuple[str, str, str]:
    labels = result.get("labels") or []
    return (
        str(result.get("from_id")),
        str(result.get("to_id")),
        str(labels[0]) if labels else "",
    )


def curate_results(
    prediction_results: list[dict[str, Any]],
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    drop_entities = set(decision.get("drop_entity_ids", []))
    results = [
        copy.deepcopy(result)
        for result in prediction_results
        if not (
            result.get("type") == "labels"
            and str(result.get("id")) in drop_entities
        )
    ]
    entity_ids = {
        str(result.get("id"))
        for result in results
        if result.get("type") == "labels"
    }

    keep_relations = {
        tuple(str(value) for value in key)
        for key in decision.get("keep_relations", [])
    }
    drop_all_relations = bool(decision.get("drop_all_relations"))
    curated: list[dict[str, Any]] = []
    for result in results:
        if result.get("type") != "relation":
            curated.append(result)
            continue
        key = relation_key(result)
        if drop_all_relations:
            continue
        if keep_relations and key not in keep_relations:
            continue
        if key[0] not in entity_ids or key[1] not in entity_ids:
            continue
        curated.append(result)

    existing_relations = {
        relation_key(result)
        for result in curated
        if result.get("type") == "relation"
    }
    for source, target, label in decision.get("add_relations", []):
        key = (str(source), str(target), str(label))
        if source not in entity_ids or target not in entity_ids:
            raise ValueError(
                f"Added relation references a removed/missing entity: {key}"
            )
        if key in existing_relations:
            continue
        curated.append(
            {
                "from_id": str(source),
                "to_id": str(target),
                "type": "relation",
                "direction": "right",
                "labels": [str(label)],
            }
        )
        existing_relations.add(key)
    return curated


def apply_decisions(
    tasks: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    curator: str,
    remaining_per_stratum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_by_candidate = {
        str(task.get("data", {}).get("candidate_id")): task for task in tasks
    }
    decision_by_candidate: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id in decision_by_candidate:
            raise ValueError(f"Duplicate decision for {candidate_id}")
        if candidate_id not in task_by_candidate:
            raise ValueError(f"Unknown candidate in decisions: {candidate_id}")
        decision_by_candidate[candidate_id] = decision

    curated_tasks: list[dict[str, Any]] = []
    explicit_deferred: list[dict[str, Any]] = []
    for candidate_id, decision in decision_by_candidate.items():
        status = decision["status"]
        task = copy.deepcopy(task_by_candidate[candidate_id])
        task["data"]["curation_status"] = status
        task["data"]["curation_notes"] = decision.get("notes", "")
        task["meta"]["curator"] = curator
        task["meta"]["curation_status"] = status
        if status == "deferred":
            explicit_deferred.append(task)
            continue
        if status not in {"accepted", "corrected", "hard_negative"}:
            raise ValueError(f"Unsupported curation status: {status}")
        predictions = task.get("predictions") or []
        if len(predictions) != 1:
            raise ValueError(f"Expected one prediction for {candidate_id}")
        results = curate_results(predictions[0].get("result", []), decision)
        task["annotations"] = [
            {
                "id": -len(curated_tasks) - 1,
                "completed_by": curator,
                "result": results,
                "ground_truth": True,
                "was_cancelled": False,
                "lead_time": 0.0,
                "prediction": predictions[0],
            }
        ]
        curated_tasks.append(task)

    undecided = [
        copy.deepcopy(task)
        for task in tasks
        if str(task.get("data", {}).get("candidate_id"))
        not in decision_by_candidate
    ]
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in undecided:
        data = task["data"]
        strata[(data["review_tier"], data["sampled_relation"])].append(task)
    selected_remaining = list(explicit_deferred)
    selected_ids = {
        task["data"]["candidate_id"] for task in selected_remaining
    }
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda task: (
                task["meta"]["seeded_entities"]
                + 2 * task["meta"]["seeded_relations"],
                len(task["data"]["text"]),
                task["data"]["review_rank"],
            ),
        )
        for task in ranked[:remaining_per_stratum]:
            if task["data"]["candidate_id"] not in selected_ids:
                selected_remaining.append(task)
                selected_ids.add(task["data"]["candidate_id"])
    selected_remaining.sort(
        key=lambda task: (
            0 if task["data"].get("curation_status") == "deferred" else 1,
            task["data"]["review_tier"],
            task["data"]["sampled_relation"],
            task["data"]["review_rank"],
        )
    )
    for rank, task in enumerate(selected_remaining, start=1):
        task["data"]["human_review_rank"] = rank

    status_counts = Counter(
        decision["status"] for decision in decision_by_candidate.values()
    )
    report = {
        "source_tasks": len(tasks),
        "decisions": len(decisions),
        "decision_status_counts": dict(sorted(status_counts.items())),
        "curated_tasks": len(curated_tasks),
        "remaining_review_tasks": len(selected_remaining),
        "deferred_unselected_tasks": (
            len(tasks) - len(curated_tasks) - len(selected_remaining)
        ),
        "remaining_per_stratum": remaining_per_stratum,
        "curator": curator,
    }
    return curated_tasks, selected_remaining, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--curated-output", required=True, type=Path)
    parser.add_argument("--remaining-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--curator", default="Codex")
    parser.add_argument("--remaining-per-stratum", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = json.loads(args.queue.read_text())
    decisions = json.loads(args.decisions.read_text())
    curated, remaining, report = apply_decisions(
        tasks,
        decisions,
        curator=args.curator,
        remaining_per_stratum=args.remaining_per_stratum,
    )
    args.curated_output.parent.mkdir(parents=True, exist_ok=True)
    args.curated_output.write_text(
        json.dumps(curated, indent=2, ensure_ascii=False) + "\n"
    )
    args.remaining_output.parent.mkdir(parents=True, exist_ok=True)
    args.remaining_output.write_text(
        json.dumps(remaining, indent=2, ensure_ascii=False) + "\n"
    )
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "queue": str(args.queue),
            "queue_sha256": sha256(args.queue),
            "decisions_file": str(args.decisions),
            "decisions_sha256": sha256(args.decisions),
            "curated_output": str(args.curated_output),
            "curated_output_sha256": sha256(args.curated_output),
            "remaining_output": str(args.remaining_output),
            "remaining_output_sha256": sha256(args.remaining_output),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
