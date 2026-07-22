#!/usr/bin/env python3
"""Build a conservative Label Studio queue for annotation error review."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from annotation_utils import load_annotations


@dataclass(frozen=True)
class Entity:
    task_id: str
    result_id: str
    start: int
    end: int
    surface: str
    label: str


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_entities(task: dict[str, Any]) -> dict[str, Entity]:
    task_id = str(task.get("id"))
    text = task.get("data", {}).get("text", "")
    entities: dict[str, Entity] = {}
    if not task.get("annotations"):
        return entities
    for result in task["annotations"][0].get("result", []):
        if result.get("type") != "labels":
            continue
        value = result.get("value", {})
        labels = value.get("labels", [])
        if len(labels) != 1:
            continue
        start = int(value["start"])
        end = int(value["end"])
        result_id = str(result.get("id"))
        entities[result_id] = Entity(
            task_id=task_id,
            result_id=result_id,
            start=start,
            end=end,
            surface=str(value.get("text", text[start:end])),
            label=str(labels[0]),
        )
    return entities


def task_relations(task: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    if not task.get("annotations"):
        return relations
    for result in task["annotations"][0].get("result", []):
        if result.get("type") == "relation":
            relations[(str(result.get("from_id")), str(result.get("to_id")))].update(
                str(value) for value in result.get("labels", [])
            )
    return relations


def relation_key(source: Entity, target: Entity) -> tuple[str, str, str]:
    return (
        normalize(source.surface),
        normalize(target.surface),
        f"{source.label}-{target.label}",
    )


def collect_evidence(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    mention_labels: dict[str, Counter[str]] = defaultdict(Counter)
    mention_variants: dict[str, Counter[str]] = defaultdict(Counter)
    entity_occurrences: dict[str, list[Entity]] = defaultdict(list)
    pair_evidence: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"positive": Counter(), "negative": []}
    )

    for task in tasks:
        entities = task_entities(task)
        relations = task_relations(task)
        for entity in entities.values():
            key = normalize(entity.surface)
            mention_labels[key][entity.label] += 1
            mention_variants[key][entity.surface] += 1
            entity_occurrences[key].append(entity)
        for source_id, source in entities.items():
            for target_id, target in entities.items():
                if source_id == target_id:
                    continue
                key = relation_key(source, target)
                labels = relations.get((source_id, target_id), set())
                if labels:
                    pair_evidence[key]["positive"].update(labels)
                else:
                    pair_evidence[key]["negative"].append(
                        (str(task.get("id")), source_id, target_id)
                    )
    return {
        "mention_labels": mention_labels,
        "mention_variants": mention_variants,
        "entity_occurrences": entity_occurrences,
        "pair_evidence": pair_evidence,
    }


def issue(
    *,
    category: str,
    score: float,
    task_id: str,
    summary: str,
    action: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "category": category,
        "score": round(float(score), 6),
        "task_id": task_id,
        "summary": summary,
        "action": action,
        "evidence": evidence,
    }


def entity_conflict_issues(
    evidence: dict[str, Any], *, min_support: int, min_purity: float
) -> list[dict[str, Any]]:
    output = []
    for surface, counts in evidence["mention_labels"].items():
        total = sum(counts.values())
        if total < min_support or len(counts) < 2:
            continue
        dominant_label, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / total
        if purity < min_purity:
            continue
        for occurrence in evidence["entity_occurrences"][surface]:
            if occurrence.label == dominant_label:
                continue
            output.append(
                issue(
                    category="entity_label_conflict",
                    score=0.9 + 0.09 * purity,
                    task_id=occurrence.task_id,
                    summary=(
                        f"Review {occurrence.surface!r}: currently {occurrence.label}, "
                        f"but {dominant_count}/{total} matching annotations use {dominant_label}."
                    ),
                    action={
                        "kind": "replace_entity_label",
                        "result_id": occurrence.result_id,
                        "current_label": occurrence.label,
                        "suggested_label": dominant_label,
                    },
                    evidence={
                        "surface": occurrence.surface,
                        "label_counts": dict(sorted(counts.items())),
                        "dominant_fraction": purity,
                    },
                )
            )
    return output


def entity_omission_issues(
    tasks: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    min_specific_support: int,
    min_general_support: int,
    max_per_surface: int,
) -> list[dict[str, Any]]:
    lexicon = []
    for surface, counts in evidence["mention_labels"].items():
        if len(counts) != 1 or len(surface) < 4 or not any(char.isalpha() for char in surface):
            continue
        label, support = counts.most_common(1)[0]
        threshold = (
            min_specific_support if label in {"STRAIN", "SPECIES"} else min_general_support
        )
        if support < threshold:
            continue
        # Use the most frequently annotated literal spelling. Exact matching
        # keeps offsets valid and avoids fuzzy suggestions.
        variant = evidence["mention_variants"][surface].most_common(1)[0][0]
        lexicon.append((support, surface, variant, label))
    lexicon.sort(reverse=True)

    output = []
    surface_counts: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task.get("id"))
        text = str(task.get("data", {}).get("text", ""))
        occupied = [(entity.start, entity.end) for entity in task_entities(task).values()]
        for support, surface, variant, label in lexicon:
            if surface_counts[surface] >= max_per_surface:
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                if any(match.start() < end and match.end() > start for start, end in occupied):
                    continue
                surface_counts[surface] += 1
                output.append(
                    issue(
                        category="entity_omission",
                        score=min(0.97, 0.82 + 0.07 * math.log10(support)),
                        task_id=task_id,
                        summary=(
                            f"Review unannotated {match.group()!r} as {label}; the exact mention "
                            f"is labeled {label} in {support} other occurrences."
                        ),
                        action={
                            "kind": "add_entity",
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "suggested_label": label,
                        },
                        evidence={"surface": surface, "support": support, "label": label},
                    )
                )
                if surface_counts[surface] >= max_per_surface:
                    break
    return output


def relation_omission_issues(
    evidence: dict[str, Any],
    *,
    configured_relations: set[str],
    min_positive: int,
    min_rate: float,
) -> list[dict[str, Any]]:
    output = []
    for key, values in evidence["pair_evidence"].items():
        positive: Counter[str] = values["positive"]
        negatives: list[tuple[str, str, str]] = values["negative"]
        if not positive or not negatives:
            continue
        relation, positive_count = positive.most_common(1)[0]
        if positive_count < min_positive or positive_count != sum(positive.values()):
            continue
        typed_relation = f"{key[2]}:{relation}"
        if typed_relation not in configured_relations:
            continue
        rate = positive_count / (positive_count + len(negatives))
        if rate < min_rate:
            continue
        for task_id, source_id, target_id in negatives:
            output.append(
                issue(
                    category="relation_omission",
                    score=0.85 + 0.14 * rate,
                    task_id=task_id,
                    summary=(
                        f"Review missing {typed_relation} between {key[0]!r} and {key[1]!r}; "
                        f"it is present in {positive_count}/{positive_count + len(negatives)} "
                        "matching contexts."
                    ),
                    action={
                        "kind": "add_relation",
                        "from_id": source_id,
                        "to_id": target_id,
                        "suggested_label": relation,
                    },
                    evidence={
                        "typed_relation": typed_relation,
                        "positive": positive_count,
                        "negative": len(negatives),
                        "positive_fraction": rate,
                    },
                )
            )
    return output


def result_id(task_id: str, action: dict[str, Any]) -> str:
    payload = json.dumps([task_id, action], sort_keys=True).encode()
    return "review_" + hashlib.sha1(payload).hexdigest()[:12]


def apply_suggestions(task: dict[str, Any], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if task.get("annotations"):
        results = copy.deepcopy(task["annotations"][0].get("result", []))
    else:
        results = []
    by_id = {str(item.get("id")): item for item in results if item.get("id") is not None}
    existing_relations = {
        (str(item.get("from_id")), str(item.get("to_id")), tuple(item.get("labels", [])))
        for item in results
        if item.get("type") == "relation"
    }
    for candidate in issues:
        action = candidate["action"]
        if action["kind"] == "replace_entity_label":
            target = by_id.get(str(action["result_id"]))
            if target is not None:
                target["value"]["labels"] = [action["suggested_label"]]
        elif action["kind"] == "add_entity":
            identifier = result_id(str(task.get("id")), action)
            result = {
                "id": identifier,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": action["start"],
                    "end": action["end"],
                    "text": action["text"],
                    "labels": [action["suggested_label"]],
                },
            }
            results.append(result)
            by_id[identifier] = result
        elif action["kind"] == "add_relation":
            relation = (
                str(action["from_id"]),
                str(action["to_id"]),
                (action["suggested_label"],),
            )
            if relation not in existing_relations:
                results.append(
                    {
                        "from_id": action["from_id"],
                        "to_id": action["to_id"],
                        "type": "relation",
                        "direction": "right",
                        "labels": [action["suggested_label"]],
                    }
                )
                existing_relations.add(relation)
    return results


def build_queue(
    tasks: list[dict[str, Any]],
    *,
    configured_relations: set[str],
    max_tasks: int,
    conflict_min_support: int = 5,
    conflict_min_purity: float = 0.85,
    omission_specific_support: int = 8,
    omission_general_support: int = 20,
    omission_max_per_surface: int = 20,
    relation_min_positive: int = 4,
    relation_min_rate: float = 0.8,
    include_label_conflicts: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence = collect_evidence(tasks)
    candidates = []
    if include_label_conflicts:
        candidates.extend(
            entity_conflict_issues(
                evidence,
                min_support=conflict_min_support,
                min_purity=conflict_min_purity,
            )
        )
    candidates.extend(
        entity_omission_issues(
            tasks,
            evidence,
            min_specific_support=omission_specific_support,
            min_general_support=omission_general_support,
            max_per_surface=omission_max_per_surface,
        )
    )
    candidates.extend(
        relation_omission_issues(
            evidence,
            configured_relations=configured_relations,
            min_positive=relation_min_positive,
            min_rate=relation_min_rate,
        )
    )
    category_order = {
        "entity_label_conflict": 0,
        "entity_omission": 1,
        "relation_omission": 2,
    }
    candidates.sort(
        key=lambda item: (
            category_order[item["category"]],
            -item["score"],
            item["task_id"],
            json.dumps(item["action"], sort_keys=True),
        )
    )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate["task_id"]].append(candidate)
    ranked_tasks = sorted(
        by_task,
        key=lambda task_id: (
            min(category_order[item["category"]] for item in by_task[task_id]),
            -max(item["score"] for item in by_task[task_id]),
            task_id,
        ),
    )[:max_tasks]

    tasks_by_id = {str(task.get("id")): task for task in tasks}
    queue = []
    selected_issues = []
    for rank, task_id in enumerate(ranked_tasks, 1):
        task = tasks_by_id[task_id]
        task_issues = by_task[task_id]
        selected_issues.extend(task_issues)
        summaries = [item["summary"] for item in task_issues]
        score = max(item["score"] for item in task_issues)
        queue.append(
            {
                "data": {
                    "text": task.get("data", {}).get("text", ""),
                    "original_task_id": task_id,
                    "review_rank": rank,
                    "review_score": score,
                    "review_categories": ", ".join(
                        sorted({item["category"] for item in task_issues})
                    ),
                    "review_summary": " | ".join(summaries),
                },
                "meta": {
                    "original_task_id": task_id,
                    "review_rank": rank,
                    "review_score": score,
                    "review_issues": task_issues,
                },
                "predictions": [
                    {
                        "model_version": "annotation-review-heuristics-v1",
                        "score": score,
                        "result": apply_suggestions(task, task_issues),
                    }
                ],
            }
        )

    candidate_counts = Counter(item["category"] for item in candidates)
    selected_counts = Counter(item["category"] for item in selected_issues)
    summary = {
        "tasks_in_source": len(tasks),
        "candidate_issues": len(candidates),
        "candidate_tasks": len(by_task),
        "candidate_issues_by_category": dict(sorted(candidate_counts.items())),
        "selected_tasks": len(queue),
        "selected_issues": len(selected_issues),
        "selected_issues_by_category": dict(sorted(selected_counts.items())),
        "thresholds": {
            "conflict_min_support": conflict_min_support,
            "conflict_min_purity": conflict_min_purity,
            "omission_specific_support": omission_specific_support,
            "omission_general_support": omission_general_support,
            "omission_max_per_surface": omission_max_per_surface,
            "relation_min_positive": relation_min_positive,
            "relation_min_rate": relation_min_rate,
            "max_tasks": max_tasks,
            "include_label_conflicts": include_label_conflicts,
        },
    }
    return queue, selected_issues, summary


def write_issue_tsv(path: Path, issues: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "category", "score", "summary", "action", "evidence"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for item in issues:
            writer.writerow(
                {
                    **{key: item[key] for key in ("task_id", "category", "score", "summary")},
                    "action": json.dumps(item["action"], sort_keys=True),
                    "evidence": json.dumps(item["evidence"], sort_keys=True),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--issues-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=250)
    parser.add_argument("--skip-label-conflicts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_annotations(args.annotations)
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    queue, issues, summary = build_queue(
        tasks,
        configured_relations=set(config.get("rel_labels", [])),
        max_tasks=args.max_tasks,
        include_label_conflicts=not args.skip_label_conflicts,
    )
    summary.update(
        {
            "source": str(args.annotations),
            "source_sha256": sha256(args.annotations),
            "config": str(args.config),
        }
    )
    args.queue_output.parent.mkdir(parents=True, exist_ok=True)
    args.queue_output.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n")
    write_issue_tsv(args.issues_output, issues)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
