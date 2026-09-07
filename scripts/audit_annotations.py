#!/usr/bin/env python3
"""Audit Label Studio exports used by the NER and relation pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from annotation_utils import select_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "p95": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
    }


def result_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    if result.get("type") == "labels":
        value = result.get("value", {})
        return (
            "labels",
            value.get("start"),
            value.get("end"),
            value.get("text"),
            tuple(value.get("labels", [])),
        )
    if result.get("type") == "relation":
        return (
            "relation",
            result.get("from_id"),
            result.get("to_id"),
            tuple(result.get("labels", [])),
        )
    return (result.get("type"), json.dumps(result, sort_keys=True))


def annotation_signature(task: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    annotation = select_annotation(task)
    if annotation is None:
        return ()
    results = annotation.get("result", [])
    entity_endpoints = {}
    for item in results:
        if item.get("type") != "labels":
            continue
        value = item.get("value", {})
        entity_endpoints[item.get("id")] = (
            value.get("start"),
            value.get("end"),
            value.get("text"),
            tuple(value.get("labels", [])),
        )

    signatures = []
    for item in results:
        if item.get("type") != "relation":
            signatures.append(result_signature(item))
            continue
        from_id = item.get("from_id")
        to_id = item.get("to_id")
        signatures.append(
            (
                "relation",
                entity_endpoints.get(from_id, ("missing", from_id)),
                entity_endpoints.get(to_id, ("missing", to_id)),
                tuple(item.get("labels", [])),
            )
        )
    return tuple(sorted(signatures, key=repr))


def audit_export(path: Path, configured_entities: set[str], configured_relations: set[str]) -> dict[str, Any]:
    with path.open() as handle:
        tasks = json.load(handle)

    task_ids = [task.get("id") for task in tasks]
    text_counts = Counter(task.get("data", {}).get("text", "") for task in tasks)
    normalized_text_counts = Counter(
        re.sub(r"\s+", " ", task.get("data", {}).get("text", "")).strip()
        for task in tasks
    )

    annotators: Counter[str] = Counter()
    annotator_empty_tasks: Counter[str] = Counter()
    annotator_entity_labels: defaultdict[str, Counter[str]] = defaultdict(Counter)
    annotator_relation_labels: defaultdict[str, Counter[str]] = defaultdict(Counter)
    entity_labels: Counter[str] = Counter()
    entity_tasks: defaultdict[str, set[str]] = defaultdict(set)
    relation_labels: Counter[str] = Counter()
    relation_patterns: Counter[str] = Counter()
    relation_pattern_tasks: defaultdict[str, set[str]] = defaultdict(set)
    annotation_counts: list[int] = []
    multi_annotator_tasks = 0
    data_fields: set[str] = set()
    metadata_fields: set[str] = set()
    entity_counts_per_task: list[int] = []
    relation_counts_per_task: list[int] = []
    active_annotations = 0
    canonical_annotations = 0
    cancelled_annotations = 0
    invalid_spans = 0
    span_text_mismatches = 0
    multi_label_spans = 0
    duplicate_spans = 0
    overlapping_span_pairs = 0
    same_entity_type_overlap_pairs = 0
    cross_entity_type_overlap_pairs = 0
    missing_relation_endpoints = 0
    self_relations = 0
    unlabeled_relations = 0
    multi_label_relations = 0
    duplicate_relations = 0
    unknown_result_types: Counter[str] = Counter()
    issue_samples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def sample(kind: str, payload: dict[str, Any]) -> None:
        if len(issue_samples[kind]) < 20:
            issue_samples[kind].append(payload)

    for task in tasks:
        task_id = task.get("id")
        text = task.get("data", {}).get("text", "")
        annotations = task.get("annotations", [])
        active_task_annotations = [
            annotation
            for annotation in annotations
            if not annotation.get("was_cancelled", False)
        ]
        data_fields.update(task.get("data", {}).keys())
        metadata_fields.update(task.get("meta", {}).keys())
        annotation_counts.append(len(active_task_annotations))
        if len(active_task_annotations) > 1:
            sample(
                "multiple_active_annotations",
                {
                    "task_id": task_id,
                    "annotation_ids": [
                        annotation.get("id") for annotation in active_task_annotations
                    ],
                },
            )
        task_annotators = {
            str(annotation.get("completed_by"))
            for annotation in active_task_annotations
        }
        multi_annotator_tasks += len(task_annotators) > 1
        task_entity_count = 0
        task_relation_count = 0

        active_annotations += len(active_task_annotations)
        cancelled_annotations += sum(
            annotation.get("was_cancelled", False) for annotation in annotations
        )
        selected = select_annotation(task)
        selected_annotations = [selected] if selected is not None else []
        canonical_annotations += len(selected_annotations)

        for annotation in selected_annotations:
            if annotation.get("was_cancelled"):
                continue
            annotator = str(annotation.get("completed_by"))
            annotators[annotator] += 1
            results = annotation.get("result", [])
            if not results:
                annotator_empty_tasks[annotator] += 1
            entities: dict[str, list[str]] = {}
            spans: list[tuple[int, int, tuple[str, ...], str | None]] = []
            seen_spans: Counter[tuple[Any, ...]] = Counter()
            seen_relations: Counter[tuple[Any, ...]] = Counter()

            for result in results:
                result_type = result.get("type")
                if result_type == "labels":
                    task_entity_count += 1
                    value = result.get("value", {})
                    labels = value.get("labels", [])
                    start = value.get("start")
                    end = value.get("end")
                    annotated_text = value.get("text")
                    result_id = result.get("id")
                    entities[result_id] = labels
                    if len(labels) != 1:
                        multi_label_spans += 1
                        sample("multi_label_spans", {"task_id": task_id, "result_id": result_id, "labels": labels})
                    entity_labels.update(labels)
                    annotator_entity_labels[annotator].update(labels)
                    for label in labels:
                        entity_tasks[label].add(str(task_id))

                    valid = (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start < end <= len(text)
                    )
                    if not valid:
                        invalid_spans += 1
                        sample("invalid_spans", {"task_id": task_id, "result_id": result_id, "start": start, "end": end, "text_length": len(text)})
                    elif text[start:end] != annotated_text:
                        span_text_mismatches += 1
                        sample("span_text_mismatches", {"task_id": task_id, "result_id": result_id, "slice": text[start:end], "annotated_text": annotated_text})

                    span_key = (start, end, tuple(labels), annotated_text)
                    seen_spans[span_key] += 1
                    if valid:
                        spans.append((start, end, tuple(labels), result_id))
                elif result_type == "relation":
                    task_relation_count += 1
                else:
                    unknown_result_types[str(result_type)] += 1

            duplicate_spans += sum(count - 1 for count in seen_spans.values() if count > 1)
            spans.sort()
            for index, left in enumerate(spans):
                for right in spans[index + 1 :]:
                    if right[0] >= left[1]:
                        break
                    overlapping_span_pairs += 1
                    if set(left[2]) & set(right[2]):
                        same_entity_type_overlap_pairs += 1
                    else:
                        cross_entity_type_overlap_pairs += 1

            for result in results:
                if result.get("type") != "relation":
                    continue
                from_id = result.get("from_id")
                to_id = result.get("to_id")
                labels = result.get("labels", [])
                relation_labels.update(labels)
                annotator_relation_labels[annotator].update(labels)
                if not labels:
                    unlabeled_relations += 1
                    sample("unlabeled_relations", {"task_id": task_id, "from_id": from_id, "to_id": to_id})
                if len(labels) > 1:
                    multi_label_relations += 1
                if from_id == to_id:
                    self_relations += 1
                if from_id not in entities or to_id not in entities:
                    missing_relation_endpoints += 1
                    sample("missing_relation_endpoints", {"task_id": task_id, "from_id": from_id, "to_id": to_id})
                from_labels = entities.get(from_id, ["MISSING"])
                to_labels = entities.get(to_id, ["MISSING"])
                for from_label in from_labels:
                    for to_label in to_labels:
                        for relation_label in labels:
                            pattern = f"{from_label}-{to_label}:{relation_label}"
                            relation_patterns[pattern] += 1
                            relation_pattern_tasks[pattern].add(str(task_id))
                relation_key = (from_id, to_id, tuple(labels))
                seen_relations[relation_key] += 1
            duplicate_relations += sum(count - 1 for count in seen_relations.values() if count > 1)

        entity_counts_per_task.append(task_entity_count)
        relation_counts_per_task.append(task_relation_count)

    duplicate_text_groups = sum(1 for text, count in text_counts.items() if text and count > 1)
    normalized_duplicate_groups = sum(
        1 for text, count in normalized_text_counts.items() if text and count > 1
    )
    observed_entities = set(entity_labels)
    observed_relations = set(relation_patterns)
    unconfigured_relations = observed_relations - configured_relations
    unconfigured_relation_labels = sum(
        relation_patterns[pattern] for pattern in unconfigured_relations
    )
    unconfigured_strain_relations = {
        pattern
        for pattern in unconfigured_relations
        if "STRAIN" in pattern.partition(":")[0].split("-")
    }
    configured_relation_support = {
        pattern: {
            "positive_labels": relation_patterns[pattern],
            "positive_tasks": len(relation_pattern_tasks[pattern]),
        }
        for pattern in sorted(configured_relations)
    }
    low_support_relations = [
        pattern
        for pattern, support in configured_relation_support.items()
        if support["positive_tasks"] < 50
    ]
    annotator_profiles = {}
    for annotator, task_count in sorted(annotators.items()):
        entity_count = sum(annotator_entity_labels[annotator].values())
        relation_count = sum(annotator_relation_labels[annotator].values())
        annotator_profiles[annotator] = {
            "tasks": task_count,
            "empty_tasks": annotator_empty_tasks[annotator],
            "entities": entity_count,
            "relations": relation_count,
            "entities_per_task": round(entity_count / task_count, 3),
            "relations_per_task": round(relation_count / task_count, 3),
            "entity_labels_per_100_tasks": {
                label: round(100 * count / task_count, 3)
                for label, count in sorted(annotator_entity_labels[annotator].items())
            },
            "relation_labels_per_100_tasks": {
                label: round(100 * count / task_count, 3)
                for label, count in sorted(annotator_relation_labels[annotator].items())
            },
        }

    return {
        "path": str(path),
        "sha256": sha256(path),
        "tasks": len(tasks),
        "unique_task_ids": len(set(task_ids)),
        "duplicate_task_ids": len(task_ids) - len(set(task_ids)),
        "empty_text_tasks": text_counts.get("", 0),
        "duplicate_text_groups": duplicate_text_groups,
        "normalized_duplicate_text_groups": normalized_duplicate_groups,
        "annotations": {
            "active_records": active_annotations,
            "canonical": canonical_annotations,
            "selection_policy": "ground_truth_else_latest_active",
            "cancelled": cancelled_annotations,
            "per_task": distribution(annotation_counts),
            "tasks_with_multiple": sum(count > 1 for count in annotation_counts),
            "tasks_with_multiple_annotators": multi_annotator_tasks,
            "annotators": dict(sorted(annotators.items())),
            "annotator_profiles": annotator_profiles,
        },
        "provenance": {
            "data_fields": sorted(data_fields),
            "metadata_fields": sorted(metadata_fields),
            "has_document_identifier": any(
                field.lower() in {"pmcid", "pmid", "doi", "document_id", "article_id"}
                for field in data_fields | metadata_fields
            ),
        },
        "entities": {
            "total": sum(entity_labels.values()),
            "by_label": dict(sorted(entity_labels.items())),
            "tasks_by_label": {
                label: len(tasks) for label, tasks in sorted(entity_tasks.items())
            },
            "per_task": distribution(entity_counts_per_task),
            "configured_but_unobserved": sorted(configured_entities - observed_entities),
            "observed_but_unconfigured": sorted(observed_entities - configured_entities),
        },
        "relations": {
            "total": sum(relation_labels.values()),
            "by_label": dict(sorted(relation_labels.items())),
            "by_typed_pattern": dict(sorted(relation_patterns.items())),
            "per_task": distribution(relation_counts_per_task),
            "configured_but_unobserved": sorted(configured_relations - observed_relations),
            "observed_but_unconfigured": sorted(unconfigured_relations),
            "observed_but_unconfigured_label_count": unconfigured_relation_labels,
            "observed_but_unconfigured_strain_patterns": sorted(
                unconfigured_strain_relations
            ),
            "observed_but_unconfigured_strain_label_count": sum(
                relation_patterns[pattern] for pattern in unconfigured_strain_relations
            ),
            "configured_support": configured_relation_support,
            "configured_with_fewer_than_50_positive_tasks": low_support_relations,
        },
        "integrity": {
            "invalid_spans": invalid_spans,
            "span_text_mismatches": span_text_mismatches,
            "multi_label_spans": multi_label_spans,
            "duplicate_spans": duplicate_spans,
            "overlapping_span_pairs": overlapping_span_pairs,
            "same_entity_type_overlap_pairs": same_entity_type_overlap_pairs,
            "cross_entity_type_overlap_pairs": cross_entity_type_overlap_pairs,
            "missing_relation_endpoints": missing_relation_endpoints,
            "self_relations": self_relations,
            "unlabeled_relations": unlabeled_relations,
            "multi_label_relations": multi_label_relations,
            "duplicate_relations": duplicate_relations,
            "unknown_result_types": dict(sorted(unknown_result_types.items())),
        },
        "issue_samples": dict(issue_samples),
        "_task_signatures": {str(task.get("id")): annotation_signature(task) for task in tasks},
    }


def compare_exports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_signatures = left["_task_signatures"]
    right_signatures = right["_task_signatures"]
    left_ids = set(left_signatures)
    right_ids = set(right_signatures)
    shared = left_ids & right_ids
    changed = sorted(task_id for task_id in shared if left_signatures[task_id] != right_signatures[task_id])
    signature_changes: Counter[str] = Counter()
    changed_task_types: Counter[str] = Counter()
    for task_id in changed:
        removed = set(left_signatures[task_id]) - set(right_signatures[task_id])
        added = set(right_signatures[task_id]) - set(left_signatures[task_id])
        for signature in added:
            signature_changes[f"{signature[0]}_added"] += 1
        for signature in removed:
            signature_changes[f"{signature[0]}_removed"] += 1
        for result_type in ("labels", "relation"):
            if any(signature[0] == result_type for signature in added | removed):
                changed_task_types[f"tasks_with_{result_type}_changes"] += 1
    return {
        "older": left["path"],
        "newer": right["path"],
        "added_task_ids": sorted(right_ids - left_ids),
        "removed_task_ids": sorted(left_ids - right_ids),
        "changed_task_count": len(changed),
        "changed_task_ids_sample": changed[:50],
        "signature_changes": dict(sorted(signature_changes.items())),
        "changed_task_types": dict(sorted(changed_task_types.items())),
    }


def render_markdown(report: dict[str, Any]) -> str:
    latest = report["exports"][-1]
    annotations = latest["annotations"]
    entities = latest["entities"]
    relations = latest["relations"]
    integrity = latest["integrity"]
    unconfigured_relation_count = relations[
        "observed_but_unconfigured_label_count"
    ]
    unconfigured_relation_fraction = (
        100 * unconfigured_relation_count / relations["total"]
        if relations["total"]
        else 0.0
    )
    lines = [
        "# Ground-truth annotation audit",
        "",
        f"Latest export: `{latest['path']}` (`{latest['sha256'][:12]}…`)",
        "",
        "## Summary",
        "",
        f"- Tasks: {latest['tasks']:,}; raw active annotation records: {annotations['active_records']:,}; canonical annotations: {annotations['canonical']:,}",
        f"- Canonical annotators: {annotations['annotators']}; selection policy: `{annotations['selection_policy']}`",
        f"- Entity spans: {entities['total']:,}; relation labels: {relations['total']:,}",
        f"- Exact duplicate texts: {latest['duplicate_text_groups']:,}; tasks with multiple annotations: {annotations['tasks_with_multiple']:,}",
        f"- Tasks annotated independently by multiple annotators: {annotations['tasks_with_multiple_annotators']:,}",
        f"- Document provenance available: {latest['provenance']['has_document_identifier']} (data fields: {latest['provenance']['data_fields']}; metadata fields: {latest['provenance']['metadata_fields']})",
        "",
        "## Entity counts",
        "",
        "| Entity | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {count:,} |" for label, count in entities["by_label"].items())
    lines.extend(
        [
            "",
            "## Annotator profiles",
            "",
            "The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.",
            "",
            "| Annotator | Tasks | Empty | Entities/task | Relations/task |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {annotator} | {profile['tasks']:,} | {profile['empty_tasks']:,} | {profile['entities_per_task']:.2f} | {profile['relations_per_task']:.2f} |"
        for annotator, profile in annotations["annotator_profiles"].items()
    )
    annotator_ids = list(annotations["annotator_profiles"])
    entity_profile_labels = sorted(
        {
            label
            for profile in annotations["annotator_profiles"].values()
            for label in profile["entity_labels_per_100_tasks"]
        }
    )
    relation_profile_labels = sorted(
        {
            label
            for profile in annotations["annotator_profiles"].values()
            for label in profile["relation_labels_per_100_tasks"]
        }
    )
    lines.extend(
        [
            "",
            "### Entity labels per 100 tasks",
            "",
            "| Label | " + " | ".join(f"Annotator {value}" for value in annotator_ids) + " |",
            "|---|" + "---:|" * len(annotator_ids),
        ]
    )
    for label in entity_profile_labels:
        values = [
            annotations["annotator_profiles"][annotator][
                "entity_labels_per_100_tasks"
            ].get(label, 0.0)
            for annotator in annotator_ids
        ]
        lines.append(f"| {label} | " + " | ".join(f"{value:.1f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "### Relation labels per 100 tasks",
            "",
            "| Label | " + " | ".join(f"Annotator {value}" for value in annotator_ids) + " |",
            "|---|" + "---:|" * len(annotator_ids),
        ]
    )
    for label in relation_profile_labels:
        values = [
            annotations["annotator_profiles"][annotator][
                "relation_labels_per_100_tasks"
            ].get(label, 0.0)
            for annotator in annotator_ids
        ]
        lines.append(f"| {label} | " + " | ".join(f"{value:.1f}" for value in values) + " |")
    lines.extend(["", "## Typed relation counts", "", "| Relation | Count |", "|---|---:|"])
    lines.extend(
        f"| {label} | {count:,} |" for label, count in relations["by_typed_pattern"].items()
    )
    lines.extend(
        [
            "",
            "## Configured relation training support",
            "",
            "| Classifier | Positive labels | Positive tasks |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {label} | {support['positive_labels']:,} | {support['positive_tasks']:,} |"
        for label, support in relations["configured_support"].items()
    )
    lines.extend(
        [
            "",
            f"Configured classifiers with fewer than 50 positive tasks: {relations['configured_with_fewer_than_50_positive_tasks'] or 'none'}",
            "",
            "## Configuration drift",
            "",
            f"- Configured entities without examples: {entities['configured_but_unobserved'] or 'none'}",
            f"- Observed entities absent from config: {entities['observed_but_unconfigured'] or 'none'}",
            f"- Configured relations without positive examples: {relations['configured_but_unobserved'] or 'none'}",
            f"- Observed typed relations absent from config: {len(relations['observed_but_unconfigured']):,} patterns / {unconfigured_relation_count:,} labels ({unconfigured_relation_fraction:.1f}%); complete list is retained in the JSON audit",
            f"- Of those, STRAIN-involving omissions: {len(relations['observed_but_unconfigured_strain_patterns']):,} patterns / {relations['observed_but_unconfigured_strain_label_count']:,} labels",
            "",
            "## Ground-truth actions before retraining",
            "",
            "- Keep PMCID, article version, paragraph and sentence identifiers in every new annotation task.",
            "- Independently double-annotate a stratified subset and adjudicate it before reporting inter-annotator agreement.",
            "- Collect or merge evidence for configured relation classifiers with fewer than 50 positive tasks; do not interpret their current holdout scores as stable.",
            "- Adjudicate observed typed relations that are outside the configured model taxonomy instead of silently converting them to negatives.",
            "",
            "## Integrity checks",
            "",
        ]
    )
    lines.extend(f"- {key.replace('_', ' ')}: {value}" for key, value in integrity.items())
    for comparison in report.get("comparisons", []):
        lines.extend(
            [
                "",
                "## Export comparison",
                "",
                f"- Older: `{comparison['older']}`",
                f"- Newer: `{comparison['newer']}`",
                f"- Added tasks: {len(comparison['added_task_ids']):,}",
                f"- Removed tasks: {len(comparison['removed_task_ids']):,}",
                f"- Tasks with changed annotations: {comparison['changed_task_count']:,}",
                f"- Changed-task categories: {comparison['changed_task_types']}",
                f"- Signature changes: {comparison['signature_changes']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    configured_entities = set(config.get("ner_labels", []))
    configured_relations = set(config.get("rel_labels", []))
    exports = [
        audit_export(path, configured_entities, configured_relations)
        for path in args.exports
    ]
    report = {
        "config": str(args.config),
        "exports": exports,
        "comparisons": [
            compare_exports(left, right) for left, right in zip(exports, exports[1:])
        ],
    }
    for export in exports:
        export.pop("_task_signatures", None)

    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_markdown(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered_markdown)
    if not args.json_output and not args.markdown_output:
        print(rendered_json, end="")


if __name__ == "__main__":
    main()
