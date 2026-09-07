#!/usr/bin/env python3
"""Find ontology-backed entity mentions missing from reviewed annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.dataset as ds

from annotation_utils import load_annotations
from build_annotation_review_queue import (
    apply_suggestions,
    sha256,
    task_entities,
    write_issue_tsv,
)
from ontology_grounding import (
    extract_abbreviation_definitions,
    normalize_formula_surface,
    normalize_surface,
)


TOKEN_RE = re.compile(r"[^\W_](?:[\w'’/+\-‐‑–—]*[^\W_])?", re.UNICODE)
GENERIC_SURFACES = {
    "acid",
    "activity",
    "agar",
    "carbon",
    "cell",
    "cells",
    "compound",
    "culture",
    "cultures",
    "disease",
    "growth",
    "infection",
    "medium",
    "organism",
    "organisms",
    "production",
    "resistance",
    "sample",
    "samples",
    "species",
    "strain",
    "strains",
}


@dataclass(frozen=True)
class AliasCandidate:
    entity_type: str
    ontology: str
    concept_id: str
    concept_label: str
    alias: str
    scope: str
    mode: str
    case_sensitive: bool


def iter_ngram_spans(
    text: str, *, max_tokens: int = 6
) -> Iterator[tuple[int, int, str]]:
    tokens = list(TOKEN_RE.finditer(text))
    for start_index, first in enumerate(tokens):
        for end_index in range(start_index, min(len(tokens), start_index + max_tokens)):
            last = tokens[end_index]
            yield first.start(), last.end(), text[first.start() : last.end()]


def possible_lookup_keys(
    tasks: list[dict[str, Any]], *, max_tokens: int
) -> tuple[set[str], set[str]]:
    normalized: set[str] = set()
    formulas: set[str] = set()
    for task in tasks:
        text = str(task.get("data", {}).get("text") or "")
        for _start, _end, surface in iter_ngram_spans(
            text, max_tokens=max_tokens
        ):
            key = normalize_surface(surface)
            if key:
                normalized.add(key)
            formula = normalize_formula_surface(surface)
            if formula:
                formulas.add(formula)
    return normalized, formulas


def _requires_case(alias: str) -> bool:
    letters = "".join(character for character in alias if character.isalpha())
    compact = re.sub(r"[^A-Za-z0-9]", "", alias)
    return (
        2 <= len(letters) <= 8
        and len(compact) <= 10
        and letters == letters.upper()
    )


def _choose_alias(rows: list[AliasCandidate]) -> AliasCandidate:
    scope_order = {"LABEL": 0, "EXACT": 1, "RELATED": 2}
    return min(
        rows,
        key=lambda row: (
            scope_order.get(row.scope, 9),
            len(row.alias),
            row.alias,
        ),
    )


def load_unique_aliases(
    aliases_path: Path,
    normalized_keys: set[str],
    formula_keys: set[str],
) -> tuple[dict[str, AliasCandidate], dict[str, AliasCandidate], dict[str, int]]:
    dataset = ds.dataset(aliases_path, format="parquet")
    expression = ds.field("auto_eligible") & (
        ds.field("normalized_alias").isin(sorted(normalized_keys))
        | ds.field("formula_alias").isin(sorted(formula_keys))
    )
    scanner = dataset.scanner(
        columns=[
            "entity_type",
            "ontology",
            "concept_id",
            "concept_label",
            "alias",
            "normalized_alias",
            "formula_alias",
            "scope",
        ],
        filter=expression,
    )
    grouped: dict[tuple[str, str], dict[tuple[str, str, str], list[AliasCandidate]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    rows_scanned = 0
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            rows_scanned += 1
            if row["scope"] != "RELATED" and row["normalized_alias"]:
                candidate = AliasCandidate(
                    entity_type=row["entity_type"],
                    ontology=row["ontology"],
                    concept_id=row["concept_id"],
                    concept_label=row["concept_label"],
                    alias=row["alias"],
                    scope=row["scope"],
                    mode="normalized",
                    case_sensitive=_requires_case(row["alias"]),
                )
                identity = (
                    candidate.entity_type,
                    candidate.ontology,
                    candidate.concept_id,
                )
                grouped[("normalized", row["normalized_alias"])][identity].append(
                    candidate
                )
            if row["formula_alias"]:
                candidate = AliasCandidate(
                    entity_type=row["entity_type"],
                    ontology=row["ontology"],
                    concept_id=row["concept_id"],
                    concept_label=row["concept_label"],
                    alias=row["alias"],
                    scope=row["scope"],
                    mode="formula",
                    case_sensitive=True,
                )
                identity = (
                    candidate.entity_type,
                    candidate.ontology,
                    candidate.concept_id,
                )
                grouped[("formula", row["formula_alias"])][identity].append(candidate)

    normalized: dict[str, AliasCandidate] = {}
    formulas: dict[str, AliasCandidate] = {}
    ambiguous = 0
    for (mode, key), concepts in grouped.items():
        if len(concepts) != 1:
            ambiguous += 1
            continue
        candidate = _choose_alias(next(iter(concepts.values())))
        if mode == "normalized":
            normalized[key] = candidate
        else:
            formulas[key] = candidate
    return normalized, formulas, {
        "alias_rows_scanned": rows_scanned,
        "unique_normalized_aliases": len(normalized),
        "unique_formula_aliases": len(formulas),
        "ambiguous_alias_keys_excluded": ambiguous,
    }


def annotated_surface_evidence(
    tasks: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        for entity in task_entities(task).values():
            evidence[normalize_surface(entity.surface)][entity.label] += 1
    return evidence


def _same_case(surface: str, alias: str) -> bool:
    left = re.sub(r"[^A-Za-z0-9]", "", surface)
    right = re.sub(r"[^A-Za-z0-9]", "", alias)
    return left == right


def context_is_credible(
    text: str,
    surface: str,
    candidate: AliasCandidate,
    abbreviations: dict[str, str],
) -> bool:
    normalized = normalize_surface(surface)
    surface_letters = "".join(character for character in surface if character.isalpha())
    alias_letters = "".join(
        character for character in candidate.alias if character.isalpha()
    )
    if (
        len(surface_letters) >= 4
        and surface_letters == surface_letters.upper()
        and alias_letters != alias_letters.upper()
    ):
        return False
    expansion = abbreviations.get(normalized)
    if expansion and candidate.case_sensitive:
        expected = {
            normalize_surface(candidate.alias),
            normalize_surface(candidate.concept_label),
        }
        normalized_expansion = normalize_surface(expansion)
        if not any(
            normalized_expansion == value
            or normalized_expansion in value
            or value in normalized_expansion
            for value in expected
        ):
            return False

    lowered = normalize_surface(text)
    if candidate.entity_type == "ISOLATE" and normalized == "blood":
        source_patterns = (
            r"\bblood (?:isolates?|strains?|origin)\b",
            r"\b(?:isolated|recovered|obtained)[^.]{0,80}\bfrom (?:the )?blood\b",
            r"\bfrom (?:the )?blood\b",
            r"\bin (?:the )?blood\b",
            r"\bbloodstream\b",
            r"\bbacteremia\b",
        )
        if not any(re.search(pattern, lowered) for pattern in source_patterns):
            return False
    if candidate.entity_type == "ISOLATE" and normalized == "root nodule":
        if not re.search(
            r"\b(?:isolated|recovered|obtained|collected|associated|inhabit|from)\b",
            lowered,
        ):
            return False
    if candidate.entity_type == "PHENOTYPE" and normalized == "circular":
        if re.search(
            r"\b(?:chromosome|genome|plasmid|replicon|representation|dna|"
            r"dichroism|spectroscopy)\b",
            lowered,
        ):
            return False
    if candidate.entity_type == "COMPOUND" and normalized in {
        "amino acid",
        "amino acids",
    }:
        if re.search(
            r"\b(?:sequence|identity|substitution|replacement|residue|alignment)\w*\b",
            lowered,
        ):
            return False
    return True


def score_candidate(
    surface: str,
    candidate: AliasCandidate,
    evidence: dict[str, Counter[str]],
    *,
    has_strain: bool,
) -> tuple[float, str] | None:
    normalized = normalize_surface(surface)
    if normalized in GENERIC_SURFACES:
        return None
    if candidate.case_sensitive and not _same_case(surface, candidate.alias):
        return None

    counts = evidence.get(normalized, Counter())
    support = counts.get(candidate.entity_type, 0)
    conflicting_support = sum(counts.values()) - support
    token_count = len(list(TOKEN_RE.finditer(surface)))
    alpha_length = sum(character.isalpha() for character in surface)

    if conflicting_support:
        return None
    if support >= 2:
        score = min(0.99, 0.93 + min(support, 12) * 0.005)
        tier = "annotation_and_ontology"
    elif support == 1:
        score = 0.86
        tier = "single_annotation_and_ontology"
    elif candidate.mode == "formula":
        score = 0.90
        tier = "case_preserving_formula"
    elif token_count >= 2 and alpha_length >= 8:
        score = 0.89 if candidate.scope == "LABEL" else 0.87
        tier = "specific_multiword_ontology_alias"
    elif candidate.entity_type == "MEDIUM" and candidate.case_sensitive:
        score = 0.88
        tier = "resource_stated_medium_alias"
    elif (
        candidate.entity_type in {"COMPOUND", "DISEASE", "ISOLATE"}
        and alpha_length >= 8
        and candidate.scope == "LABEL"
    ):
        score = 0.84
        tier = "long_ontology_label"
    else:
        return None

    if has_strain:
        score = min(0.99, score + 0.01)
    return score, tier


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in occupied)


def find_omission_issues(
    tasks: list[dict[str, Any]],
    normalized_aliases: dict[str, AliasCandidate],
    formula_aliases: dict[str, AliasCandidate],
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    evidence = annotated_surface_evidence(tasks)
    issues: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id"))
        text = str(task.get("data", {}).get("text") or "")
        entities = list(task_entities(task).values())
        occupied = [(entity.start, entity.end) for entity in entities]
        has_strain = any(entity.label == "STRAIN" for entity in entities)
        abbreviations = extract_abbreviation_definitions(text)
        task_candidates: dict[tuple[int, int, str], dict[str, Any]] = {}
        for start, end, surface in iter_ngram_spans(text, max_tokens=max_tokens):
            if _overlaps(start, end, occupied):
                continue
            candidates = []
            normalized = normalize_surface(surface)
            if normalized in normalized_aliases:
                candidates.append(normalized_aliases[normalized])
            formula = normalize_formula_surface(surface)
            if formula in formula_aliases:
                candidates.append(formula_aliases[formula])
            for candidate in candidates:
                if not context_is_credible(
                    text, surface, candidate, abbreviations
                ):
                    continue
                scored = score_candidate(
                    surface, candidate, evidence, has_strain=has_strain
                )
                if scored is None:
                    continue
                score, tier = scored
                key = (start, end, candidate.entity_type)
                item = {
                    "category": "ontology_entity_omission",
                    "score": round(score, 6),
                    "task_id": task_id,
                    "summary": (
                        f"Review unannotated {surface!r} as {candidate.entity_type}; "
                        f"it uniquely matches {candidate.concept_id} "
                        f"({candidate.concept_label})."
                    ),
                    "action": {
                        "kind": "add_entity",
                        "start": start,
                        "end": end,
                        "text": surface,
                        "suggested_label": candidate.entity_type,
                    },
                    "evidence": {
                        "tier": tier,
                        "ontology": candidate.ontology,
                        "concept_id": candidate.concept_id,
                        "concept_label": candidate.concept_label,
                        "matched_alias": candidate.alias,
                        "alias_scope": candidate.scope,
                        "match_mode": candidate.mode,
                        "annotated_support": evidence.get(normalized, {}).get(
                            candidate.entity_type, 0
                        ),
                    },
                }
                previous = task_candidates.get(key)
                if previous is None or item["score"] > previous["score"]:
                    task_candidates[key] = item

        ranked = sorted(
            task_candidates.values(),
            key=lambda item: (
                -item["score"],
                -(item["action"]["end"] - item["action"]["start"]),
                item["action"]["start"],
            ),
        )
        selected: list[dict[str, Any]] = []
        selected_spans: list[tuple[int, int]] = []
        for item in ranked:
            action = item["action"]
            if _overlaps(action["start"], action["end"], selected_spans):
                continue
            selected.append(item)
            selected_spans.append((action["start"], action["end"]))
        issues.extend(selected)
    return sorted(
        issues,
        key=lambda item: (
            -item["score"],
            item["task_id"],
            item["action"]["start"],
        ),
    )


def build_queue(
    tasks: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    min_score: float,
    max_tasks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in issues:
        if item["score"] >= min_score:
            by_task[item["task_id"]].append(item)
    ranked_tasks = sorted(
        by_task,
        key=lambda task_id: (
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
        score = max(item["score"] for item in task_issues)
        queue.append(
            {
                "data": {
                    "text": task.get("data", {}).get("text", ""),
                    "original_task_id": task_id,
                    "review_rank": rank,
                    "review_score": score,
                    "review_categories": "ontology_entity_omission",
                    "review_summary": " | ".join(
                        item["summary"] for item in task_issues
                    ),
                },
                "meta": {
                    "original_task_id": task_id,
                    "review_rank": rank,
                    "review_score": score,
                    "review_issues": task_issues,
                },
                "predictions": [
                    {
                        "model_version": "ontology-omission-audit-v1",
                        "score": score,
                        "result": apply_suggestions(task, task_issues),
                    }
                ],
            }
        )
    return queue, selected_issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=Path("resources/ontologies/runtime"),
    )
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--issues-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.93)
    parser.add_argument("--max-tokens", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_annotations(args.annotations)
    normalized_keys, formula_keys = possible_lookup_keys(
        tasks, max_tokens=args.max_tokens
    )
    normalized_aliases, formula_aliases, alias_summary = load_unique_aliases(
        args.ontology_dir / "aliases.parquet",
        normalized_keys,
        formula_keys,
    )
    issues = find_omission_issues(
        tasks,
        normalized_aliases,
        formula_aliases,
        max_tokens=args.max_tokens,
    )
    queue, selected_issues = build_queue(
        tasks,
        issues,
        min_score=args.min_score,
        max_tasks=args.max_tasks,
    )
    issue_counts = Counter(item["evidence"]["tier"] for item in issues)
    selected_counts = Counter(item["evidence"]["tier"] for item in selected_issues)
    summary = {
        "source": str(args.annotations),
        "source_sha256": sha256(args.annotations),
        "ontology_manifest": str(args.ontology_dir / "manifest.json"),
        "tasks_in_source": len(tasks),
        "possible_normalized_keys": len(normalized_keys),
        "possible_formula_keys": len(formula_keys),
        **alias_summary,
        "candidate_issues": len(issues),
        "candidate_issues_by_tier": dict(sorted(issue_counts.items())),
        "selected_tasks": len(queue),
        "selected_issues": len(selected_issues),
        "selected_issues_by_tier": dict(sorted(selected_counts.items())),
        "min_score": args.min_score,
        "max_tasks": args.max_tasks,
        "max_tokens": args.max_tokens,
    }
    args.queue_output.parent.mkdir(parents=True, exist_ok=True)
    args.queue_output.write_text(
        json.dumps(queue, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_issue_tsv(args.issues_output, selected_issues)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
