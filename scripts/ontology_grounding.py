"""Conservative ontology grounding helpers for NLP4Pheno entity mentions."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from annotation_utils import load_annotations


HYPHENS = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})
GREEK = {
    "α": " alpha ",
    "β": " beta ",
    "γ": " gamma ",
    "δ": " delta ",
    "ε": " epsilon ",
    "κ": " kappa ",
    "λ": " lambda ",
    "μ": " mu ",
}


def normalize_surface(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").translate(HYPHENS).casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n.,;:")


def relaxed_surface(value: str) -> str:
    value = normalize_surface(value)
    for symbol, replacement in GREEK.items():
        value = value.replace(symbol, replacement)
    # PMC XML conversion can leave TeX-like subscript separators in formulas.
    value = re.sub(r"(?<=[a-z])\s*_\s*(?=\d)", "", value)
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_formula_surface(value: str) -> str:
    """Remove PMC/XML formula formatting while preserving element case."""
    value = unicodedata.normalize("NFKC", value or "").translate(HYPHENS)
    value = re.sub(r"[\s_]+", "", value)
    value = value.strip("[]")
    value = re.sub(r"\^(\d+)([+-])$", r"(\1\2)", value)
    value = re.sub(r"\s*:\s*", ":", value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_long_form(short_form: str, candidate: str) -> str | None:
    short = re.sub(r"[^A-Za-z0-9]", "", short_form)
    if len(short) < 2:
        return None
    short_index = len(short) - 1
    long_index = len(candidate) - 1
    while short_index >= 0:
        character = short[short_index].casefold()
        while long_index >= 0 and candidate[long_index].casefold() != character:
            long_index -= 1
        if long_index < 0:
            return None
        if short_index == 0:
            while long_index > 0 and candidate[long_index - 1].isalnum():
                long_index -= 1
            if candidate[long_index].casefold() != character:
                return None
        long_index -= 1
        short_index -= 1
    return candidate[long_index + 1 :].strip(" \t,;:-")


def extract_abbreviation_definitions(text: str) -> dict[str, str]:
    """Extract local `long form (SF)` and `SF (long form)` definitions."""
    definitions: dict[str, str] = {}
    for match in re.finditer(r"\(([^()]{2,100})\)", text):
        inside = match.group(1).strip()
        before = text[: match.start()].rstrip()
        previous = re.search(r"([A-Za-z][A-Za-z0-9+/-]{1,14})\s*$", before)

        compact_inside = re.sub(r"[^A-Za-z0-9]", "", inside)
        if " " not in inside and 2 <= len(compact_inside) <= 15:
            word_limit = min(len(compact_inside) + 5, len(compact_inside) * 2)
            candidate = " ".join(before.split()[-word_limit:])
            long_form = _best_long_form(inside, candidate)
            if long_form:
                definitions[normalize_surface(inside)] = long_form
        elif previous:
            short_form = previous.group(1)
            compact_short = re.sub(r"[^A-Za-z0-9]", "", short_form)
            if 2 <= len(compact_short) <= 15:
                long_form = _best_long_form(short_form, inside)
                if long_form:
                    definitions[normalize_surface(short_form)] = long_form
    return definitions


def extract_mentions(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for task in tasks:
        text = str(task.get("data", {}).get("text") or "")
        abbreviations = extract_abbreviation_definitions(text)
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") != "labels":
                    continue
                value = result.get("value", {})
                labels = value.get("labels") or []
                if not labels:
                    continue
                start = int(value.get("start") or 0)
                end = int(value.get("end") or 0)
                surface = str(value.get("text") or text[start:end])
                mentions.append(
                    {
                        "task_id": str(task.get("id", "")),
                        "annotation_id": str(annotation.get("id", "")),
                        "span_id": str(result.get("id", "")),
                        "entity_type": str(labels[0]),
                        "surface": surface,
                        "normalized_surface": normalize_surface(surface),
                        "relaxed_surface": relaxed_surface(surface),
                        "formula_surface": normalize_formula_surface(surface),
                        "expanded_form": abbreviations.get(normalize_surface(surface)),
                        "start": start,
                        "end": end,
                    }
                )
    return mentions


@dataclass(frozen=True)
class Candidate:
    ontology: str
    concept_id: str
    concept_label: str
    alias: str
    scope: str


def _candidate_key(candidate: Candidate) -> tuple[str, str]:
    return candidate.ontology, candidate.concept_id


def _deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    scope_order = {"LABEL": 0, "EXACT": 1, "BROAD": 2, "NARROW": 3, "RELATED": 4}
    best: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        # Taxonomy root is a technical container, not a biologically useful
        # entity.  Its synonym "all" otherwise creates a unique but spurious
        # exact match for ordinary prose.
        if candidate.ontology == "NCBITAXON" and candidate.concept_id == "NCBITaxon:1":
            continue
        key = _candidate_key(candidate)
        previous = best.get(key)
        if previous is None or scope_order.get(candidate.scope, 99) < scope_order.get(
            previous.scope, 99
        ):
            best[key] = candidate
    return sorted(best.values(), key=lambda item: (item.ontology, item.concept_id))


def load_candidate_indexes(
    aliases_path: Path, mentions: list[dict[str, Any]]
) -> tuple[
    dict[tuple[str, str], list[Candidate]],
    dict[tuple[str, str], list[Candidate]],
    dict[tuple[str, str], list[Candidate]],
]:
    entity_types = sorted({mention["entity_type"] for mention in mentions})
    normalized_values = {
        value
        for mention in mentions
        for value in (
            mention["normalized_surface"],
            normalize_surface(mention["expanded_form"] or ""),
        )
        if value
    }
    relaxed_values = {
        value
        for mention in mentions
        for value in (
            mention["relaxed_surface"],
            relaxed_surface(mention["expanded_form"] or ""),
        )
        if len(value) >= 4
    }
    formula_values = {
        mention["formula_surface"]
        for mention in mentions
        if mention["entity_type"] == "COMPOUND" and mention["formula_surface"]
    }

    dataset = ds.dataset(aliases_path, format="parquet")
    alias_expressions = []
    if normalized_values:
        alias_expressions.append(
            ds.field("normalized_alias").isin(sorted(normalized_values))
        )
    if relaxed_values:
        alias_expressions.append(
            ds.field("relaxed_alias").isin(sorted(relaxed_values))
        )
    if formula_values:
        alias_expressions.append(
            ds.field("formula_alias").isin(sorted(formula_values))
        )
    if not alias_expressions:
        return {}, {}, {}
    alias_expression = alias_expressions[0]
    for additional_expression in alias_expressions[1:]:
        alias_expression = alias_expression | additional_expression
    expression = (
        ds.field("auto_eligible")
        & ds.field("entity_type").isin(entity_types)
        & alias_expression
    )
    table = dataset.to_table(
        columns=[
            "entity_type",
            "ontology",
            "concept_id",
            "concept_label",
            "alias",
            "normalized_alias",
            "relaxed_alias",
            "formula_alias",
            "scope",
        ],
        filter=expression,
    )
    exact: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    relaxed: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    formula: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for row in table.to_pylist():
        candidate = Candidate(
            ontology=row["ontology"],
            concept_id=row["concept_id"],
            concept_label=row["concept_label"],
            alias=row["alias"],
            scope=row["scope"],
        )
        if row["scope"] != "RELATED":
            exact[(row["entity_type"], row["normalized_alias"])].append(candidate)
            if len(row["relaxed_alias"]) >= 4:
                relaxed[(row["entity_type"], row["relaxed_alias"])].append(candidate)
        if row["formula_alias"]:
            formula[(row["entity_type"], row["formula_alias"])].append(candidate)
    return exact, relaxed, formula


def ground_mentions(
    mentions: list[dict[str, Any]],
    exact_index: dict[tuple[str, str], list[Candidate]],
    relaxed_index: dict[tuple[str, str], list[Candidate]],
    formula_index: dict[tuple[str, str], list[Candidate]] | None = None,
    supported_entity_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    grounded: list[dict[str, Any]] = []
    for mention in mentions:
        entity_type = mention["entity_type"]
        if (
            supported_entity_types is not None
            and entity_type not in supported_entity_types
        ):
            row = dict(mention)
            row.update(
                {
                    "status": "unsupported",
                    "candidate_count": 0,
                    "candidates_json": "[]",
                    "match_method": None,
                    "ontology": None,
                    "concept_id": None,
                    "concept_label": None,
                    "matched_alias": None,
                    "alias_scope": None,
                }
            )
            grounded.append(row)
            continue
        attempts = [
            ("direct", mention["normalized_surface"], mention["relaxed_surface"]),
        ]
        if mention["expanded_form"]:
            attempts.insert(
                0,
                (
                    "abbreviation",
                    normalize_surface(mention["expanded_form"]),
                    relaxed_surface(mention["expanded_form"]),
                ),
            )

        candidates: list[Candidate] = []
        method = None
        if formula_index and mention.get("formula_surface"):
            candidates = _deduplicate_candidates(
                formula_index.get(
                    (entity_type, mention["formula_surface"]), []
                )
            )
            if candidates:
                method = "direct_formula"
        for prefix, exact_key, relaxed_key in attempts:
            if candidates:
                break
            candidates = _deduplicate_candidates(
                exact_index.get((entity_type, exact_key), [])
            )
            if candidates:
                method = f"{prefix}_exact"
                break
            if len(relaxed_key) >= 4:
                candidates = _deduplicate_candidates(
                    relaxed_index.get((entity_type, relaxed_key), [])
                )
                if candidates:
                    method = f"{prefix}_normalized"
                    break

        row = dict(mention)
        row["candidate_count"] = len(candidates)
        row["candidates_json"] = json.dumps(
            [
                {
                    "ontology": candidate.ontology,
                    "concept_id": candidate.concept_id,
                    "concept_label": candidate.concept_label,
                    "alias": candidate.alias,
                    "scope": candidate.scope,
                }
                for candidate in candidates
            ],
            ensure_ascii=False,
        )
        if len(candidates) == 1:
            candidate = candidates[0]
            row.update(
                {
                    "status": "matched",
                    "match_method": method,
                    "ontology": candidate.ontology,
                    "concept_id": candidate.concept_id,
                    "concept_label": candidate.concept_label,
                    "matched_alias": candidate.alias,
                    "alias_scope": candidate.scope,
                }
            )
        elif candidates:
            row.update(
                {
                    "status": "ambiguous",
                    "match_method": method,
                    "ontology": None,
                    "concept_id": None,
                    "concept_label": None,
                    "matched_alias": None,
                    "alias_scope": None,
                }
            )
        else:
            row.update(
                {
                    "status": "unmatched",
                    "match_method": None,
                    "ontology": None,
                    "concept_id": None,
                    "concept_label": None,
                    "matched_alias": None,
                    "alias_scope": None,
                }
            )
        grounded.append(row)
    return grounded


def summarize_groundings(
    rows: list[dict[str, Any]], ontology_manifest: dict[str, Any]
) -> dict[str, Any]:
    supported = sorted(
        {
            entity_type
            for metadata in ontology_manifest["sources"].values()
            for entity_type in metadata["entity_types"]
        }
    )
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supported_entity_types": supported,
        "ontology_manifest": ontology_manifest,
        "entity_types": {},
    }
    for entity_type in sorted({row["entity_type"] for row in rows}):
        selected = [row for row in rows if row["entity_type"] == entity_type]
        counts = Counter(row["status"] for row in selected)
        total = len(selected)
        surfaces: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            surfaces[row["normalized_surface"]].append(row)
        unique_matched = sum(
            1
            for values in surfaces.values()
            if len(
                {
                    value["concept_id"]
                    for value in values
                    if value["status"] == "matched"
                }
            )
            == 1
            and all(value["status"] == "matched" for value in values)
        )
        unmatched = Counter(
            row["surface"] for row in selected if row["status"] == "unmatched"
        )
        ambiguous = Counter(
            row["surface"] for row in selected if row["status"] == "ambiguous"
        )
        matched_by_ontology = Counter(
            row["ontology"] for row in selected if row["status"] == "matched"
        )
        matched_by_method = Counter(
            row["match_method"] for row in selected if row["status"] == "matched"
        )
        summary["entity_types"][entity_type] = {
            "mentions": total,
            "matched": counts["matched"],
            "ambiguous": counts["ambiguous"],
            "unmatched": counts["unmatched"],
            "unsupported": counts["unsupported"],
            "coverage": round(counts["matched"] / total, 4) if total else 0.0,
            "unique_surfaces": len(surfaces),
            "uniquely_consistent_surfaces_matched": unique_matched,
            "matched_by_ontology": dict(matched_by_ontology),
            "matched_by_method": dict(matched_by_method),
            "top_unmatched": unmatched.most_common(20),
            "top_ambiguous": ambiguous.most_common(20),
        }
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Ontology grounding pilot",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "Only unique exact label/synonym matches and conservative normalized matches "
        "are accepted. No fuzzy or embedding match is counted as grounded.",
        "",
        "This is a coverage benchmark against annotated mention strings, not a "
        "concept-ID accuracy gold standard. Ambiguous candidates require review.",
        "",
        f"Annotation source: `{summary['annotation_source']['path']}` "
        f"(SHA-256 `{summary['annotation_source']['sha256'][:12]}…`).",
        "",
        "## Coverage",
        "",
        "| Entity | Mentions | Matched | Ambiguous | Unmatched | Coverage | Unique surfaces |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entity_type, values in summary["entity_types"].items():
        if entity_type not in summary["supported_entity_types"]:
            continue
        lines.append(
            f"| {entity_type} | {values['mentions']:,} | {values['matched']:,} | "
            f"{values['ambiguous']:,} | {values['unmatched']:,} | "
            f"{values['coverage']:.1%} | {values['unique_surfaces']:,} |"
        )

    lines.extend(["", "## Ontology snapshots", ""])
    for ontology, values in summary["ontology_manifest"]["sources"].items():
        version = values.get("ontology_header", {}).get("data-version") or "not declared"
        lines.append(
            f"- **{ontology}**: {values['terms']:,} terms; "
            f"{values['aliases']:,} labels/synonyms; version `{version}`; "
            f"SHA-256 `{values['sha256'][:12]}…`"
        )

    for entity_type, values in summary["entity_types"].items():
        if entity_type not in summary["supported_entity_types"]:
            continue
        sources = ", ".join(
            f"{name} {count:,}"
            for name, count in sorted(values["matched_by_ontology"].items())
        )
        methods = ", ".join(
            f"{name} {count:,}"
            for name, count in sorted(values["matched_by_method"].items())
        )
        lines.extend(
            [
                "",
                f"## {entity_type}: accepted-match breakdown",
                "",
                f"- Ontologies: {sources or 'none'}",
                f"- Methods: {methods or 'none'}",
            ]
        )
        if values["top_ambiguous"]:
            lines.extend(
                [
                    "",
                    "Frequent ambiguous mentions:",
                    "",
                    "| Surface | Count |",
                    "|---|---:|",
                ]
            )
            for surface, count in values["top_ambiguous"]:
                escaped_surface = surface.replace("|", r"\|")
                lines.append(f"| {escaped_surface} | {count} |")
        lines.extend(["", f"## {entity_type}: frequent unmatched mentions", ""])
        if not values["top_unmatched"]:
            lines.append("_None_")
        else:
            lines.append("| Surface | Count |")
            lines.append("|---|---:|")
            for surface, count in values["top_unmatched"]:
                escaped_surface = surface.replace("|", r"\|")
                lines.append(f"| {escaped_surface} | {count} |")
    lines.append("")
    return "\n".join(lines)


def run_annotation_pilot(
    annotations_path: Path,
    aliases_path: Path,
    manifest_path: Path,
    output_path: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    tasks = load_annotations(annotations_path)
    mentions = extract_mentions(tasks)
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    supported = {
        entity_type
        for metadata in manifest["sources"].values()
        for entity_type in metadata["entity_types"]
    }
    exact, relaxed, formula = load_candidate_indexes(aliases_path, mentions)
    rows = ground_mentions(mentions, exact, relaxed, formula, supported)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa_table := _rows_to_table(rows), output_path, compression="zstd")
    del pa_table
    summary = summarize_groundings(rows, manifest)
    summary["annotation_source"] = {
        "path": str(annotations_path),
        "sha256": sha256_file(annotations_path),
        "tasks": len(tasks),
        "mentions": len(mentions),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    report_path.write_text(render_report(summary), encoding="utf-8")
    return summary


def _rows_to_table(rows: list[dict[str, Any]]):
    import pyarrow as pa

    return pa.Table.from_pylist(rows)
