#!/usr/bin/env python3
"""Mine a balanced, fully pre-annotated Label Studio queue from PMC predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from annotation_utils import load_annotations


TIER_ORDER = {
    "high_grounded": 0,
    "high_unresolved": 1,
    "relation_boundary": 2,
    "entity_tension": 3,
}


def stable_id(*values: Any, prefix: str) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode()
    return f"{prefix}_{hashlib.sha1(payload).hexdigest()[:14]}"


def assign_tier() -> pl.Expr:
    high_entities = (pl.col("ner_score") >= 0.90) & (
        pl.col("score_strain") >= 0.90
    )
    return (
        pl.when(
            (pl.col("score_rel") >= 0.95)
            & high_entities
            & (pl.col("ontology_status") == "matched")
        )
        .then(pl.lit("high_grounded"))
        .when(
            (pl.col("score_rel") >= 0.90)
            & high_entities
            & pl.col("ontology_status").is_in(
                ["ambiguous", "unmatched", "unsupported"]
            )
        )
        .then(pl.lit("high_unresolved"))
        .when(
            pl.col("score_rel").is_between(0.50, 0.67, closed="both")
            & high_entities
        )
        .then(pl.lit("relation_boundary"))
        .when(
            (pl.col("score_rel") >= 0.85)
            & (
                (pl.col("ner_score") < 0.82)
                | (pl.col("score_strain") < 0.82)
            )
        )
        .then(pl.lit("entity_tension"))
        .otherwise(None)
    )


def select_candidate_sentences(
    predictions_file: Path,
    existing_texts: set[str],
    *,
    per_relation_tier: int,
    seed: int,
    max_text_chars: int,
    include_relations: set[str] | None = None,
) -> pl.DataFrame:
    columns = [
        "text",
        "pmcid",
        "article_version",
        "paragraph",
        "sentence_range",
        "rel",
        "score_rel",
        "ner_score",
        "score_strain",
        "ner",
        "word",
        "word_qc_group",
        "word_strain",
        "word_strain_qc",
        "straininfo_si_id",
        "straininfo_taxon",
        "straininfo_status",
        "straininfo_method",
        "start",
        "end",
        "start_strain",
        "end_strain",
        "ontology_status",
        "ontology_candidate_count",
        "ontology",
        "ontology_id",
        "ontology_label",
        "ontology_match_method",
        "ontology_match_confidence",
    ]
    source = pl.scan_parquet(predictions_file)
    if include_relations:
        source = source.filter(pl.col("rel").is_in(sorted(include_relations)))
    candidates = (
        source
        .select(columns)
        .filter(
            pl.col("pmcid").is_not_null()
            & pl.col("straininfo_si_id").is_not_null()
            & pl.col("text").is_not_null()
            & (pl.col("text").str.len_chars() <= max_text_chars)
        )
        .with_columns(assign_tier().alias("review_tier"))
        .filter(pl.col("review_tier").is_not_null())
        .with_columns(
            pl.concat_str(
                "text",
                "rel",
                "start_strain",
                "end_strain",
                "start",
                "end",
                separator="|",
            )
            .hash(seed=seed)
            .alias("_sample_hash")
        )
        .unique(
            subset=[
                "text",
                "rel",
                "start_strain",
                "end_strain",
                "start",
                "end",
                "review_tier",
            ],
            keep="first",
        )
        .sort(["rel", "review_tier", "_sample_hash"])
        .group_by("rel", "review_tier", maintain_order=True)
        .head(per_relation_tier * 4)
        .collect(engine="streaming")
    )
    if existing_texts:
        candidates = candidates.filter(~pl.col("text").is_in(existing_texts))

    # Identical sentences occasionally occur in multiple PMC articles. They
    # cannot be assigned a unique source-document split, so omit them.
    candidate_texts = candidates.get_column("text").unique().to_list()
    provenance = (
        pl.scan_parquet(predictions_file)
        .filter(pl.col("text").is_in(candidate_texts))
        .group_by("text")
        .agg(pl.col("pmcid").drop_nulls().unique().alias("_pmcids"))
        .collect(engine="streaming")
    )
    candidates = (
        candidates.join(provenance, on="text", how="left")
        .filter(pl.col("_pmcids").list.len() == 1)
        .drop("_pmcids")
        .sort(["rel", "review_tier", "_sample_hash"])
        .group_by("rel", "review_tier", maintain_order=True)
        .head(per_relation_tier)
        .with_columns(
            pl.col("review_tier")
            .replace_strict(TIER_ORDER)
            .alias("_tier_order")
        )
        .sort(["_tier_order", "rel", "_sample_hash"])
        .drop("_tier_order")
    )

    # A sentence is one review unit even when several sampled edges selected it.
    return candidates.unique(subset=["text"], keep="first", maintain_order=True)


def collect_sentence_predictions(
    selected: pl.DataFrame,
    ner_file: Path,
    relations_file: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    texts = selected.get_column("text").to_list()
    ner = (
        pl.scan_parquet(ner_file)
        .filter(pl.col("text").is_in(texts))
        .select(
            "text",
            "start_strain",
            "end_strain",
            "word_strain",
            "score_strain",
            "start",
            "end",
            "word",
            "ner",
            pl.col("score").alias("ner_score"),
        )
        .collect(engine="streaming")
    )
    relation_columns = [
        "text",
        "start_strain",
        "end_strain",
        "word_strain",
        "score_strain",
        "start",
        "end",
        "word",
        "ner",
        "ner_score",
        "rel",
        "score_rel",
        "word_qc_group",
        "straininfo_si_id",
        "straininfo_taxon",
        "ontology_status",
        "ontology",
        "ontology_id",
        "ontology_label",
    ]
    relations = (
        pl.scan_parquet(relations_file)
        .filter(pl.col("text").is_in(texts))
        .select(relation_columns)
        .unique()
        .collect(engine="streaming")
    )
    sampled_keys = {
        (
            row["text"],
            int(row["start_strain"]),
            int(row["end_strain"]),
            int(row["start"]),
            int(row["end"]),
            row["rel"],
        )
        for row in selected.iter_rows(named=True)
    }
    relation_records = [
        row
        for row in relations.iter_rows(named=True)
        if float(row["score_rel"]) >= 0.90
        or (
            row["text"],
            int(row["start_strain"]),
            int(row["end_strain"]),
            int(row["start"]),
            int(row["end"]),
            row["rel"],
        )
        in sampled_keys
    ]
    relations = (
        pl.DataFrame(relation_records, schema=relations.schema)
        if relation_records
        else relations.head(0)
    )
    # Seed auxiliary NER suggestions only when both sides of the candidate pair
    # are reasonably strong. Lower-scoring endpoints of sampled relations are
    # added separately by build_results.
    ner = ner.filter(
        (pl.col("ner_score") >= 0.85) & (pl.col("score_strain") >= 0.85)
    )
    return ner, relations


def _valid_span(text: str, start: Any, end: Any) -> tuple[int, int] | None:
    try:
        start_value = int(start)
        end_value = int(end)
    except (TypeError, ValueError):
        return None
    if start_value < 0 or end_value <= start_value or end_value > len(text):
        return None
    return start_value, end_value


def build_results(
    text: str,
    ner_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entities: dict[tuple[int, int, str], dict[str, Any]] = {}

    def add_entity(
        start: Any,
        end: Any,
        label: str,
        score: Any,
        fallback_surface: Any,
    ) -> str | None:
        span = _valid_span(text, start, end)
        if span is None:
            return None
        start_value, end_value = span
        surface = text[start_value:end_value]
        if not surface.strip():
            surface = str(fallback_surface or "")
        key = (start_value, end_value, label)
        identifier = stable_id(*key, prefix="pred_entity")
        previous = entities.get(key)
        score_value = float(score or 0.0)
        if previous is None or score_value > previous["_score"]:
            entities[key] = {
                "id": identifier,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start_value,
                    "end": end_value,
                    "text": surface,
                    "labels": [label],
                },
                "_score": score_value,
            }
        return identifier

    for row in ner_rows:
        add_entity(
            row["start_strain"],
            row["end_strain"],
            "STRAIN",
            row["score_strain"],
            row["word_strain"],
        )
        add_entity(
            row["start"],
            row["end"],
            str(row["ner"]),
            row["ner_score"],
            row["word"],
        )
    # Guarantee that every proposed relation endpoint exists, even if the NER
    # candidate table was filtered differently.
    for row in relation_rows:
        add_entity(
            row["start_strain"],
            row["end_strain"],
            "STRAIN",
            row["score_strain"],
            row["word_strain"],
        )
        add_entity(
            row["start"],
            row["end"],
            str(row["ner"]),
            row["ner_score"],
            row["word"],
        )

    results = [
        {key: value for key, value in entity.items() if key != "_score"}
        for entity in sorted(
            entities.values(),
            key=lambda value: (
                value["value"]["start"],
                value["value"]["end"],
                value["value"]["labels"][0],
            ),
        )
    ]
    entity_ids = {item["id"] for item in results}
    relation_seen: set[tuple[str, str, str]] = set()
    skipped_relations = 0
    for row in sorted(
        relation_rows,
        key=lambda value: (
            value["start_strain"],
            value["start"],
            value["rel"],
        ),
    ):
        strain_id = stable_id(
            int(row["start_strain"]),
            int(row["end_strain"]),
            "STRAIN",
            prefix="pred_entity",
        )
        entity_id = stable_id(
            int(row["start"]),
            int(row["end"]),
            str(row["ner"]),
            prefix="pred_entity",
        )
        pair, relation_name = str(row["rel"]).split(":", 1)
        if pair.startswith("STRAIN-"):
            source_id, target_id = strain_id, entity_id
        else:
            source_id, target_id = entity_id, strain_id
        relation_key = (source_id, target_id, relation_name)
        if relation_key in relation_seen:
            continue
        if source_id not in entity_ids or target_id not in entity_ids:
            skipped_relations += 1
            continue
        results.append(
            {
                "from_id": source_id,
                "to_id": target_id,
                "type": "relation",
                "direction": "right",
                "labels": [relation_name],
            }
        )
        relation_seen.add(relation_key)
    return results, {
        "entities": len(entities),
        "relations": len(relation_seen),
        "skipped_relations": skipped_relations,
    }


def build_queue(
    selected: pl.DataFrame,
    ner: pl.DataFrame,
    relations: pl.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ner_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ner.iter_rows(named=True):
        ner_by_text[row["text"]].append(row)
    relation_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relations.iter_rows(named=True):
        relation_by_text[row["text"]].append(row)

    tasks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for rank, selected_row in enumerate(selected.iter_rows(named=True), start=1):
        text = selected_row["text"]
        prediction_results, result_counts = build_results(
            text,
            ner_by_text[text],
            relation_by_text[text],
        )
        candidate_id = stable_id(
            selected_row["pmcid"],
            selected_row["sentence_range"],
            text,
            prefix="pmc_candidate",
        )
        review_summary = (
            f"{selected_row['review_tier']} | {selected_row['rel']} | "
            f"RE {selected_row['score_rel']:.3f} | "
            f"entity {selected_row['ner_score']:.3f} | "
            f"strain {selected_row['score_strain']:.3f} | "
            f"ontology {selected_row['ontology_status']}"
        )
        tasks.append(
            {
                "data": {
                    "text": text,
                    "candidate_id": candidate_id,
                    "review_rank": rank,
                    "review_tier": selected_row["review_tier"],
                    "review_summary": review_summary,
                    "sampled_relation": selected_row["rel"],
                    "pmcid": selected_row["pmcid"],
                    "article_version": selected_row["article_version"],
                    "paragraph": selected_row["paragraph"],
                    "sentence_range": selected_row["sentence_range"],
                    "ontology_status": selected_row["ontology_status"],
                    "ontology": selected_row["ontology"],
                    "ontology_id": selected_row["ontology_id"],
                    "ontology_label": selected_row["ontology_label"],
                    "straininfo_si_id": selected_row["straininfo_si_id"],
                    "straininfo_taxon": selected_row["straininfo_taxon"],
                },
                "meta": {
                    "source": "full_pmc_model_prediction",
                    "candidate_id": candidate_id,
                    "review_tier": selected_row["review_tier"],
                    "seeded_entities": result_counts["entities"],
                    "seeded_relations": result_counts["relations"],
                },
                "predictions": [
                    {
                        "model_version": "NLP4Pheno-2509-ontology-review",
                        "score": float(selected_row["score_rel"]),
                        "result": prediction_results,
                    }
                ],
            }
        )
        issue_row = dict(selected_row)
        issue_row.update(
            {
                "candidate_id": candidate_id,
                "review_rank": rank,
                **{
                    f"seeded_{key}": value
                    for key, value in result_counts.items()
                },
            }
        )
        issues.append(issue_row)
    return tasks, issues


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relations", type=Path)
    parser.add_argument("ner", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--queue-output", required=True, type=Path)
    parser.add_argument("--issues-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--per-relation-tier", type=int, default=3)
    parser.add_argument(
        "--include-relation",
        action="append",
        default=[],
        help="Restrict sampling to this typed relation; repeat as needed",
    )
    parser.add_argument("--seed", type=int, default=2509)
    parser.add_argument("--max-text-chars", type=int, default=450)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    existing_texts = {
        str(task.get("data", {}).get("text") or "") for task in annotations
    }
    selected = select_candidate_sentences(
        args.relations,
        existing_texts,
        per_relation_tier=args.per_relation_tier,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
        include_relations=set(args.include_relation) or None,
    )
    ner, relations = collect_sentence_predictions(
        selected,
        args.ner,
        args.relations,
    )
    tasks, issues = build_queue(selected, ner, relations)

    args.queue_output.parent.mkdir(parents=True, exist_ok=True)
    args.queue_output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_tsv(args.issues_output, issues)
    tier_counts = Counter(row["data"]["review_tier"] for row in tasks)
    relation_counts = Counter(row["data"]["sampled_relation"] for row in tasks)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "relations": str(args.relations),
        "ner": str(args.ner),
        "annotations": str(args.annotations),
        "seed": args.seed,
        "per_relation_tier": args.per_relation_tier,
        "include_relations": sorted(set(args.include_relation)),
        "max_text_chars": args.max_text_chars,
        "tasks": len(tasks),
        "tier_counts": dict(sorted(tier_counts.items())),
        "sampled_relation_counts": dict(sorted(relation_counts.items())),
        "seeded_entities": sum(
            int(task["meta"]["seeded_entities"]) for task in tasks
        ),
        "seeded_relations": sum(
            int(task["meta"]["seeded_relations"]) for task in tasks
        ),
        "selection_policy": {
            "high_grounded": "RE >= .95, both NER scores >= .90, unique ontology match",
            "high_unresolved": "RE >= .90, both NER scores >= .90, ontology unresolved",
            "relation_boundary": ".50 <= RE <= .67, both NER scores >= .90",
            "entity_tension": "RE >= .85 and at least one NER score < .82",
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
