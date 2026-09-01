#!/usr/bin/env python3
"""Build a pre-annotated review queue from edges that changed between runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from annotation_utils import load_annotations
from build_prediction_review_queue import build_results, stable_id, write_tsv


DEFAULT_RELATIONS = {
    "STRAIN-ORGANISM:INFECTS",
    "STRAIN-ORGANISM:INHABITS",
}
PREDICTION_COLUMNS = [
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
    "start",
    "end",
    "start_strain",
    "end_strain",
    "ontology_status",
    "ontology",
    "ontology_id",
    "ontology_label",
]
EDGE_COLUMNS = ["_source", "_target", "_relation"]


def with_edge_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    columns = set(frame.collect_schema().names())
    entity_node = (
        pl.coalesce("ontology_node_id", "word_qc_group")
        if "ontology_node_id" in columns
        else pl.col("word_qc_group")
    )
    return frame.with_columns(
        pl.concat_str(
            pl.lit("SI-ID"),
            pl.col("straininfo_si_id").cast(pl.Int64).cast(pl.String),
        ).alias("_strain_id"),
        entity_node.alias("_entity_id"),
        pl.col("rel").str.split(":").list.get(1).alias("_relation"),
    ).with_columns(
        pl.when(pl.col("rel").str.starts_with("STRAIN-"))
        .then(pl.col("_strain_id"))
        .otherwise(pl.col("_entity_id"))
        .alias("_source"),
        pl.when(pl.col("rel").str.starts_with("STRAIN-"))
        .then(pl.col("_entity_id"))
        .otherwise(pl.col("_strain_id"))
        .alias("_target"),
    )


def changed_edges(
    old_predictions: Path,
    new_predictions: Path,
    relations: set[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    def keys(path: Path) -> pl.LazyFrame:
        return (
            with_edge_columns(pl.scan_parquet(path))
            .filter(
                pl.col("rel").is_in(sorted(relations))
                & pl.col("straininfo_si_id").is_not_null()
                & pl.col("_entity_id").is_not_null()
            )
            .select(EDGE_COLUMNS)
            .unique()
        )

    old = keys(old_predictions)
    new = keys(new_predictions)
    old_only = old.join(new, on=EDGE_COLUMNS, how="anti").collect(engine="streaming")
    new_only = new.join(old, on=EDGE_COLUMNS, how="anti").collect(engine="streaming")
    return old_only, new_only


def _candidate_rows(
    predictions: Path,
    edges: pl.DataFrame,
    direction: str,
    relations: set[str],
    max_text_chars: int,
) -> pl.DataFrame:
    if edges.is_empty():
        return pl.DataFrame()
    source = with_edge_columns(pl.scan_parquet(predictions))
    available = set(source.collect_schema().names())
    selected_columns = [name for name in PREDICTION_COLUMNS if name in available]
    return (
        source.filter(pl.col("rel").is_in(sorted(relations)))
        .join(edges.lazy(), on=EDGE_COLUMNS, how="inner")
        .filter(
            pl.col("text").is_not_null()
            & pl.col("pmcid").is_not_null()
            & (pl.col("text").str.len_chars() <= max_text_chars)
        )
        .select(*selected_columns, *EDGE_COLUMNS)
        .with_columns(
            pl.lit(direction).alias("review_direction"),
            pl.min_horizontal("score_rel", "ner_score", "score_strain").alias(
                "_minimum_score"
            ),
        )
        .unique(
            subset=[
                "text",
                "rel",
                "start_strain",
                "end_strain",
                "start",
                "end",
            ]
        )
        .collect(engine="streaming")
    )


def select_disagreements(
    old_predictions: Path,
    new_predictions: Path,
    existing_texts: set[str],
    *,
    relations: set[str],
    per_relation_direction: int,
    max_text_chars: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    old_only, new_only = changed_edges(old_predictions, new_predictions, relations)
    candidates = pl.concat(
        [
            _candidate_rows(
                old_predictions,
                old_only,
                "old_only",
                relations,
                max_text_chars,
            ),
            _candidate_rows(
                new_predictions,
                new_only,
                "new_only",
                relations,
                max_text_chars,
            ),
        ],
        how="diagonal_relaxed",
    )
    if candidates.is_empty():
        return candidates, {
            "old_only_edges": old_only.height,
            "new_only_edges": new_only.height,
        }
    if existing_texts:
        candidates = candidates.filter(~pl.col("text").is_in(existing_texts))
    candidates = candidates.sort(
        [
            "review_direction",
            "rel",
            "score_rel",
            "_minimum_score",
            "pmcid",
            "text",
        ],
        descending=[False, False, True, True, False, False],
    )

    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for row in candidates.iter_rows(named=True):
        group = (str(row["review_direction"]), str(row["rel"]))
        text = str(row["text"])
        if counts[group] >= per_relation_direction or text in seen_texts:
            continue
        selected.append(row)
        seen_texts.add(text)
        counts[group] += 1
    result = pl.DataFrame(selected, schema=candidates.schema) if selected else candidates.head(0)
    return result, {
        "old_only_edges": old_only.height,
        "new_only_edges": new_only.height,
    }


def build_queue(selected: pl.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for rank, row in enumerate(selected.iter_rows(named=True), start=1):
        results, counts = build_results(str(row["text"]), [], [row])
        candidate_id = stable_id(
            row["review_direction"],
            row["pmcid"],
            row["sentence_range"],
            row["text"],
            row["rel"],
            prefix="network_disagreement",
        )
        direction_explanation = (
            "present after retraining, absent before"
            if row["review_direction"] == "new_only"
            else "present before retraining, absent after"
        )
        review_summary = (
            f"{row['review_direction']} ({direction_explanation}) | {row['rel']} | "
            f"RE {float(row['score_rel']):.3f} | "
            f"entity {float(row['ner_score']):.3f} | "
            f"strain {float(row['score_strain']):.3f}"
        )
        task = {
            "data": {
                "text": row["text"],
                "candidate_id": candidate_id,
                "review_rank": rank,
                "review_tier": row["review_direction"],
                "review_summary": review_summary,
                "sampled_relation": row["rel"],
                "pmcid": row["pmcid"],
                "article_version": row["article_version"],
                "paragraph": row["paragraph"],
                "sentence_range": row["sentence_range"],
                "ontology_status": row.get("ontology_status"),
                "ontology": row.get("ontology"),
                "ontology_id": row.get("ontology_id"),
                "ontology_label": row.get("ontology_label"),
                "straininfo_si_id": row["straininfo_si_id"],
                "straininfo_taxon": row.get("straininfo_taxon"),
            },
            "meta": {
                "source": "network_run_disagreement",
                "candidate_id": candidate_id,
                "review_direction": row["review_direction"],
                "seeded_entities": counts["entities"],
                "seeded_relations": counts["relations"],
            },
            "predictions": [
                {
                    "model_version": f"NLP4Pheno-{row['review_direction']}",
                    "score": float(row["score_rel"]),
                    "result": results,
                }
            ],
        }
        tasks.append(task)
        issue = dict(row)
        issue.update({"candidate_id": candidate_id, "review_rank": rank, **counts})
        issues.append(issue)
    return tasks, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_predictions", type=Path)
    parser.add_argument("new_predictions", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--queue-output", required=True, type=Path)
    parser.add_argument("--issues-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--per-relation-direction", type=int, default=10)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--include-relation", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relations = set(args.include_relation) or DEFAULT_RELATIONS
    existing = {
        str(task.get("data", {}).get("text") or "")
        for task in load_annotations(args.annotations)
    }
    selected, edge_counts = select_disagreements(
        args.old_predictions,
        args.new_predictions,
        existing,
        relations=relations,
        per_relation_direction=args.per_relation_direction,
        max_text_chars=args.max_text_chars,
    )
    tasks, issues = build_queue(selected)
    args.queue_output.parent.mkdir(parents=True, exist_ok=True)
    args.queue_output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_tsv(args.issues_output, issues)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "old_predictions": str(args.old_predictions),
        "new_predictions": str(args.new_predictions),
        "annotations": str(args.annotations),
        "relations": sorted(relations),
        "per_relation_direction": args.per_relation_direction,
        "max_text_chars": args.max_text_chars,
        "tasks": len(tasks),
        "task_counts": dict(
            sorted(
                Counter(
                    f"{task['data']['review_tier']}:{task['data']['sampled_relation']}"
                    for task in tasks
                ).items()
            )
        ),
        **edge_counts,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
