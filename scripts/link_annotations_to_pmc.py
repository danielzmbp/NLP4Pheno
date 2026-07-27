#!/usr/bin/env python3
"""Recover PMC provenance for Label Studio tasks by literal corpus matching.

The historical Label Studio export contains only task text.  This script does
not guess a source article: it records every exact literal occurrence of that
text in the current PMC corpus and labels tasks with multiple candidate PMCIDs
as ambiguous.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import polars as pl

from annotation_utils import load_annotations


PROVENANCE_COLUMNS = [
    "pmcid",
    "article_version",
    "section",
    "paragraph",
    "sentence_range",
    "text",
]


def normalize_text(value: str | None) -> str:
    """Apply only lossless display normalization used by the PMC parser."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def annotation_frame(tasks: list[dict[str, Any]]) -> pl.DataFrame:
    records = [
        {
            "task_id": str(task.get("id")),
            "annotation_text": normalize_text(task.get("data", {}).get("text")),
        }
        for task in tasks
    ]
    frame = pl.DataFrame(records, schema={"task_id": pl.String, "annotation_text": pl.String})
    empty = frame.filter(pl.col("annotation_text") == "")
    if empty.height:
        raise ValueError(f"{empty.height} annotation tasks have empty text")
    return frame


def match_annotations(
    tasks: list[dict[str, Any]],
    corpus_paths: list[str | Path],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return all literal annotation occurrences and a task-level summary."""
    annotations = annotation_frame(tasks)
    patterns = annotations.get_column("annotation_text").unique().implode()

    corpus = pl.scan_parquet([str(path) for path in corpus_paths])
    missing = set(PROVENANCE_COLUMNS) - set(corpus.collect_schema().names())
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    occurrences = (
        corpus.select(
            *[pl.col(column) for column in PROVENANCE_COLUMNS[:-1]],
            pl.col("text").str.replace_all(r"\s+", " ").str.strip_chars().alias("corpus_text"),
        )
        .with_columns(
            pl.col("corpus_text").str.extract_many(patterns).alias("annotation_text")
        )
        .filter(pl.col("annotation_text").list.len() > 0)
        .explode("annotation_text", empty_as_null=False)
        .join(annotations.lazy(), on="annotation_text", how="inner")
        .with_columns(
            pl.when(pl.col("corpus_text") == pl.col("annotation_text"))
            .then(pl.lit("row_exact"))
            .otherwise(pl.lit("contained_exact"))
            .alias("match_kind")
        )
        .group_by(
            "task_id",
            "annotation_text",
            "pmcid",
            "article_version",
            "section",
            "paragraph",
            "sentence_range",
            "match_kind",
        )
        .agg(pl.len().alias("occurrences"))
        # The PMC corpus is tens of gigabytes. The default in-memory engine can
        # materialize the projected text column and exceed 64 GB even though
        # the final literal-match table is small. Force Polars' streaming
        # engine so Parquet row groups are matched and aggregated in bounded
        # batches.
        .collect(engine="streaming")
    )

    if occurrences.height:
        task_counts = occurrences.group_by("task_id").agg(
            pl.col("pmcid").n_unique().alias("candidate_pmcids"),
            pl.col("occurrences").sum().alias("task_occurrences"),
        )
        occurrences = (
            occurrences.join(task_counts, on="task_id", how="left")
            .with_columns(
                pl.when(pl.col("candidate_pmcids") == 1)
                .then(pl.lit("unique_pmcid"))
                .otherwise(pl.lit("ambiguous_pmcid"))
                .alias("status")
            )
            .sort("task_id", "pmcid", "paragraph", "sentence_range")
        )
        matched_tasks = task_counts.height
        unique_tasks = task_counts.filter(pl.col("candidate_pmcids") == 1).height
        ambiguous_tasks = task_counts.filter(pl.col("candidate_pmcids") > 1).height
        unique_pmcids = occurrences.get_column("pmcid").n_unique()
        occurrence_count = int(occurrences.get_column("occurrences").sum())
    else:
        matched_tasks = unique_tasks = ambiguous_tasks = unique_pmcids = occurrence_count = 0

    total_tasks = annotations.height
    summary = {
        "matching_policy": "NFKC + whitespace normalization; exact literal row or substring only",
        "tasks": {
            "total": total_tasks,
            "matched": matched_tasks,
            "unmatched": total_tasks - matched_tasks,
            "unique_pmcid": unique_tasks,
            "ambiguous_pmcid": ambiguous_tasks,
        },
        "matches": {
            "rows": occurrences.height,
            "literal_occurrences": occurrence_count,
            "unique_pmcids": unique_pmcids,
        },
    }
    return occurrences, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("corpus", nargs="+", type=Path)
    parser.add_argument("--matches-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches, summary = match_annotations(load_annotations(args.annotations), args.corpus)
    args.matches_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    matches.write_parquet(args.matches_output, compression="zstd")
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
