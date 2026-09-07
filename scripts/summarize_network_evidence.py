#!/usr/bin/env python3
"""Summarize network evidence and emit a conservative high-trust edge view."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl


PROVENANCE_COLUMNS = [
    "pmcid",
    "article_version",
    "paragraph",
    "sentence_range",
    "text",
]
SCORE_COLUMNS = ["score_rel", "ner_score", "score_strain"]
BASE_EDGE_COLUMNS = ["source", "target", "rel", "source_ner", "target_ner"]


def summarize_network_evidence(
    evidence_file: Path,
    summary_file: Path,
    core_file: Path,
    *,
    min_pmcs: int = 2,
    min_relation_score: float = 0.90,
    min_entity_score: float = 0.90,
    min_strain_score: float = 0.95,
) -> None:
    source = pl.scan_csv(evidence_file, separator="\t")
    columns = source.collect_schema().names()
    missing = set(BASE_EDGE_COLUMNS + PROVENANCE_COLUMNS + SCORE_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"Network evidence is missing columns: {sorted(missing)}")

    evidence_columns = set(PROVENANCE_COLUMNS + SCORE_COLUMNS)
    edge_columns = [name for name in columns if name not in evidence_columns]
    high_confidence = (
        (pl.col("score_rel") >= min_relation_score)
        & (pl.col("ner_score") >= min_entity_score)
        & (pl.col("score_strain") >= min_strain_score)
    ).fill_null(False)
    summary = (
        source.group_by(*edge_columns)
        .agg(
            pl.struct(PROVENANCE_COLUMNS).n_unique().alias("evidence_count"),
            pl.col("pmcid").drop_nulls().n_unique().alias("pmcid_count"),
            high_confidence.cast(pl.UInt32)
            .sum()
            .alias("high_confidence_evidence_count"),
            pl.col("score_rel").max().alias("max_relation_score"),
            pl.col("ner_score").max().alias("max_entity_score"),
            pl.col("score_strain").max().alias("max_strain_score"),
        )
        .with_columns(
            pl.when(pl.col("pmcid_count") >= min_pmcs)
            .then(pl.lit("multi_article"))
            .when(pl.col("high_confidence_evidence_count") > 0)
            .then(pl.lit("high_confidence_single_article"))
            .otherwise(pl.lit("supporting_evidence"))
            .alias("evidence_tier")
        )
        .sort(BASE_EDGE_COLUMNS)
    )
    core = summary.filter(
        (pl.col("pmcid_count") >= min_pmcs)
        | (pl.col("high_confidence_evidence_count") > 0)
    )

    for frame, destination in ((summary, summary_file), (core, core_file)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.sink_csv(temporary, separator="\t", engine="streaming")
        os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--core-output", required=True, type=Path)
    parser.add_argument("--min-pmcs", type=int, default=2)
    parser.add_argument("--min-relation-score", type=float, default=0.90)
    parser.add_argument("--min-entity-score", type=float, default=0.90)
    parser.add_argument("--min-strain-score", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_network_evidence(
        args.evidence,
        args.summary_output,
        args.core_output,
        min_pmcs=args.min_pmcs,
        min_relation_score=args.min_relation_score,
        min_entity_score=args.min_entity_score,
        min_strain_score=args.min_strain_score,
    )


if __name__ == "__main__":
    main()
