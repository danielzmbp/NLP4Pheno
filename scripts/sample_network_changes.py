#!/usr/bin/env python3
"""Extract auditable prediction examples for the largest network changes."""

from __future__ import annotations

import argparse
import json
import re
from functools import reduce
from pathlib import Path
from typing import Any

import polars as pl


def load_edges(
    comparison_path: Path, change_key: str, limit: int
) -> list[dict[str, Any]]:
    report = json.loads(comparison_path.read_text())
    return report["comparison"]["ranked_changes"][change_key][:limit]


def edge_expression(edges: list[dict[str, Any]]) -> pl.Expr:
    expressions = []
    for edge in edges:
        strain = (
            edge["source"]
            if edge["source"].startswith("SI-ID")
            else edge["target"]
        )
        entity = edge["target"] if edge["source"].startswith("SI-ID") else edge["source"]
        match = re.fullmatch(r"SI-ID(\d+)", strain)
        if match is None:
            continue
        expressions.append(
            (pl.col("straininfo_si_id") == int(match.group(1)))
            & (pl.col("word_qc_group") == entity)
            & pl.col("rel").str.ends_with(f":{edge['relation']}")
        )
    if not expressions:
        return pl.lit(False)
    return reduce(lambda left, right: left | right, expressions)


def extract_examples(
    predictions_path: Path,
    edges: list[dict[str, Any]],
    change_kind: str,
) -> pl.DataFrame:
    metadata = {
        (edge["source"], edge["target"], edge["relation"]): edge for edge in edges
    }
    selected = (
        pl.scan_parquet(predictions_path)
        .filter(edge_expression(edges))
        .select(
            "rel",
            "score_rel",
            "ner_score",
            "score_strain",
            "word_strain",
            "word_strain_qc",
            "straininfo_si_id",
            "straininfo_taxon",
            "straininfo_method",
            "word",
            "word_qc_group",
            "ner",
            "text",
            "pmcid",
            "article_version",
            "paragraph",
            "sentence_range",
        )
        .collect(engine="streaming")
    )
    records = []
    for row in selected.iter_rows(named=True):
        strain = f"SI-ID{row['straininfo_si_id']}"
        relation = row["rel"].rsplit(":", 1)[-1]
        if row["rel"].startswith("STRAIN"):
            edge = (strain, row["word_qc_group"], relation)
        else:
            edge = (row["word_qc_group"], strain, relation)
        edge_metadata = metadata.get(edge)
        if edge_metadata is None:
            continue
        records.append(
            {
                "change_kind": change_kind,
                "source": edge[0],
                "target": edge[1],
                "relation": relation,
                "old_evidence_rows": edge_metadata["old_evidence_rows"],
                "new_evidence_rows": edge_metadata["new_evidence_rows"],
                **row,
            }
        )
    if not records:
        return pl.DataFrame()
    return (
        pl.DataFrame(records)
        .sort(
            ["source", "target", "relation", "score_rel", "ner_score", "score_strain"],
            descending=[False, False, False, True, True, True],
        )
        .unique(subset=["source", "target", "relation"], keep="first")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--old-predictions", required=True, type=Path)
    parser.add_argument("--new-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    additions = load_edges(args.comparison, "largest_new_edges", args.limit)
    removals = load_edges(args.comparison, "largest_removed_edges", args.limit)
    frames = [
        extract_examples(args.new_predictions, additions, "new_edge"),
        extract_examples(args.old_predictions, removals, "removed_edge"),
    ]
    frames = [frame for frame in frames if not frame.is_empty()]
    result = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    result.write_csv(temporary, separator="\t")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
