#!/usr/bin/env python3
"""Audit ontology-based same-taxon relation removals against an unfiltered file."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import polars as pl

from ground_relation_ontology import (
    GUARDED_SAME_TAXON_RELATIONS,
    TAXON_ENTITY_TYPES,
)


PREFERRED_OUTPUT_COLUMNS = [
    "text",
    "pmcid",
    "article_version",
    "paragraph",
    "sentence_range",
    "rel",
    "straininfo_si_id",
    "straininfo_taxon",
    "word_strain_qc",
    "ner",
    "word_qc_group",
    "ontology_id",
    "ontology_label",
    "score_rel",
    "ner_score",
    "score_strain",
]


def audit_taxonomy_consistency(
    predictions_file: Path,
    strain_mapping_file: Path,
    rows_output: Path,
    summary_output: Path,
) -> dict:
    source = pl.scan_parquet(predictions_file)
    source_columns = source.collect_schema().names()
    required = {"ner", "rel", "straininfo_taxon", "ontology_status", "ontology_id"}
    missing = required - set(source_columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    strain_mapping = pl.scan_parquet(strain_mapping_file).select(
        pl.col("grouped_surface").alias("straininfo_taxon"),
        pl.col("ontology_status").alias("strain_taxonomy_status"),
        pl.col("ontology_id").alias("strain_taxonomy_id"),
        pl.col("ontology_label").alias("strain_taxonomy_label"),
    )
    same_taxon = (
        pl.col("ner").is_in(TAXON_ENTITY_TYPES)
        & pl.col("rel")
        .str.split(":")
        .list.get(1)
        .is_in(GUARDED_SAME_TAXON_RELATIONS)
        & (pl.col("ontology_status") == "matched")
        & (pl.col("strain_taxonomy_status") == "matched")
        & (pl.col("ontology_id") == pl.col("strain_taxonomy_id"))
    ).fill_null(False)
    selected_columns = [
        name for name in PREFERRED_OUTPUT_COLUMNS if name in source_columns
    ]
    rows = (
        source.join(strain_mapping, on="straininfo_taxon", how="left")
        .filter(same_taxon)
        .select(*selected_columns, "strain_taxonomy_id", "strain_taxonomy_label")
        .collect(engine="streaming")
        .sort(["rel", "straininfo_taxon", "word_qc_group"])
    )

    rows_output.parent.mkdir(parents=True, exist_ok=True)
    rows_temporary = rows_output.with_suffix(rows_output.suffix + ".tmp")
    rows.write_csv(rows_temporary, separator="\t")
    os.replace(rows_temporary, rows_output)

    pair_counts = (
        rows.group_by("straininfo_taxon", "word_qc_group", "ontology_id")
        .len(name="prediction_rows")
        .sort(
            ["prediction_rows", "straininfo_taxon", "word_qc_group"],
            descending=[True, False, False],
        )
        .head(50)
        .to_dicts()
    )
    summary = {
        "predictions": str(predictions_file),
        "strain_mapping": str(strain_mapping_file),
        "guarded_relations": sorted(GUARDED_SAME_TAXON_RELATIONS),
        "same_taxon_rows": rows.height,
        "rows_by_relation": dict(
            sorted(Counter(rows.get_column("rel").to_list()).items())
        ),
        "top_taxon_pairs": pair_counts,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_output.with_suffix(summary_output.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_temporary, summary_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("strain_mapping", type=Path)
    parser.add_argument("--rows-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_taxonomy_consistency(
        args.predictions,
        args.strain_mapping,
        args.rows_output,
        args.summary_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
