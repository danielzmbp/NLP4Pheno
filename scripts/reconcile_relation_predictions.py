#!/usr/bin/env python3
"""Resolve competing entity types and impossible same-taxon relations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl


GUARDED_SAME_TAXON_RELATIONS = {"INFECTS", "INHABITS", "SYMBIONT_OF"}
TAXON_ENTITY_TYPES = {"ORGANISM", "SPECIES"}
EDGE_KEY = [
    "straininfo_si_id",
    "word_qc_group",
    "_relation_name",
    "_strain_is_source",
]


def _normalized_tokens(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
        .str.split(" ")
    )


def same_taxon_expression() -> pl.Expr:
    """Identify exact or abbreviated binomial equality at relation endpoints."""
    entity_tokens = _normalized_tokens("word_qc_group")
    taxon_tokens = _normalized_tokens("straininfo_taxon")
    entity_genus = entity_tokens.list.get(0, null_on_oob=True)
    entity_species = entity_tokens.list.get(1, null_on_oob=True)
    taxon_genus = taxon_tokens.list.get(0, null_on_oob=True)
    taxon_species = taxon_tokens.list.get(1, null_on_oob=True)
    genus_matches = (entity_genus == taxon_genus) | (
        (entity_genus.str.len_chars() == 1)
        & taxon_genus.str.starts_with(entity_genus)
    )
    return (
        (pl.col("word_qc_group").is_not_null())
        & (pl.col("straininfo_taxon").is_not_null())
        & genus_matches.fill_null(False)
        & (entity_species == taxon_species).fill_null(False)
    )


def _prepared(source: pl.LazyFrame) -> pl.LazyFrame:
    return source.with_columns(
        pl.col("rel").str.split(":").list.get(1).alias("_relation_name"),
        pl.col("rel").str.starts_with("STRAIN").alias("_strain_is_source"),
        (
            pl.col("ner_score").fill_null(0.0)
            * pl.col("score_rel").fill_null(0.0)
        ).alias("_joint_score"),
    )


def reconcile_lazy(
    source: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Return reconciled rows and per-edge candidate-type statistics."""
    prepared = _prepared(source)
    matched = prepared.filter(pl.col("straininfo_si_id").is_not_null())
    unmatched = prepared.filter(pl.col("straininfo_si_id").is_null())

    candidates = matched.group_by(*EDGE_KEY, "rel").agg(
        pl.col("_joint_score").mean().alias("_mean_joint_score"),
        pl.col("_joint_score").max().alias("_max_joint_score"),
        pl.len().alias("_support_rows"),
    )
    choices = (
        candidates.sort(
            [
                *EDGE_KEY,
                "_mean_joint_score",
                "_support_rows",
                "_max_joint_score",
                "rel",
            ],
            descending=[False] * len(EDGE_KEY) + [True, True, True, False],
        )
        .unique(subset=EDGE_KEY, keep="first", maintain_order=True)
        .select(*EDGE_KEY, "rel")
    )
    selected_before_guard = matched.join(
        choices,
        on=[*EDGE_KEY, "rel"],
        how="semi",
    )
    impossible_same_taxon = (
        pl.col("ner").is_in(TAXON_ENTITY_TYPES)
        & pl.col("_relation_name").is_in(GUARDED_SAME_TAXON_RELATIONS)
        & same_taxon_expression()
    )
    selected = selected_before_guard.filter(~impossible_same_taxon)
    reconciled = pl.concat([unmatched, selected], how="vertical").drop(
        "_relation_name",
        "_strain_is_source",
        "_joint_score",
    )
    return reconciled, candidates, selected_before_guard


def reconcile_relation_predictions(
    input_file: Path,
    output_file: Path,
    summary_file: Path,
) -> dict:
    source = pl.scan_parquet(input_file)
    required = {
        "ner",
        "ner_score",
        "rel",
        "score_rel",
        "straininfo_si_id",
        "straininfo_taxon",
        "word_qc_group",
    }
    missing = required - set(source.collect_schema().names())
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    reconciled, candidates, selected_before_guard = reconcile_lazy(source)
    input_counts = _prepared(source).select(
        pl.len().alias("input_rows"),
        pl.col("straininfo_si_id").is_not_null().sum().alias("matched_input_rows"),
    ).collect(engine="streaming").row(0, named=True)
    conflicting_edges = (
        candidates.group_by(*EDGE_KEY)
        .agg(pl.col("rel").n_unique().alias("_candidate_types"))
        .select(
            (pl.col("_candidate_types") > 1).sum().alias("conflicting_edges")
        )
        .collect(engine="streaming")
        .item()
    )
    matched_selected_rows = selected_before_guard.select(pl.len()).collect(
        engine="streaming"
    ).item()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    reconciled.sink_parquet(
        temporary,
        compression="snappy",
        engine="streaming",
    )
    os.replace(temporary, output_file)

    output_counts = pl.scan_parquet(output_file).select(
        pl.len().alias("output_rows"),
        pl.col("straininfo_si_id").is_not_null().sum().alias("matched_output_rows"),
    ).collect(engine="streaming").row(0, named=True)
    summary = {
        **{key: int(value) for key, value in input_counts.items()},
        "conflicting_edges": int(conflicting_edges),
        "matched_selected_rows_before_same_taxon_guard": int(
            matched_selected_rows
        ),
        **{key: int(value) for key, value in output_counts.items()},
    }
    summary["removed_rows"] = summary["input_rows"] - summary["output_rows"]
    summary["removed_matched_rows"] = (
        summary["matched_input_rows"] - summary["matched_output_rows"]
    )
    summary["removed_competing_type_rows"] = (
        summary["matched_input_rows"]
        - summary["matched_selected_rows_before_same_taxon_guard"]
    )
    summary["removed_same_taxon_rows"] = (
        summary["matched_selected_rows_before_same_taxon_guard"]
        - summary["matched_output_rows"]
    )

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_file.with_suffix(summary_file.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    os.replace(summary_temporary, summary_file)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = reconcile_relation_predictions(
        args.input,
        args.output,
        args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
