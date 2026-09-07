#!/usr/bin/env python3
"""Create the relation network and matched-strain manifest with Polars streaming."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl


def create_relation_network(
    predictions_file: Path,
    network_file: Path,
    strains_file: Path,
) -> None:
    source = pl.scan_parquet(predictions_file)
    source_columns = set(source.collect_schema().names())
    ontology_grounded_input = {
        "ontology_status",
        "ontology_node_id",
        "ontology_node_label",
    }.issubset(source_columns)
    matched = (
        source.filter(
            ~pl.col("word_strain_qc")
            .str.contains("adapted|covid")
            .fill_null(False)
        )
        .filter(pl.col("straininfo_si_id").is_not_null())
        .with_columns(
            pl.concat_str(
                pl.lit("SI-ID"),
                pl.col("straininfo_si_id").cast(pl.Int64).cast(pl.String),
            ).alias("strain_id")
        )
    )
    if ontology_grounded_input:
        matched = matched.with_columns(
            pl.coalesce("ontology_node_id", "word_qc_group").alias("entity_node_id"),
            pl.coalesce("ontology_node_label", "word_qc_group").alias(
                "entity_node_label"
            ),
        )
    else:
        matched = matched.with_columns(
            pl.col("word_qc_group").alias("entity_node_id"),
            pl.col("word_qc_group").alias("entity_node_label"),
            pl.lit("not_run").alias("ontology_status"),
            pl.lit(None, dtype=pl.String).alias("ontology"),
            pl.lit(None, dtype=pl.String).alias("ontology_id"),
            pl.lit(None, dtype=pl.String).alias("ontology_match_method"),
            pl.lit(None, dtype=pl.Float64).alias("ontology_match_confidence"),
            pl.lit(None, dtype=pl.String).alias("ontology_matched_alias"),
            pl.lit(None, dtype=pl.String).alias("ontology_alias_scope"),
        )

    network = (
        matched.select(
            "strain_id",
            "word_qc_group",
            "rel",
            "entity_node_id",
            "entity_node_label",
            "ontology_status",
            "ontology",
            "ontology_id",
            "ontology_match_method",
            "ontology_match_confidence",
            "ontology_matched_alias",
            "ontology_alias_scope",
        )
        .group_by(
            "strain_id",
            "rel",
            "entity_node_id",
            "entity_node_label",
            "ontology_status",
            "ontology",
            "ontology_id",
        )
        .agg(
            pl.col("word_qc_group").unique().sort().alias("_entity_surfaces"),
            pl.col("ontology_match_method")
            .drop_nulls()
            .unique()
            .sort()
            .alias("_ontology_methods"),
            pl.col("ontology_match_confidence")
            .max()
            .alias("ontology_match_confidence"),
            pl.col("ontology_matched_alias")
            .drop_nulls()
            .unique()
            .sort()
            .alias("_ontology_aliases"),
            pl.col("ontology_alias_scope")
            .drop_nulls()
            .unique()
            .sort()
            .alias("_ontology_scopes"),
        )
        .with_columns(
            pl.col("_entity_surfaces")
            .list.join(" | ")
            .alias("entity_surfaces"),
            pl.when(pl.col("_ontology_methods").list.len() > 0)
            .then(pl.col("_ontology_methods").list.join(" | "))
            .otherwise(None)
            .alias("ontology_match_method"),
            pl.when(pl.col("_ontology_aliases").list.len() > 0)
            .then(pl.col("_ontology_aliases").list.join(" | "))
            .otherwise(None)
            .alias("ontology_matched_alias"),
            pl.when(pl.col("_ontology_scopes").list.len() > 0)
            .then(pl.col("_ontology_scopes").list.join(" | "))
            .otherwise(None)
            .alias("ontology_alias_scope"),
        )
        .with_columns(
            pl.when(pl.col("rel").str.starts_with("STRAIN"))
            .then(pl.col("strain_id"))
            .otherwise(pl.col("entity_node_id"))
            .alias("source"),
            pl.when(pl.col("rel").str.starts_with("STRAIN"))
            .then(pl.col("entity_node_id"))
            .otherwise(pl.col("strain_id"))
            .alias("target"),
            pl.col("rel").str.split(":").list.get(0).alias("entity_pair"),
            pl.col("rel").str.split(":").list.get(1).alias("relation_name"),
        )
        .with_columns(
            pl.col("entity_pair").str.split("-").list.get(0).alias("source_ner"),
            pl.col("entity_pair").str.split("-").list.get(1).alias("target_ner"),
        )
        .select(
            "source",
            "target",
            pl.col("relation_name").alias("rel"),
            "source_ner",
            "target_ner",
            pl.col("entity_node_label").alias("entity_label"),
            "entity_surfaces",
            "ontology_status",
            "ontology",
            "ontology_id",
            "ontology_match_method",
            "ontology_match_confidence",
            "ontology_matched_alias",
            "ontology_alias_scope",
        )
    )
    if not ontology_grounded_input:
        network = network.select(
            "source",
            "target",
            "rel",
            "source_ner",
            "target_ner",
        )
    strains = matched.select("strain_id").unique().sort("strain_id")

    network_file.parent.mkdir(parents=True, exist_ok=True)
    network_temporary = network_file.with_suffix(network_file.suffix + ".tmp")
    network.sink_csv(
        network_temporary,
        separator="\t",
        engine="streaming",
    )
    os.replace(network_temporary, network_file)

    strains_file.parent.mkdir(parents=True, exist_ok=True)
    strains_temporary = strains_file.with_suffix(strains_file.suffix + ".tmp")
    strains.sink_csv(
        strains_temporary,
        include_header=False,
        engine="streaming",
    )
    os.replace(strains_temporary, strains_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--network-output", required=True, type=Path)
    parser.add_argument("--strains-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_relation_network(
        args.predictions,
        args.network_output,
        args.strains_output,
    )


if __name__ == "__main__":
    main()
