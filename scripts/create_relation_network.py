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

    network = (
        matched.select("strain_id", "word_qc_group", "rel")
        .unique()
        .with_columns(
            pl.when(pl.col("rel").str.starts_with("STRAIN"))
            .then(pl.col("strain_id"))
            .otherwise(pl.col("word_qc_group"))
            .alias("source"),
            pl.when(pl.col("rel").str.starts_with("STRAIN"))
            .then(pl.col("word_qc_group"))
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
        )
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
