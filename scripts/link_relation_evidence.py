#!/usr/bin/env python3
"""Stream PMC provenance onto relation predictions and network edges."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl


PMC_COLUMNS = [
    "text",
    "pmcid",
    "article_version",
    "paragraph",
    "sentence_range",
]


def link_predictions_to_pmc(
    predictions_file: Path,
    pmc_file: Path,
    output_file: Path,
) -> None:
    linked = pl.scan_parquet(predictions_file).join(
        pl.scan_parquet(pmc_file).select(PMC_COLUMNS),
        on="text",
        how="left",
        maintain_order="left",
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    linked.sink_parquet(
        temporary,
        compression="snappy",
        engine="streaming",
    )
    os.replace(temporary, output_file)


def link_network_to_evidence(
    network_file: Path,
    predictions_file: Path,
    output_file: Path,
) -> None:
    evidence = (
        pl.scan_parquet(predictions_file)
        .filter(pl.col("straininfo_si_id").is_not_null())
        .with_columns(
            pl.concat_str(
                pl.lit("SI-ID"),
                pl.col("straininfo_si_id").cast(pl.Int64).cast(pl.String),
            ).alias("strain_id")
        )
        .with_columns(
            pl.when(pl.col("rel").str.starts_with("STRAIN"))
            .then(pl.col("strain_id"))
            .otherwise(pl.col("word_qc_group"))
            .alias("source"),
            pl.when(pl.col("rel").str.starts_with("STRAIN"))
            .then(pl.col("word_qc_group"))
            .otherwise(pl.col("strain_id"))
            .alias("target"),
            pl.col("rel").str.split(":").list.get(1).alias("rel_name"),
        )
        .select(
            "source",
            "target",
            "rel_name",
            "pmcid",
            "article_version",
            "paragraph",
            "sentence_range",
        )
        .unique()
    )
    linked = pl.scan_csv(network_file, separator="\t").join(
        evidence,
        left_on=["source", "target", "rel"],
        right_on=["source", "target", "rel_name"],
        how="left",
        maintain_order="left",
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    linked.sink_csv(
        temporary,
        separator="\t",
        engine="streaming",
    )
    os.replace(temporary, output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predictions_parser = subparsers.add_parser("predictions")
    predictions_parser.add_argument("predictions", type=Path)
    predictions_parser.add_argument("pmc", type=Path)
    predictions_parser.add_argument("--output", required=True, type=Path)

    network_parser = subparsers.add_parser("network")
    network_parser.add_argument("network", type=Path)
    network_parser.add_argument("predictions", type=Path)
    network_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "predictions":
        link_predictions_to_pmc(args.predictions, args.pmc, args.output)
    else:
        link_network_to_evidence(args.network, args.predictions, args.output)


if __name__ == "__main__":
    main()
