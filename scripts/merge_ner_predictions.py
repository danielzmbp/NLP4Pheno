#!/usr/bin/env python3
"""Join STRAIN and non-STRAIN NER predictions with bounded memory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl


def merge_ner_predictions(
    strains_file: str | Path,
    other_entities_file: str | Path,
    output_file: str | Path,
    cutoff: float,
) -> None:
    """Write the relation-candidate entity pairs as a streaming Parquet join."""
    strains_file = Path(strains_file)
    other_entities_file = Path(other_entities_file)
    output_file = Path(output_file)

    strains = pl.scan_parquet(strains_file)
    others = pl.scan_parquet(other_entities_file)

    strain_columns = strains.collect_schema().names()
    other_columns = others.collect_schema().names()
    if not strain_columns or strain_columns[0] != "text":
        raise ValueError("The STRAIN prediction table must start with a text column.")
    for required in ("text", "word", "score"):
        if required not in other_columns:
            raise ValueError(
                f"The non-STRAIN prediction table is missing required column: {required}"
            )

    renamed_strain_columns = [
        pl.col(column).alias(f"{column}_strain")
        for column in strain_columns
        if column != "text"
    ]
    strains = strains.select(pl.col("text"), *renamed_strain_columns)
    others = others.filter(
        pl.col("word").is_not_null() & (pl.col("score") > cutoff)
    )

    merged = strains.join(others, on="text", how="inner")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(f".{output_file.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        merged.sink_parquet(
            temporary_output,
            compression="snappy",
            maintain_order=True,
            mkdir=True,
            engine="streaming",
        )
        os.replace(temporary_output, output_file)
    finally:
        temporary_output.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-join STRAIN and other NER prediction Parquets."
    )
    parser.add_argument("strains")
    parser.add_argument("other_entities")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_ner_predictions(
        args.strains,
        args.other_entities,
        args.output,
        args.cutoff,
    )


if __name__ == "__main__":
    main()
