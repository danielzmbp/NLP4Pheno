#!/usr/bin/env python3
"""Add relation-model entity markers without materializing the full corpus."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

from relation_prediction_utils import add_formatted_text


def format_relation_sentences(
    input_file: str | Path,
    output_file: str | Path,
) -> None:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(f".{output_file.name}.tmp")
    temporary_output.unlink(missing_ok=True)

    formatted = add_formatted_text(pl.scan_parquet(input_file))
    try:
        formatted.sink_parquet(
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
        description="Stream relation-model marker formatting to Parquet."
    )
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    format_relation_sentences(args.input, args.output)


if __name__ == "__main__":
    main()
