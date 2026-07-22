#!/usr/bin/env python3
"""Attach conservative StrainInfo alias matches to relation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from straininfo_matching import resolve_mentions


def match_predictions(
    predictions_file: Path,
    designations_file: Path,
    output_file: Path,
    summary_file: Path,
) -> dict:
    predictions = pl.read_parquet(predictions_file)
    mention_column = "word_strain_qc"
    if mention_column not in predictions.columns:
        raise ValueError(f"Predictions are missing {mention_column!r}")
    aliases = pl.read_parquet(designations_file)
    mentions = (
        predictions.get_column(mention_column).drop_nulls().unique().to_list()
    )
    resolved = resolve_mentions(mentions, aliases)
    output = predictions.join(
        resolved,
        left_on=mention_column,
        right_on="mention",
        how="left",
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    output.write_parquet(temporary, compression="zstd")
    temporary.replace(output_file)

    status_counts = {
        str(row["straininfo_status"]): int(row["len"])
        for row in resolved.group_by("straininfo_status").len().iter_rows(named=True)
    }
    method_counts = {
        str(row["straininfo_method"]): int(row["len"])
        for row in resolved.filter(pl.col("straininfo_method").is_not_null())
        .group_by("straininfo_method")
        .len()
        .iter_rows(named=True)
    }
    matched_rows = output.filter(pl.col("straininfo_si_id").is_not_null()).height
    summary = {
        "policy": "exact or token-bounded alias; taxonomy contradictions rejected; no fuzzy fallback",
        "prediction_rows": predictions.height,
        "matched_prediction_rows": matched_rows,
        "unique_mentions": len(mentions),
        "status_counts": dict(sorted(status_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_file.with_suffix(summary_file.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_summary.replace(summary_file)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("designations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = match_predictions(
        args.predictions, args.designations, args.output, args.summary
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
