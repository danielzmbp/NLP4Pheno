#!/usr/bin/env python3
"""Merge the compact StrainInfo snapshot with detailed alias exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


REQUIRED_COMPACT_COLUMNS = {
    "designation_key",
    "designation",
    "si_id",
    "taxon",
    "type_strain",
}
REQUIRED_DETAILED_COLUMNS = {
    "SI_ID",
    "Taxon_Name",
    "Type_Strain",
    "Deposit_Designations",
    "Other_Designations",
}


def designation_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_aliases(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = REQUIRED_COMPACT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Compact aliases are missing columns: {sorted(missing)}")
    return frame.select(*sorted(REQUIRED_COMPACT_COLUMNS)).with_columns(
        pl.lit(True).alias("in_compact"),
        pl.lit(False).alias("in_detailed_deposit"),
        pl.lit(False).alias("in_detailed_other"),
        pl.lit(0, dtype=pl.Int8).alias("source_rank"),
    )


def _detailed_aliases(frame: pl.DataFrame, column: str, source: str) -> pl.DataFrame:
    is_deposit = source == "deposit"
    return (
        frame.select(
            pl.col("SI_ID").cast(pl.Int64).alias("si_id"),
            pl.col("Taxon_Name").cast(pl.String).alias("taxon"),
            pl.col("Type_Strain")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
            .alias("type_strain"),
            pl.col(column).cast(pl.String).str.split(";").alias("designation"),
        )
        .explode("designation", empty_as_null=True)
        .with_columns(pl.col("designation").str.strip_chars())
        .filter(pl.col("designation").is_not_null() & (pl.col("designation") != ""))
        .with_columns(
            pl.col("designation")
            .map_elements(designation_key, return_dtype=pl.String)
            .alias("designation_key"),
            pl.lit(False).alias("in_compact"),
            pl.lit(is_deposit).alias("in_detailed_deposit"),
            pl.lit(not is_deposit).alias("in_detailed_other"),
            pl.lit(1 if is_deposit else 2, dtype=pl.Int8).alias("source_rank"),
        )
        .filter(pl.col("designation_key") != "")
    )


def detailed_aliases(path: Path) -> tuple[pl.DataFrame, dict[str, int]]:
    frame = pl.read_csv(path, infer_schema_length=10_000, null_values=[""])
    missing = REQUIRED_DETAILED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Detailed aliases are missing columns: {sorted(missing)}")
    id_counts = frame.group_by("SI_ID").len()
    stats = {
        "row_count": frame.height,
        "unique_si_id_count": frame.get_column("SI_ID").n_unique(),
        "duplicated_si_id_count": id_counts.filter(pl.col("len") > 1).height,
    }
    aliases = pl.concat(
        [
            _detailed_aliases(frame, "Deposit_Designations", "deposit"),
            _detailed_aliases(frame, "Other_Designations", "other"),
        ],
        how="vertical",
    )
    return aliases, stats


def merge_aliases(compact: pl.DataFrame, detailed: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "designation_key",
        "designation",
        "si_id",
        "taxon",
        "type_strain",
        "in_compact",
        "in_detailed_deposit",
        "in_detailed_other",
        "source_rank",
    ]
    combined = pl.concat(
        [compact.select(columns), detailed.select(columns)], how="vertical"
    ).sort(["designation_key", "si_id", "source_rank"])
    return (
        combined.group_by("designation_key", "si_id", maintain_order=True)
        .agg(
            pl.col("designation").first(),
            pl.col("taxon").drop_nulls().first(),
            pl.col("type_strain").any(),
            pl.col("in_compact").any(),
            pl.col("in_detailed_deposit").any(),
            pl.col("in_detailed_other").any(),
        )
        .sort(["designation_key", "si_id"])
    )


def build_union(compact_path: Path, detailed_path: Path) -> tuple[pl.DataFrame, dict]:
    compact = compact_aliases(compact_path)
    detailed, detailed_stats = detailed_aliases(detailed_path)
    union = merge_aliases(compact, detailed)
    compact_ids = set(compact.get_column("si_id").unique().to_list())
    detailed_ids = set(detailed.get_column("si_id").unique().to_list())
    key_counts = union.group_by("designation_key").agg(pl.col("si_id").n_unique())
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compact_input": {
            "name": compact_path.name,
            "sha256": sha256_file(compact_path),
            "alias_pair_count": compact.height,
            "si_id_count": len(compact_ids),
        },
        "detailed_input": {
            "name": detailed_path.name,
            "sha256": sha256_file(detailed_path),
            **detailed_stats,
            "alias_pair_count_before_union": detailed.unique(
                subset=["designation_key", "si_id"]
            ).height,
        },
        "union": {
            "alias_pair_count": union.height,
            "designation_key_count": union.get_column("designation_key").n_unique(),
            "si_id_count": union.get_column("si_id").n_unique(),
            "ambiguous_designation_key_count": key_counts.filter(
                pl.col("si_id") > 1
            ).height,
            "compact_si_ids_missing_from_detailed": len(compact_ids - detailed_ids),
            "detailed_si_ids_missing_from_compact": len(detailed_ids - compact_ids),
            "compact_only_alias_pairs": union.filter(
                pl.col("in_compact")
                & ~pl.col("in_detailed_deposit")
                & ~pl.col("in_detailed_other")
            ).height,
            "detailed_only_alias_pairs": union.filter(~pl.col("in_compact")).height,
        },
        "matching_policy": (
            "compact and detailed aliases are retained; alias-to-multiple-SI-ID "
            "ambiguity is preserved for contextual resolution"
        ),
    }
    return union, summary


def write_outputs(
    union: pl.DataFrame, summary: dict, output: Path, summary_output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    union.write_parquet(temporary, compression="zstd")
    temporary.replace(output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_output.with_suffix(summary_output.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True, type=Path)
    parser.add_argument("--detailed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    union, summary = build_union(args.compact, args.detailed)
    write_outputs(union, summary, args.output, args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
