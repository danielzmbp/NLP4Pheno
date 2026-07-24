"""Vectorized preparation of NER pairs for relation prediction."""

from __future__ import annotations

import polars as pl


def add_formatted_text(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Insert both entity markers once, using offsets from the same sentence."""
    strain_first = (
        pl.col("text").str.slice(0, pl.col("start_strain"))
        + pl.lit("@STRAIN$")
        + pl.col("text").str.slice(
            pl.col("end_strain"),
            (pl.col("start") - pl.col("end_strain")).clip(lower_bound=0),
        )
        + pl.lit("@")
        + pl.col("ner")
        + pl.lit("$")
        + pl.col("text").str.slice(pl.col("end"))
    )
    other_first = (
        pl.col("text").str.slice(0, pl.col("start"))
        + pl.lit("@")
        + pl.col("ner")
        + pl.lit("$")
        + pl.col("text").str.slice(
            pl.col("end"),
            (pl.col("start_strain") - pl.col("end")).clip(lower_bound=0),
        )
        + pl.lit("@STRAIN$")
        + pl.col("text").str.slice(pl.col("end_strain"))
    )
    return frame.with_columns(
        pl.when(pl.col("end_strain") <= pl.col("start"))
        .then(strain_first)
        .when(pl.col("end") <= pl.col("start_strain"))
        .then(other_first)
        .otherwise(None)
        .alias("formatted_text")
    ).filter(pl.col("formatted_text").is_not_null())
