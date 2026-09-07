#!/usr/bin/env python3
"""Group near-identical relation entities without a dense all-pairs matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def _empty_mapping() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ner": pl.String,
            "word_qc": pl.String,
            "word_qc_group": pl.String,
        }
    )


def _entity_mapping(
    ner: str,
    words: list[str],
    frequencies: list[int],
    *,
    cutoff: float,
    workers: int,
    matrix_mb: int,
) -> list[tuple[str, str, str]]:
    """Return changed word-to-consensus mappings for one entity class."""
    if not words:
        return []

    query_indices = [index for index, count in enumerate(frequencies) if count > 1]
    if not query_indices:
        return []

    parent = np.arange(len(words), dtype=np.int64)
    word_to_index = {word: index for index, word in enumerate(words)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # RapidFuzz returns float32 scores by default. Bound each temporary matrix
    # independently of vocabulary size while preserving exact cutoff behavior.
    matrix_bytes = max(1, matrix_mb) * 1024 * 1024
    rows_per_batch = max(
        1,
        matrix_bytes // (np.dtype(np.float32).itemsize * len(words)),
    )
    rows_per_batch = min(rows_per_batch, len(query_indices))
    total_batches = (len(query_indices) + rows_per_batch - 1) // rows_per_batch

    print(
        f"{ner}: {len(words):,} unique words, {len(query_indices):,} repeated "
        f"queries, {rows_per_batch:,} queries/batch ({total_batches:,} batches)",
        flush=True,
    )

    for batch_number, start in enumerate(
        range(0, len(query_indices), rows_per_batch), start=1
    ):
        batch_indices = query_indices[start : start + rows_per_batch]
        query_words = [words[index] for index in batch_indices]
        scores = process.cdist(
            query_words,
            words,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=cutoff,
            dtype=np.float32,
            workers=workers,
        )
        for row_index, query_word in enumerate(query_words):
            query_index = word_to_index[query_word]
            for match_index in np.flatnonzero(scores[row_index] >= cutoff):
                union(query_index, int(match_index))
        del scores

        if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_batches:
            print(
                f"{ner}: completed similarity batch "
                f"{batch_number:,}/{total_batches:,}",
                flush=True,
            )

    best_by_root: dict[int, int] = {}
    for index, (word, frequency) in enumerate(zip(words, frequencies, strict=True)):
        root = find(index)
        current = best_by_root.get(root)
        if current is None:
            best_by_root[root] = index
            continue
        current_key = (frequencies[current], words[current])
        candidate_key = (frequency, word)
        if candidate_key > current_key:
            best_by_root[root] = index

    mapping: list[tuple[str, str, str]] = []
    for index, word in enumerate(words):
        consensus = words[best_by_root[find(index)]]
        if consensus != word:
            mapping.append((ner, word, consensus))
    return mapping


def build_consensus_mapping(
    counts: pl.DataFrame,
    *,
    cutoff: float = 95,
    workers: int = 1,
    matrix_mb: int = 256,
) -> pl.DataFrame:
    """Build deterministic near-duplicate mappings from per-entity word counts."""
    required = {"ner", "word_qc", "count"}
    missing = required.difference(counts.columns)
    if missing:
        raise ValueError(f"Counts are missing required columns: {sorted(missing)}")

    mappings: list[tuple[str, str, str]] = []
    entity_types = counts.get_column("ner").drop_nulls().unique().sort().to_list()
    for ner in entity_types:
        entity_counts = counts.filter(pl.col("ner") == ner).sort(
            ["count", "word_qc"],
            descending=[True, False],
        )
        mappings.extend(
            _entity_mapping(
                str(ner),
                entity_counts.get_column("word_qc").to_list(),
                entity_counts.get_column("count").cast(pl.Int64).to_list(),
                cutoff=cutoff,
                workers=workers,
                matrix_mb=matrix_mb,
            )
        )

    if not mappings:
        return _empty_mapping()
    return pl.DataFrame(
        mappings,
        schema=["ner", "word_qc", "word_qc_group"],
        orient="row",
    )


def group_relation_entities(
    input_file: Path,
    output_file: Path,
    *,
    cutoff: float = 95,
    workers: int = 1,
    matrix_mb: int = 256,
) -> None:
    """Build the fuzzy mapping in bounded blocks and stream it onto all rows."""
    source = pl.scan_parquet(input_file)
    counts = (
        source.select("ner", "word_qc")
        .filter(pl.col("ner").is_not_null() & pl.col("word_qc").is_not_null())
        .group_by("ner", "word_qc")
        .len(name="count")
        .collect(engine="streaming")
    )
    print(
        f"Loaded {counts.height:,} unique (entity type, word) counts",
        flush=True,
    )
    mapping = build_consensus_mapping(
        counts,
        cutoff=cutoff,
        workers=workers,
        matrix_mb=matrix_mb,
    )
    print(f"Applying {mapping.height:,} changed-word mappings", flush=True)

    prepared = (
        source.drop(["label_rel", "label"], strict=False)
        .drop("word_qc_group", strict=False)
        .rename({"score": "ner_score"})
    )
    if mapping.height:
        prepared = prepared.join(
            mapping.lazy(),
            on=["ner", "word_qc"],
            how="left",
            maintain_order="left",
        ).with_columns(
            pl.coalesce("word_qc_group", "word_qc").alias("word_qc_group")
        )
    else:
        prepared = prepared.with_columns(pl.col("word_qc").alias("word_qc_group"))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    prepared.sink_parquet(
        temporary,
        compression="snappy",
        engine="streaming",
    )
    os.replace(temporary, output_file)
    print(f"Wrote grouped predictions to {output_file}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=float, default=95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--matrix-mb",
        type=int,
        default=256,
        help="Maximum size of each temporary float32 similarity matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_relation_entities(
        args.input,
        args.output,
        cutoff=args.cutoff,
        workers=args.workers,
        matrix_mb=args.matrix_mb,
    )


if __name__ == "__main__":
    main()
