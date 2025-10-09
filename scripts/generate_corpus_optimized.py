#!/usr/bin/env python3
"""Generate corpus shards from a Parquet file using a single streaming pass."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shard a Parquet `text` column into flat text files"
    )
    parser.add_argument("input", help="Path to the source Parquet file")
    parser.add_argument("output_dir", help="Directory to emit corpus shards")
    parser.add_argument("parts", type=int, help="Number of shards to create")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
        help="Rows to read per Arrow batch",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for output files",
    )
    parser.add_argument(
        "--buffer-bytes",
        type=int,
        default=256 * 1024,
        help="Buffered writer size to use while flushing shard files",
    )
    return parser.parse_args()


def _partition_sizes(total_rows: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    base = total_rows // parts
    remainder = total_rows % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _write_shard(path: Path, lines: List[str], *, encoding: str, buffer_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        path.write_text("", encoding=encoding)
        return
    with path.open("w", encoding=encoding, buffering=buffer_bytes) as handle:
        handle.write("\n".join(lines))


def generate_corpus(
    input_parquet: Path,
    output_paths: Sequence[Path],
    *,
    batch_size: int = 1_000_000,
    encoding: str = "utf-8",
    buffer_bytes: int = 256 * 1024,
) -> None:
    if not output_paths:
        print("No output shards requested; exiting.")
        return

    parquet_file = pq.ParquetFile(str(input_parquet))
    metadata = parquet_file.metadata
    if metadata is not None and metadata.num_rows is not None:
        total_rows = metadata.num_rows
    elif metadata is not None:
        total_rows = 0
        for i in range(metadata.num_row_groups):
            total_rows += metadata.row_group(i).num_rows
    else:
        # Fallback: stream once just to count rows (should rarely happen)
        total_rows = 0
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=batch_size):
            total_rows += len(batch.column(0))

    shard_sizes = _partition_sizes(total_rows, len(output_paths))

    print(f"Total rows: {total_rows:,}")
    print(f"Shards requested: {len(output_paths)}")
    if shard_sizes:
        preview = ", ".join(
            f"{idx:04d}:{size:,}" for idx, size in enumerate(shard_sizes[:5])
        )
        suffix = " …" if len(shard_sizes) > 5 else ""
        print(f"First shards: {preview}{suffix}")

    dataset = ds.dataset(str(input_parquet), format="parquet")
    scanner = dataset.scanner(columns=["text"], batch_size=batch_size, use_threads=True)

    current_part = 0
    buffer: List[str] = []
    filled = 0

    for batch in scanner.to_batches():
        column = batch.column(0)
        if column.null_count:
            column = pc.drop_null(column)
        if len(column) == 0:
            continue

        values = column.to_pylist()
        idx = 0

        while idx < len(values) and current_part < len(output_paths):
            target = shard_sizes[current_part] if shard_sizes else 0

            if target == 0:
                _write_shard(
                    output_paths[current_part],
                    [],
                    encoding=encoding,
                    buffer_bytes=buffer_bytes,
                )
                current_part += 1
                filled = 0
                buffer.clear()
                continue

            remaining = target - filled
            take = min(remaining, len(values) - idx)
            buffer.extend(values[idx : idx + take])
            filled += take
            idx += take

            if filled == target:
                _write_shard(
                    output_paths[current_part],
                    buffer,
                    encoding=encoding,
                    buffer_bytes=buffer_bytes,
                )
                current_part += 1
                buffer = []
                filled = 0

    if buffer and current_part < len(output_paths):
        _write_shard(
            output_paths[current_part],
            buffer,
            encoding=encoding,
            buffer_bytes=buffer_bytes,
        )
        current_part += 1

    while current_part < len(output_paths):
        _write_shard(
            output_paths[current_part],
            [],
            encoding=encoding,
            buffer_bytes=buffer_bytes,
        )
        current_part += 1

    print("Done.")


def _run_from_snakemake() -> None:
    smk = globals()["snakemake"]
    input_parquet = Path(str(smk.input[0]))
    output_paths = [Path(p) for p in smk.output]
    params = getattr(smk, "params", {})
    batch_size = int(params.get("batch_size", 1_000_000))
    encoding = params.get("encoding", "utf-8")
    buffer_bytes = int(params.get("buffer_bytes", 256 * 1024))

    generate_corpus(
        input_parquet=input_parquet,
        output_paths=output_paths,
        batch_size=batch_size,
        encoding=encoding,
        buffer_bytes=buffer_bytes,
    )


def _run_from_cli() -> None:
    args = _parse_cli_args()
    output_paths = [
        Path(args.output_dir) / f"{idx:04d}.txt" for idx in range(args.parts)
    ]
    generate_corpus(
        input_parquet=Path(args.input),
        output_paths=output_paths,
        batch_size=args.batch_size,
        encoding=args.encoding,
        buffer_bytes=args.buffer_bytes,
    )


if "snakemake" in globals():
    _run_from_snakemake()
elif __name__ == "__main__":
    _run_from_cli()
