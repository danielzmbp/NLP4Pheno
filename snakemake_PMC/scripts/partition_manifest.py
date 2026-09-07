#!/usr/bin/env python3
"""Partition the article manifest by stable numeric PMCID modulo."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(snakemake.scriptdir) if "snakemake" in globals() else Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MANIFEST_SCHEMA, empty_table, redirect_snakemake_log  # noqa: E402


def partition_manifest(
    input_path: Path,
    output_paths: list[Path],
    chunk_count: int,
    batch_size: int = 100_000,
) -> None:
    if len(output_paths) != chunk_count:
        raise ValueError(
            f"Expected {chunk_count} output paths, received {len(output_paths)}"
        )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    source = pq.ParquetFile(input_path)
    if not source.schema_arrow.equals(MANIFEST_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Unexpected article manifest schema:\n{source.schema_arrow}\n"
            f"Expected:\n{MANIFEST_SCHEMA}"
        )

    temporary_paths = [path.with_suffix(path.suffix + ".tmp") for path in output_paths]
    writers: dict[int, pq.ParquetWriter] = {}
    row_counts = np.zeros(chunk_count, dtype=np.int64)
    try:
        for batch in source.iter_batches(batch_size=batch_size):
            pmcid_numbers = batch.column(
                batch.schema.get_field_index("pmcid_num")
            ).to_numpy(zero_copy_only=False)
            assignments = pmcid_numbers % chunk_count
            for chunk_index in np.unique(assignments):
                index = int(chunk_index)
                indices = np.flatnonzero(assignments == index)
                selected = pa.Table.from_batches([batch.take(pa.array(indices))])
                if index not in writers:
                    writers[index] = pq.ParquetWriter(
                        temporary_paths[index],
                        MANIFEST_SCHEMA,
                        compression="zstd",
                        write_statistics=True,
                    )
                writers[index].write_table(selected)
                row_counts[index] += len(indices)
    finally:
        for writer in writers.values():
            writer.close()

    for index, destination in enumerate(output_paths):
        temporary = temporary_paths[index]
        if index not in writers:
            pq.write_table(empty_table(MANIFEST_SCHEMA), temporary, compression="zstd")
        temporary.replace(destination)

    print(
        f"Partitioned {int(row_counts.sum()):,} article versions into "
        f"{chunk_count} stable chunks (min={int(row_counts.min()):,}, "
        f"max={int(row_counts.max()):,})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    outputs = [Path(args.output_dir) / f"{index:04d}.parquet" for index in range(args.chunk_count)]
    partition_manifest(Path(args.input), outputs, args.chunk_count)


def snakemake_entrypoint() -> None:
    partition_manifest(
        Path(str(snakemake.input[0])),
        [Path(str(value)) for value in snakemake.output],
        int(snakemake.params.chunk_count),
    )


if "snakemake" in globals():
    redirect_snakemake_log(snakemake)
    snakemake_entrypoint()
elif __name__ == "__main__":
    cli()
