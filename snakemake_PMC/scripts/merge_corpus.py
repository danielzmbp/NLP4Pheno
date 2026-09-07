#!/usr/bin/env python3
"""Stream validated PMC chunk files into final Parquet artifacts."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


SCRIPT_DIR = Path(snakemake.scriptdir) if "snakemake" in globals() else Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import ARTICLE_SCHEMA, CORPUS_SCHEMA, redirect_snakemake_log  # noqa: E402


def merge_parquet_files(
    inputs: list[Path],
    output: Path,
    schema,
    *,
    compression: str,
    row_group_size: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression=compression,
        write_statistics=True,
    )
    rows = 0
    try:
        for path in inputs:
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError(
                    f"Unexpected schema in {path}:\n{parquet.schema_arrow}\nExpected:\n{schema}"
                )
            for batch in parquet.iter_batches(batch_size=row_group_size):
                writer.write_batch(batch, row_group_size=row_group_size)
                rows += batch.num_rows
    finally:
        writer.close()
    temporary.replace(output)
    return rows


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    sentence_inputs: list[Path],
    article_inputs: list[Path],
    stats_inputs: list[Path],
    selection_path: Path,
    search_path: Path,
    inventory_path: Path,
    corpus_output: Path,
    article_output: Path,
    summary_output: Path,
    snapshot: str,
    compression: str,
    row_group_size: int,
    max_failure_fraction: float,
    min_chunk_failures_for_abort: int,
) -> None:
    if len(sentence_inputs) != len(article_inputs) or len(sentence_inputs) != len(stats_inputs):
        raise ValueError("Sentence, article and statistics chunk counts do not match")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    search = json.loads(search_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    chunk_stats = [json.loads(path.read_text(encoding="utf-8")) for path in stats_inputs]

    attempted = sum(item["articles"]["attempted"] for item in chunk_stats)
    parsed = sum(item["articles"]["parsed"] for item in chunk_stats)
    included = sum(item["articles"]["included"] for item in chunk_stats)
    excluded = sum(item["articles"]["excluded"] for item in chunk_stats)
    failed = sum(item["articles"]["failed"] for item in chunk_stats)
    expected = int(selection["selected_article_versions"])
    if attempted != expected:
        raise RuntimeError(
            f"Chunk statistics account for {attempted:,} articles, but selection contains "
            f"{expected:,}"
        )
    if parsed + failed != attempted or included + excluded != parsed:
        raise RuntimeError("Article accounting is inconsistent across chunk statistics")

    failure_fraction = failed / attempted if attempted else 0.0
    excessive_chunks = []
    for item in chunk_stats:
        articles = item["articles"]
        if (
            articles["failed"] >= min_chunk_failures_for_abort
            and articles["failure_fraction"] > max_failure_fraction
        ):
            excessive_chunks.append(
                f'{item["chunk"]}={articles["failed"]}/{articles["attempted"]}'
            )
    if failure_fraction > max_failure_fraction or excessive_chunks:
        details = ", ".join(excessive_chunks[:20]) or "none"
        raise RuntimeError(
            f"Article failure fraction {failure_fraction:.3%} exceeds or is "
            f"incompatible with the configured {max_failure_fraction:.3%} limit; "
            f"excessive chunks: {details}. Failure TSV files were preserved."
        )

    corpus_rows = merge_parquet_files(
        sentence_inputs,
        corpus_output,
        CORPUS_SCHEMA,
        compression=compression,
        row_group_size=row_group_size,
    )
    article_rows = merge_parquet_files(
        article_inputs,
        article_output,
        ARTICLE_SCHEMA,
        compression=compression,
        row_group_size=row_group_size,
    )
    if article_rows != parsed:
        raise RuntimeError(
            f"Final article metadata has {article_rows:,} rows; expected {parsed:,}"
        )
    expected_corpus_rows = sum(item["corpus_rows"] for item in chunk_stats)
    if corpus_rows != expected_corpus_rows:
        raise RuntimeError(
            f"Final corpus has {corpus_rows:,} rows; expected {expected_corpus_rows:,}"
        )
    if included and corpus_rows == 0:
        raise RuntimeError("Included articles produced an empty sentence corpus")

    exclusions: Counter[str] = Counter()
    for item in chunk_stats:
        exclusions.update(item.get("exclusion_reasons", {}))

    summary = {
        "corpus_format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_date": snapshot,
        "source": {
            "service": "PMC ESearch and PMC Open Data on AWS",
            "query": search.get("query"),
            "start_date": search.get("start_date"),
            "resolved_inventory_version": inventory.get("resolved_inventory_version"),
            "search_manifest": str(search_path.resolve()),
            "inventory_manifest": str(inventory_path.resolve()),
        },
        "articles": {
            "requested_pmcids": selection["requested_pmcids"],
            "selected_versions": expected,
            "missing_from_inventory": selection["missing_pmcids"],
            "parsed": parsed,
            "included": included,
            "excluded": excluded,
            "failed": failed,
            "failure_fraction": failure_fraction,
        },
        "corpus": {
            "rows": corpus_rows,
            "paragraphs": sum(item["paragraphs"] for item in chunk_stats),
            "rows_over_configured_max_chars": sum(
                item["rows_over_max_text_chars"] for item in chunk_stats
            ),
        },
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "outputs": {
            "corpus": str(corpus_output.resolve()),
            "corpus_bytes": corpus_output.stat().st_size,
            "article_metadata": str(article_output.resolve()),
            "article_metadata_bytes": article_output.stat().st_size,
        },
        "schemas": {
            "corpus": str(CORPUS_SCHEMA),
            "article_metadata": str(ARTICLE_SCHEMA),
        },
        "chunks": len(chunk_stats),
        "failure_policy": {
            "max_failure_fraction": max_failure_fraction,
            "min_chunk_failures_for_abort": min_chunk_failures_for_abort,
        },
    }
    atomic_json(summary_output, summary)
    print(
        f"Final corpus: {corpus_rows:,} rows from {included:,} included articles; "
        f"metadata contains {article_rows:,} parsed articles"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", action="append", required=True)
    parser.add_argument("--articles", action="append", required=True)
    parser.add_argument("--stats", action="append", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--corpus-output", required=True)
    parser.add_argument("--article-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--max-failure-fraction", type=float, default=0.01)
    parser.add_argument("--min-chunk-failures-for-abort", type=int, default=5)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    run(
        sentence_inputs=[Path(value) for value in args.sentences],
        article_inputs=[Path(value) for value in args.articles],
        stats_inputs=[Path(value) for value in args.stats],
        selection_path=Path(args.selection),
        search_path=Path(args.search),
        inventory_path=Path(args.inventory),
        corpus_output=Path(args.corpus_output),
        article_output=Path(args.article_output),
        summary_output=Path(args.summary_output),
        snapshot=args.snapshot_date,
        compression=args.compression,
        row_group_size=args.row_group_size,
        max_failure_fraction=args.max_failure_fraction,
        min_chunk_failures_for_abort=args.min_chunk_failures_for_abort,
    )


def snakemake_entrypoint() -> None:
    run(
        sentence_inputs=[Path(str(value)) for value in snakemake.input.sentences],
        article_inputs=[Path(str(value)) for value in snakemake.input.articles],
        stats_inputs=[Path(str(value)) for value in snakemake.input.stats],
        selection_path=Path(str(snakemake.input.selection)),
        search_path=Path(str(snakemake.input.search)),
        inventory_path=Path(str(snakemake.input.inventory)),
        corpus_output=Path(str(snakemake.output.corpus)),
        article_output=Path(str(snakemake.output.articles)),
        summary_output=Path(str(snakemake.output.summary)),
        snapshot=str(snakemake.params.snapshot_date),
        compression=str(snakemake.params.compression),
        row_group_size=int(snakemake.params.row_group_size),
        max_failure_fraction=float(snakemake.params.max_failure_fraction),
        min_chunk_failures_for_abort=int(
            snakemake.params.min_chunk_failures_for_abort
        ),
    )


if "snakemake" in globals():
    redirect_snakemake_log(snakemake)
    snakemake_entrypoint()
elif __name__ == "__main__":
    cli()
