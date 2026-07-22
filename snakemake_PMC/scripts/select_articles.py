#!/usr/bin/env python3
"""Join ESearch PMCIDs to the newest version in a PMC S3 inventory."""

import argparse
import csv
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(snakemake.scriptdir) if "snakemake" in globals() else Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import INVENTORY_COLUMNS, MANIFEST_SCHEMA, redirect_snakemake_log  # noqa: E402


METADATA_KEY_RE = re.compile(r"(?:^|/)PMC(\d+)\.(\d+)\.json$")


def read_pmcids(path: Path) -> set[int]:
    identifiers: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().upper()
        if not value:
            continue
        if value.startswith("PMC"):
            value = value[3:]
        if not value.isdigit():
            raise ValueError(f"Invalid PMCID: {line!r}")
        identifiers.add(int(value))
    if not identifiers:
        raise RuntimeError(f"No PMCIDs were read from {path}")
    return identifiers


def open_inventory(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def load_latest_versions(
    identifiers: set[int], inventory_files: list[Path]
) -> tuple[dict[int, tuple[int, str, str, str]], int]:
    # value: version, key, last_modified, etag
    latest: dict[int, tuple[int, str, str, str]] = {}
    rows_scanned = 0
    for path in inventory_files:
        with open_inventory(path) as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows_scanned += 1
                if len(row) < len(INVENTORY_COLUMNS):
                    continue
                _, key, last_modified, etag = row[:4]
                match = METADATA_KEY_RE.search(key)
                if not match:
                    continue
                pmcid_num = int(match.group(1))
                if pmcid_num not in identifiers:
                    continue
                version = int(match.group(2))
                previous = latest.get(pmcid_num)
                if previous is None or version > previous[0]:
                    latest[pmcid_num] = (version, key, last_modified, etag.strip('"'))
        print(f"Scanned {path.name}: {rows_scanned:,} cumulative inventory rows")
    return latest, rows_scanned


def xml_url(
    *,
    article_version: str,
    bucket_url: str,
    fixture_xml_dir: Path | None,
) -> str:
    filename = f"{article_version}.xml"
    if fixture_xml_dir is not None:
        return (fixture_xml_dir / filename).resolve().as_uri()
    key = f"{article_version}/{filename}"
    return f"{bucket_url.rstrip('/')}/{quote(key, safe='/')}"


def metadata_url(*, key: str, bucket_url: str, fixture_xml_dir: Path | None) -> str | None:
    if fixture_xml_dir is not None:
        return None
    return f"{bucket_url.rstrip('/')}/{quote(key.lstrip('/'), safe='/')}"


def write_manifest(
    path: Path,
    identifiers: set[int],
    latest: dict[int, tuple[int, str, str, str]],
    *,
    snapshot: str,
    bucket_url: str,
    fixture_xml_dir: Path | None,
    batch_size: int = 100_000,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary,
        MANIFEST_SCHEMA,
        compression="zstd",
        write_statistics=True,
    )
    written = 0
    records: list[dict] = []
    try:
        for pmcid_num in sorted(identifiers):
            selected = latest.get(pmcid_num)
            if selected is None:
                continue
            version, key, last_modified, etag = selected
            pmcid = f"PMC{pmcid_num}"
            article_version = f"{pmcid}.{version}"
            records.append(
                {
                    "pmcid": pmcid,
                    "pmcid_num": pmcid_num,
                    "version": version,
                    "article_version": article_version,
                    "inventory_last_modified": last_modified or None,
                    "source_etag": etag or None,
                    "metadata_key": key,
                    "metadata_url": metadata_url(
                        key=key,
                        bucket_url=bucket_url,
                        fixture_xml_dir=fixture_xml_dir,
                    ),
                    "xml_url": xml_url(
                        article_version=article_version,
                        bucket_url=bucket_url,
                        fixture_xml_dir=fixture_xml_dir,
                    ),
                    "snapshot_date": snapshot,
                }
            )
            if len(records) >= batch_size:
                table = pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA)
                writer.write_table(table)
                written += len(records)
                records.clear()
        if records:
            table = pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA)
            writer.write_table(table)
            written += len(records)
    finally:
        writer.close()
    temporary.replace(path)
    return written


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    ids_path: Path,
    search_manifest_path: Path,
    inventory_completion_path: Path,
    output_path: Path,
    summary_path: Path,
    bucket_url: str,
    fixture_xml_dir: Path | None,
    snapshot: str,
    max_missing_fraction: float,
) -> None:
    identifiers = read_pmcids(ids_path)
    search_manifest = json.loads(search_manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(inventory_completion_path.read_text(encoding="utf-8"))
    inventory_files = [Path(item["local_path"]) for item in completion.get("files", [])]
    if not inventory_files:
        raise RuntimeError("Inventory completion file lists no local inventory parts")
    missing_files = [str(path) for path in inventory_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Inventory parts are missing: {missing_files[:5]}")

    latest, rows_scanned = load_latest_versions(identifiers, inventory_files)
    missing = sorted(identifiers - latest.keys())
    missing_fraction = len(missing) / len(identifiers)
    written = write_manifest(
        output_path,
        identifiers,
        latest,
        snapshot=snapshot,
        bucket_url=bucket_url,
        fixture_xml_dir=fixture_xml_dir,
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_date": snapshot,
        "query": search_manifest.get("query"),
        "resolved_inventory_version": completion.get("resolved_inventory_version"),
        "inventory_rows_scanned": rows_scanned,
        "requested_pmcids": len(identifiers),
        "selected_article_versions": written,
        "missing_pmcids": len(missing),
        "missing_fraction": missing_fraction,
        "missing_pmcid_examples": [f"PMC{value}" for value in missing[:100]],
        "manifest_schema": str(MANIFEST_SCHEMA),
        "output": str(output_path.resolve()),
    }
    atomic_json(summary_path, summary)
    print(
        f"Selected {written:,}/{len(identifiers):,} PMCIDs; "
        f"{len(missing):,} were absent from the inventory"
    )
    if missing_fraction > max_missing_fraction:
        raise RuntimeError(
            f"Missing PMCID fraction {missing_fraction:.3%} exceeds "
            f"the configured limit {max_missing_fraction:.3%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--search-manifest", required=True)
    parser.add_argument("--inventory-completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--bucket-url", required=True)
    parser.add_argument("--fixture-xml-dir")
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--max-missing-fraction", type=float, default=0.01)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    run(
        ids_path=Path(args.ids),
        search_manifest_path=Path(args.search_manifest),
        inventory_completion_path=Path(args.inventory_completion),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        bucket_url=args.bucket_url,
        fixture_xml_dir=Path(args.fixture_xml_dir) if args.fixture_xml_dir else None,
        snapshot=args.snapshot_date,
        max_missing_fraction=args.max_missing_fraction,
    )


def snakemake_entrypoint() -> None:
    fixture_value = str(snakemake.params.fixture_xml_dir)
    run(
        ids_path=Path(str(snakemake.input.ids)),
        search_manifest_path=Path(str(snakemake.input.search)),
        inventory_completion_path=Path(str(snakemake.input.inventory)),
        output_path=Path(str(snakemake.output.manifest)),
        summary_path=Path(str(snakemake.output.summary)),
        bucket_url=str(snakemake.params.bucket_url),
        fixture_xml_dir=Path(fixture_value) if fixture_value else None,
        snapshot=str(snakemake.params.snapshot_date),
        max_missing_fraction=float(snakemake.params.max_missing_fraction),
    )


if "snakemake" in globals():
    redirect_snakemake_log(snakemake)
    snakemake_entrypoint()
elif __name__ == "__main__":
    cli()
