#!/usr/bin/env python3
"""Build deterministic known-positive and recent-background PMC smoke cohorts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from urllib.parse import quote

import requests


KEY_RE = re.compile(r"(?:^|/)PMC(\d+)\.(\d+)\.json$")
YEAR_RE = re.compile(r"\b((?:19|20|21)\d{2})\b")
_thread_state = threading.local()


def open_inventory(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def stable_hash(pmcid_num: int, seed: int) -> int:
    value = f"{seed}:PMC{pmcid_num}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def read_positive_pmcids(network_path: Path) -> set[int]:
    identifiers: set[int] = set()
    with network_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "pmcid" not in (reader.fieldnames or []):
            raise ValueError(f"{network_path} has no pmcid column")
        for row in reader:
            value = (row.get("pmcid") or "").strip().upper()
            if value.startswith("PMC") and value[3:].isdigit():
                identifiers.add(int(value[3:]))
    if not identifiers:
        raise RuntimeError("The benchmark network contains no valid PMCIDs")
    return identifiers


def add_reservoir_candidate(
    reservoir: dict[int, tuple[int, int, str]],
    heap: list[tuple[int, int]],
    *,
    pmcid_num: int,
    version: int,
    metadata_key: str,
    hash_value: int,
    limit: int,
) -> None:
    existing = reservoir.get(pmcid_num)
    if existing is not None:
        if version > existing[1]:
            reservoir[pmcid_num] = (hash_value, version, metadata_key)
        return
    if len(reservoir) < limit:
        reservoir[pmcid_num] = (hash_value, version, metadata_key)
        heapq.heappush(heap, (-hash_value, pmcid_num))
        return
    while heap and heap[0][1] not in reservoir:
        heapq.heappop(heap)
    worst_hash = -heap[0][0]
    if hash_value >= worst_hash:
        return
    _, worst_pmcid = heapq.heappop(heap)
    del reservoir[worst_pmcid]
    reservoir[pmcid_num] = (hash_value, version, metadata_key)
    heapq.heappush(heap, (-hash_value, pmcid_num))


def scan_inventory(
    inventory_files: list[Path],
    positives: set[int],
    *,
    seed: int,
    candidate_limit: int,
    recent_modified_after: str,
) -> tuple[dict[int, tuple[int, str]], dict[int, tuple[int, int, str]], int]:
    positive_latest: dict[int, tuple[int, str]] = {}
    background: dict[int, tuple[int, int, str]] = {}
    background_heap: list[tuple[int, int]] = []
    rows = 0
    for inventory_path in inventory_files:
        with open_inventory(inventory_path) as handle:
            for row in csv.reader(handle):
                rows += 1
                if len(row) < 4:
                    continue
                _, key, modified, _etag = row[:4]
                match = KEY_RE.search(key)
                if not match:
                    continue
                pmcid_num = int(match.group(1))
                version = int(match.group(2))
                if pmcid_num in positives:
                    current = positive_latest.get(pmcid_num)
                    if current is None or version > current[0]:
                        positive_latest[pmcid_num] = (version, key)
                    continue
                if modified < recent_modified_after:
                    continue
                add_reservoir_candidate(
                    background,
                    background_heap,
                    pmcid_num=pmcid_num,
                    version=version,
                    metadata_key=key,
                    hash_value=stable_hash(pmcid_num, seed + 1),
                    limit=candidate_limit,
                )
    return positive_latest, background, rows


def http_session() -> requests.Session:
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "NLP4Pheno bounded PMC benchmark/1.0"
        _thread_state.session = session
    return session


def fetch_metadata(bucket_url: str, item: tuple[int, int, str]) -> tuple[dict | None, str | None]:
    pmcid_num, version, key = item
    url = f"{bucket_url.rstrip('/')}/{quote(key, safe='/')}"
    try:
        response = http_session().get(url, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        if int(metadata.get("version")) != version or metadata.get("pmcid") != f"PMC{pmcid_num}":
            return None, "identifier_mismatch"
        if metadata.get("is_pmc_openaccess") is not True:
            return None, "not_open_access"
        if metadata.get("is_retracted") is True:
            return None, "retracted"
        return metadata, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def publication_year(metadata: dict) -> int | None:
    years = [int(value) for value in YEAR_RE.findall(str(metadata.get("citation") or ""))]
    return max(years) if years else None


def verified_sample(
    candidates: list[tuple[int, int, str]],
    *,
    count: int,
    bucket_url: str,
    workers: int,
    publication_year_min: int | None,
) -> tuple[list[tuple[int, dict]], dict[str, int]]:
    accepted: list[tuple[int, dict]] = []
    rejected: dict[str, int] = {}
    # ThreadPoolExecutor.map submits its whole iterable eagerly. Work in small
    # batches so reaching the requested cohort size does not trigger thousands
    # of unnecessary metadata requests from the overselected candidate pool.
    batch_size = max(workers * 4, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            results = executor.map(lambda x: fetch_metadata(bucket_url, x), batch)
            for item, result in zip(batch, results):
                metadata, error = result
                if error:
                    rejected[error] = rejected.get(error, 0) + 1
                    continue
                assert metadata is not None
                year = publication_year(metadata)
                if publication_year_min is not None and (
                    year is None or year < publication_year_min
                ):
                    rejected["publication_year"] = (
                        rejected.get("publication_year", 0) + 1
                    )
                    continue
                accepted.append((item[0], metadata))
                if len(accepted) >= count:
                    break
            if len(accepted) >= count:
                break
    if len(accepted) < count:
        raise RuntimeError(f"Only {len(accepted)} of {count} requested articles passed metadata checks")
    return accepted, rejected


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    network_path = Path(args.network)
    completion = json.loads(Path(args.inventory_completion).read_text(encoding="utf-8"))
    inventory_files = [Path(item["local_path"]) for item in completion["files"]]
    positives = read_positive_pmcids(network_path)
    overselect = max(args.cohort_size * args.overselect_factor, args.cohort_size)
    positive_latest, background, inventory_rows = scan_inventory(
        inventory_files,
        positives,
        seed=args.seed,
        candidate_limit=overselect,
        recent_modified_after=args.recent_modified_after,
    )

    positive_candidates = sorted(
        ((pmcid, version, key) for pmcid, (version, key) in positive_latest.items()),
        key=lambda item: stable_hash(item[0], args.seed),
    )[:overselect]
    background_candidates = [
        (pmcid, value[1], value[2])
        for pmcid, value in sorted(background.items(), key=lambda item: item[1][0])
    ]

    positive_sample, positive_rejected = verified_sample(
        positive_candidates,
        count=args.cohort_size,
        bucket_url=args.bucket_url,
        workers=args.workers,
        publication_year_min=None,
    )
    background_sample, background_rejected = verified_sample(
        background_candidates,
        count=args.cohort_size,
        bucket_url=args.bucket_url,
        workers=args.workers,
        publication_year_min=args.recent_publication_year_min,
    )

    cohort_rows: list[dict] = []
    for cohort, sample in (("known_positive", positive_sample), ("recent_background", background_sample)):
        for pmcid_num, metadata in sample:
            cohort_rows.append(
                {
                    "pmcid": f"PMC{pmcid_num}",
                    "cohort": cohort,
                    "article_version": f"PMC{pmcid_num}.{metadata['version']}",
                    "title": metadata.get("title") or "",
                    "citation": metadata.get("citation") or "",
                }
            )
    cohort_rows.sort(key=lambda item: item["pmcid"])
    identifiers = "".join(f"{row['pmcid']}\n" for row in cohort_rows)
    atomic_text(Path(args.output_ids), identifiers)

    cohort_output = Path(args.output_cohorts)
    cohort_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cohort_output.with_suffix(cohort_output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pmcid", "cohort", "article_version", "title", "citation"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(cohort_rows)
    temporary.replace(cohort_output)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "cohort_size": args.cohort_size,
        "known_positive_pmcids_in_network": len(positives),
        "known_positive_pmcids_in_inventory": len(positive_latest),
        "inventory_rows_scanned": inventory_rows,
        "recent_modified_after": args.recent_modified_after,
        "recent_publication_year_min": args.recent_publication_year_min,
        "positive_metadata_rejections": positive_rejected,
        "background_metadata_rejections": background_rejected,
        "inventory_version": completion.get("resolved_inventory_version"),
        "network_md5": hashlib.md5(network_path.read_bytes(), usedforsecurity=False).hexdigest(),
    }
    atomic_text(Path(args.output_summary), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(positive_sample)} known-positive and {len(background_sample)} "
        f"recent-background PMCIDs"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--inventory-completion", required=True)
    parser.add_argument("--output-ids", required=True)
    parser.add_argument("--output-cohorts", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--cohort-size", type=int, default=100)
    parser.add_argument("--overselect-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2509)
    parser.add_argument("--recent-modified-after", default="2025-01-01")
    parser.add_argument("--recent-publication-year-min", type=int, default=2025)
    parser.add_argument("--bucket-url", default="https://pmc-oa-opendata.s3.amazonaws.com")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
