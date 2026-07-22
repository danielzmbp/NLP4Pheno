#!/usr/bin/env python3
"""Discover all PMCIDs matching a reusable-content query through ESearch."""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCRIPT_DIR = Path(snakemake.scriptdir) if "snakemake" in globals() else Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import redirect_snakemake_log  # noqa: E402


MAX_ESEARCH_RESULTS = 9_999


def session_with_retries(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.monotonic()


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def date_term(base_query: str, start: date, end: date) -> str:
    date_range = f"{start:%Y/%m/%d}:{end:%Y/%m/%d}[pmcrdat]"
    return f"({base_query}) AND {date_range}"


def esearch(
    session: requests.Session,
    limiter: RateLimiter,
    *,
    base_url: str,
    query: str,
    retmax: int,
    tool: str,
    email: str,
    api_key: str | None,
    timeout: int,
) -> dict[str, Any]:
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "tool": tool,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    limiter.wait()
    response = session.get(base_url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "esearchresult" not in payload:
        raise RuntimeError(f"Unexpected ESearch response: {payload}")
    return payload["esearchresult"]


def discover_ids(
    *,
    base_url: str,
    base_query: str,
    start_date: date,
    snapshot_date: date,
    tool: str,
    email: str,
    api_key: str | None,
    timeout: int,
    retries: int,
) -> tuple[list[int], dict[str, Any]]:
    if start_date > snapshot_date:
        raise ValueError("start_date cannot be after snapshot_date")

    session = session_with_retries(retries)
    limiter = RateLimiter(9.0 if api_key else 2.8)
    pending: list[tuple[date, date]] = [(start_date, snapshot_date)]
    accepted_ranges: list[dict[str, Any]] = []
    identifiers: set[int] = set()
    request_count = 0

    while pending:
        range_start, range_end = pending.pop()
        query = date_term(base_query, range_start, range_end)
        count_result = esearch(
            session,
            limiter,
            base_url=base_url,
            query=query,
            retmax=0,
            tool=tool,
            email=email,
            api_key=api_key,
            timeout=timeout,
        )
        request_count += 1
        count = int(count_result["count"])
        if count == 0:
            accepted_ranges.append(
                {"start": range_start.isoformat(), "end": range_end.isoformat(), "count": 0}
            )
            continue

        if count > MAX_ESEARCH_RESULTS:
            if range_start == range_end:
                raise RuntimeError(
                    f"ESearch returned {count:,} results for the single day "
                    f"{range_start}; it cannot be retrieved without an additional partition."
                )
            span = (range_end - range_start).days
            midpoint = range_start + timedelta(days=span // 2)
            # Stack right first so the final processing order remains chronological.
            pending.append((midpoint + timedelta(days=1), range_end))
            pending.append((range_start, midpoint))
            continue

        result = esearch(
            session,
            limiter,
            base_url=base_url,
            query=query,
            retmax=count,
            tool=tool,
            email=email,
            api_key=api_key,
            timeout=timeout,
        )
        request_count += 1
        returned = [int(value) for value in result.get("idlist", [])]
        if len(returned) != count:
            raise RuntimeError(
                f"ESearch count mismatch for {range_start}..{range_end}: "
                f"expected {count}, received {len(returned)}"
            )
        identifiers.update(returned)
        accepted_ranges.append(
            {
                "start": range_start.isoformat(),
                "end": range_end.isoformat(),
                "count": count,
            }
        )

    accepted_ranges.sort(key=lambda item: (item["start"], item["end"]))
    manifest = {
        "source": "ncbi_esearch",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "database": "pmc",
        "query": base_query,
        "date_field": "pmcrdat",
        "start_date": start_date.isoformat(),
        "snapshot_date": snapshot_date.isoformat(),
        "pmcid_count": len(identifiers),
        "request_count": request_count,
        "api_key_used": bool(api_key),
        "ranges": accepted_ranges,
    }
    return sorted(identifiers), manifest


def normalized_fixture_ids(path: Path) -> list[int]:
    values: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().upper()
        if not value or value.startswith("#"):
            continue
        if value.startswith("PMC"):
            value = value[3:]
        if not value.isdigit():
            raise ValueError(f"Invalid PMCID in fixture: {line!r}")
        values.add(int(value))
    return sorted(values)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    mode: str,
    fixture_path: Path | None,
    ids_output: Path,
    manifest_output: Path,
    base_url: str,
    query: str,
    start_date: str,
    snapshot_date: str,
    tool: str,
    email_env: str,
    api_key_env: str,
    timeout: int,
    retries: int,
) -> None:
    if mode in {"fixture", "pmcid_list"}:
        if fixture_path is None:
            raise ValueError(f"{mode} mode requires an explicit PMCID file")
        identifiers = normalized_fixture_ids(fixture_path)
        manifest = {
            "source": mode,
            "pmcid_file": str(fixture_path.resolve()),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "start_date": start_date,
            "snapshot_date": snapshot_date,
            "pmcid_count": len(identifiers),
        }
    else:
        email = os.environ.get(email_env, "").strip()
        if not email:
            raise RuntimeError(
                f"Set {email_env} to a contact email before using NCBI ESearch."
            )
        identifiers, manifest = discover_ids(
            base_url=base_url,
            base_query=query,
            start_date=parse_date(start_date),
            snapshot_date=parse_date(snapshot_date),
            tool=tool,
            email=email,
            api_key=os.environ.get(api_key_env) or None,
            timeout=timeout,
            retries=retries,
        )

    lines = "".join(f"PMC{identifier}\n" for identifier in identifiers)
    atomic_write_text(ids_output, lines)
    atomic_write_text(manifest_output, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(identifiers):,} unique PMCIDs to {ids_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("s3", "fixture", "pmcid_list"), required=True)
    parser.add_argument("--fixture")
    parser.add_argument("--ids-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--email-env", default="NCBI_EMAIL")
    parser.add_argument("--api-key-env", default="NCBI_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    run(
        mode=args.mode,
        fixture_path=Path(args.fixture) if args.fixture else None,
        ids_output=Path(args.ids_output),
        manifest_output=Path(args.manifest_output),
        base_url=args.base_url,
        query=args.query,
        start_date=args.start_date,
        snapshot_date=args.snapshot_date,
        tool=args.tool,
        email_env=args.email_env,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
    )


def snakemake_entrypoint() -> None:
    fixture = Path(str(snakemake.input.explicit[0])) if snakemake.input.explicit else None
    run(
        mode=str(snakemake.params.mode),
        fixture_path=fixture,
        ids_output=Path(str(snakemake.output.ids)),
        manifest_output=Path(str(snakemake.output.manifest)),
        base_url=str(snakemake.params.base_url),
        query=str(snakemake.params.query),
        start_date=str(snakemake.params.start_date),
        snapshot_date=str(snakemake.params.snapshot_date),
        tool=str(snakemake.params.tool),
        email_env=str(snakemake.params.email_env),
        api_key_env=str(snakemake.params.api_key_env),
        timeout=int(snakemake.params.request_timeout),
        retries=int(snakemake.params.retries),
    )


if "snakemake" in globals():
    redirect_snakemake_log(snakemake)
    snakemake_entrypoint()
elif __name__ == "__main__":
    cli()
