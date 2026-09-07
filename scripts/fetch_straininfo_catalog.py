#!/usr/bin/env python3
"""Snapshot StrainInfo's public all-strains search catalog to Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STRAIN_SCHEMA = pa.schema(
    [
        pa.field("si_id", pa.int64(), nullable=False),
        pa.field("designations", pa.list_(pa.string()), nullable=False),
        pa.field("taxon", pa.string()),
        pa.field("type_strain", pa.bool_(), nullable=False),
        pa.field("country_code", pa.string()),
        pa.field("status_code", pa.int8(), nullable=False),
    ]
)
DESIGNATION_SCHEMA = pa.schema(
    [
        pa.field("designation_key", pa.string(), nullable=False),
        pa.field("designation", pa.string(), nullable=False),
        pa.field("si_id", pa.int64(), nullable=False),
        pa.field("taxon", pa.string()),
        pa.field("type_strain", pa.bool_(), nullable=False),
    ]
)


def designation_key(value: str) -> str:
    """Normalize spacing/punctuation without performing fuzzy matching."""
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def parse_page_rows(rows: list[list[Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        if len(row) != 6:
            raise ValueError(f"Unexpected StrainInfo compact row: {row!r}")
        si_id, designations, taxon, type_strain, country_code, status_code = row
        if not isinstance(designations, list):
            raise ValueError(f"StrainInfo designations are not a list for SI-ID {si_id}")
        records.append(
            {
                "si_id": int(si_id),
                "designations": [str(value) for value in designations if str(value)],
                "taxon": str(taxon) or None,
                "type_strain": bool(type_strain),
                "country_code": str(country_code) or None,
                "status_code": int(status_code),
            }
        )
    return records


def designation_records(strains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    seen: set[tuple[str, int]] = set()
    for strain in strains:
        for designation in strain["designations"]:
            key = designation_key(designation)
            identity = (key, strain["si_id"])
            if not key or identity in seen:
                continue
            seen.add(identity)
            records.append(
                {
                    "designation_key": key,
                    "designation": designation,
                    "si_id": strain["si_id"],
                    "taxon": strain["taxon"],
                    "type_strain": strain["type_strain"],
                }
            )
    return records


def http_session(retries: int, user_agent: str) -> requests.Session:
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
    session.headers["User-Agent"] = user_agent
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def atomic_parquet(records: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression="zstd")
    temporary.replace(path)


def fetch_catalog(
    *,
    base_url: str,
    expected_version: str | None,
    timeout: int,
    retries: int,
    user_agent: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = http_session(retries, user_agent)
    root_response = session.get(f"{base_url.rstrip('/')}/", timeout=timeout)
    root_response.raise_for_status()
    service = root_response.json()
    version = str(service.get("version") or "")
    if not version:
        raise ValueError("StrainInfo status response has no version")
    if expected_version and version != expected_version:
        raise ValueError(
            f"StrainInfo version changed: expected {expected_version}, received {version}"
        )
    if service.get("maintenance", {}).get("status"):
        raise RuntimeError(f"StrainInfo is under maintenance: {service['maintenance']}")

    strains: list[dict[str, Any]] = []
    page_hashes: list[str] = []
    expected_count: int | None = None
    page = 0
    while True:
        url = f"{base_url.rstrip('/')}/service/search/strain/all/{page}"
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        page_hashes.append(hashlib.sha256(response.content).hexdigest())
        payload = response.json()
        count = int(payload["count"])
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise RuntimeError("StrainInfo result count changed while paging")
        strains.extend(parse_page_rows(payload["data"]))
        next_page = payload.get("next")
        if next_page is None:
            break
        if int(next_page) != page + 1:
            raise RuntimeError(f"Unexpected StrainInfo next page: {next_page}")
        page = int(next_page)

    if expected_count is None or len(strains) != expected_count:
        raise RuntimeError(
            f"StrainInfo returned {len(strains):,} strains; expected {expected_count}"
        )
    ids = [record["si_id"] for record in strains]
    if len(ids) != len(set(ids)):
        raise RuntimeError("StrainInfo compact catalog contains duplicate SI-IDs")
    metadata = {
        "service_version": version,
        "base_url": base_url.rstrip("/"),
        "pages": page + 1,
        "page_sha256": page_hashes,
        "strain_count": len(strains),
    }
    return strains, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strains-output", required=True, type=Path)
    parser.add_argument("--designations-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--base-url", default="https://api.straininfo.dsmz.de")
    parser.add_argument("--expected-version")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--user-agent", default="NLP4Pheno StrainInfo snapshot builder/1.0"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strains, summary = fetch_catalog(
        base_url=args.base_url,
        expected_version=args.expected_version,
        timeout=args.timeout,
        retries=args.retries,
        user_agent=args.user_agent,
    )
    designations = designation_records(strains)
    key_to_ids: dict[str, set[int]] = {}
    for record in designations:
        key_to_ids.setdefault(record["designation_key"], set()).add(record["si_id"])
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "designation_count": len(designations),
            "designation_key_count": len(key_to_ids),
            "ambiguous_designation_keys": sum(
                len(si_ids) > 1 for si_ids in key_to_ids.values()
            ),
            "matching_policy": "exact normalized designation; ambiguous keys retained",
        }
    )
    atomic_parquet(strains, STRAIN_SCHEMA, args.strains_output)
    atomic_parquet(designations, DESIGNATION_SCHEMA, args.designations_output)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
