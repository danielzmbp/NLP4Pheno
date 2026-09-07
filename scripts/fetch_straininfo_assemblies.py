#!/usr/bin/env python3
"""Resolve one preferred genome assembly for each matched StrainInfo SI-ID."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ASSEMBLY_LEVEL_RANK = {
    "complete": 4,
    "chromosome": 3,
    "scaffold": 2,
    "contig": 1,
}
ASSEMBLY_SCHEMA = pa.schema(
    [
        pa.field("si_id", pa.int64(), nullable=False),
        pa.field("accession", pa.string(), nullable=False),
        pa.field("assembly_level", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("description", pa.string()),
        pa.field("selected", pa.bool_(), nullable=False),
        pa.field("response_sha256", pa.string(), nullable=False),
    ]
)


def parse_strain_response(si_id: int, payload: Any, response_sha256: str) -> list[dict]:
    """Validate a detailed StrainInfo response and retain genome sequences."""
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected one StrainInfo record for SI-ID {si_id}")
    strain = payload[0].get("strain", {})
    if int(strain.get("siID", -1)) != si_id:
        raise ValueError(f"StrainInfo returned the wrong SI-ID for {si_id}")
    records: list[dict] = []
    seen: set[str] = set()
    for sequence in strain.get("sequence", []):
        if str(sequence.get("type", "")).lower() != "genome":
            continue
        accession = str(sequence.get("accessionNumber") or "").strip()
        if not accession or accession in seen:
            continue
        seen.add(accession)
        year = sequence.get("year")
        records.append(
            {
                "si_id": si_id,
                "accession": accession,
                "assembly_level": str(sequence.get("assemblyLevel") or "").lower()
                or None,
                "year": int(year) if year is not None else None,
                "description": str(sequence.get("description") or "") or None,
                "selected": False,
                "response_sha256": response_sha256,
            }
        )
    if records:
        selected = max(
            records,
            key=lambda row: (
                ASSEMBLY_LEVEL_RANK.get(row["assembly_level"] or "", 0),
                row["year"] or 0,
                row["accession"],
            ),
        )
        selected["selected"] = True
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


def fetch_one(
    si_id: int,
    *,
    base_url: str,
    timeout: int,
    retries: int,
    user_agent: str,
) -> tuple[int, list[dict] | None, str | None]:
    try:
        session = http_session(retries, user_agent)
        response = session.get(
            f"{base_url.rstrip('/')}/v2/data/strain/max/{si_id}", timeout=timeout
        )
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        return si_id, parse_strain_response(si_id, response.json(), digest), None
    except Exception as error:  # Preserve the affected SI-ID in the audit summary.
        return si_id, None, f"{type(error).__name__}: {error}"


def read_si_ids(path: Path, column: str) -> list[int]:
    table = pq.read_table(path, columns=[column])
    return sorted(
        {
            int(value)
            for value in table.column(column).to_pylist()
            if value is not None
        }
    )


def atomic_parquet(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(records, schema=ASSEMBLY_SCHEMA),
        temporary,
        compression="zstd",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    session = http_session(args.retries, args.user_agent)
    status_response = session.get(f"{args.base_url.rstrip('/')}/", timeout=args.timeout)
    status_response.raise_for_status()
    service = status_response.json()
    version = str(service.get("version") or "")
    if args.expected_version and version != args.expected_version:
        raise ValueError(
            f"StrainInfo version changed: expected {args.expected_version}, received {version}"
        )

    si_ids = read_si_ids(args.input, args.id_column)
    if not si_ids:
        raise ValueError(f"No resolved SI-IDs in {args.input}")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(
            executor.map(
                lambda si_id: fetch_one(
                    si_id,
                    base_url=args.base_url,
                    timeout=args.timeout,
                    retries=args.retries,
                    user_agent=args.user_agent,
                ),
                si_ids,
            )
        )

    assemblies: list[dict] = []
    failures: dict[str, str] = {}
    without_genomes: list[int] = []
    for si_id, records, error in results:
        if error is not None:
            failures[str(si_id)] = error
        elif not records:
            without_genomes.append(si_id)
        else:
            assemblies.extend(records)
    failure_fraction = len(failures) / len(si_ids)
    if failure_fraction > args.max_failure_fraction:
        raise RuntimeError(
            f"StrainInfo detail failure fraction {failure_fraction:.2%} exceeds "
            f"{args.max_failure_fraction:.2%}: {failures}"
        )
    selected = sorted(
        (row for row in assemblies if row["selected"]),
        key=lambda row: row["si_id"],
    )
    if not selected:
        raise RuntimeError("No genome assemblies were resolved from matched SI-IDs")

    atomic_parquet(assemblies, args.output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_temp = args.manifest_output.with_suffix(args.manifest_output.suffix + ".tmp")
    manifest_temp.write_text(
        "".join(f"SI-ID{row['si_id']}/{row['accession']}\n" for row in selected),
        encoding="utf-8",
    )
    manifest_temp.replace(args.manifest_output)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "service_version": version,
        "selection_policy": "one genome per SI-ID: assembly level, then newest year, then accession",
        "si_ids": len(si_ids),
        "si_ids_with_genomes": len(selected),
        "si_ids_without_genomes": without_genomes,
        "detail_failures": failures,
        "genome_records": len(assemblies),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_temp = args.summary.with_suffix(args.summary.suffix + ".tmp")
    summary_temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_temp.replace(args.summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--id-column", default="straininfo_si_id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--base-url", default="https://api.straininfo.dsmz.de")
    parser.add_argument("--expected-version")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-failure-fraction", type=float, default=0.01)
    parser.add_argument(
        "--user-agent", default="NLP4Pheno StrainInfo assembly resolver/1.0"
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
