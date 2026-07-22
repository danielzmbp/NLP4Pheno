#!/usr/bin/env python3
"""Measure a configured PMC scope query on deterministic benchmark cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_queries(config_path: Path) -> tuple[dict[str, Any], str, str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config.get("source", {})
    if "base_query" in source:
        base = str(source["base_query"])
        scope = str(source.get("scope_query", "")).strip()
        combined = f"({base}) AND ({scope})" if scope else base
    else:
        base = str(source.get("query", "open_access[filter]"))
        combined = base
    return source, base, combined


def read_cohorts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"pmcid", "cohort", "title"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain pmcid, cohort and title columns")
    for row in rows:
        value = row["pmcid"].upper()
        if not value.startswith("PMC") or not value[3:].isdigit():
            raise ValueError(f"Invalid PMCID in {path}: {row['pmcid']!r}")
        row["pmcid"] = value
    return rows


def retrying_session(retries: int, user_agent: str) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
    )
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def intersect_query(
    session: requests.Session,
    identifiers: list[str],
    query: str,
    *,
    base_url: str,
    batch_size: int,
    timeout: int,
    tool: str,
    email: str,
    api_key: str | None,
) -> tuple[set[str], int]:
    matched: set[str] = set()
    requests_made = 0
    delay = 0.12 if api_key else 0.36
    for start in range(0, len(identifiers), batch_size):
        batch = identifiers[start : start + batch_size]
        uid_query = "(" + " OR ".join(
            f"{value.removeprefix('PMC')}[uid]" for value in batch
        ) + ")"
        data = {
            "db": "pmc",
            "term": f"{uid_query} AND ({query})",
            "retmode": "json",
            "retmax": str(len(batch)),
            "tool": tool,
            "email": email,
        }
        if api_key:
            data["api_key"] = api_key
        response = session.post(base_url, data=data, timeout=timeout)
        response.raise_for_status()
        result = response.json().get("esearchresult")
        if not result:
            raise RuntimeError(f"Unexpected ESearch response: {response.text[:500]}")
        matched.update(f"PMC{value}" for value in result.get("idlist", []))
        requests_made += 1
        time.sleep(delay)
    return matched, requests_made


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    source, base_query, scoped_query = load_queries(args.config)
    rows = read_cohorts(args.cohorts)
    identifiers = [row["pmcid"] for row in rows]
    api_key = os.environ.get(str(source.get("ncbi_api_key_env", "NCBI_API_KEY")))
    email = os.environ.get(str(source.get("ncbi_email_env", "NCBI_EMAIL")), "")
    tool = str(source.get("ncbi_tool", "nlp4pheno_pmc_builder"))
    base_url = str(
        source.get(
            "esearch_url",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        )
    )
    session = retrying_session(args.retries, args.user_agent)
    eligible, base_requests = intersect_query(
        session,
        identifiers,
        base_query,
        base_url=base_url,
        batch_size=args.batch_size,
        timeout=args.timeout,
        tool=tool,
        email=email,
        api_key=api_key,
    )
    scoped, scope_requests = intersect_query(
        session,
        identifiers,
        scoped_query,
        base_url=base_url,
        batch_size=args.batch_size,
        timeout=args.timeout,
        tool=tool,
        email=email,
        api_key=api_key,
    )

    args.decisions.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with args.decisions.open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0]) + ["eligible", "in_scope", "decision"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            is_eligible = row["pmcid"] in eligible
            in_scope = row["pmcid"] in scoped
            decision = (
                "included"
                if in_scope
                else "not_esearch_discoverable"
                if not is_eligible
                else "outside_scope"
            )
            writer.writerow(
                {
                    **row,
                    "eligible": str(is_eligible).lower(),
                    "in_scope": str(in_scope).lower(),
                    "decision": decision,
                }
            )
            counts[row["cohort"]]["total"] += 1
            counts[row["cohort"]]["eligible"] += int(is_eligible)
            counts[row["cohort"]]["in_scope"] += int(in_scope)

    cohorts = {}
    for cohort, values in sorted(counts.items()):
        eligible_count = values["eligible"]
        cohorts[cohort] = {
            **dict(values),
            "scope_fraction_of_total": values["in_scope"] / values["total"],
            "scope_fraction_of_eligible": (
                values["in_scope"] / eligible_count if eligible_count else None
            ),
        }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "cohorts": str(args.cohorts.resolve()),
        "decisions": str(args.decisions.resolve()),
        "base_query": base_query,
        "scoped_query": scoped_query,
        "counts": cohorts,
        "requests": base_requests + scope_requests,
        "api_key_used": bool(api_key),
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--user-agent", default="NLP4Pheno PMC scope benchmark/1.0"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
