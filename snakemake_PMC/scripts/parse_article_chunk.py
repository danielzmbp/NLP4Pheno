#!/usr/bin/env python3
"""Download, parse and filter one stable chunk of versioned PMC JATS XML."""

import argparse
import csv
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCRIPT_DIR = Path(snakemake.scriptdir) if "snakemake" in globals() else Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    ARTICLE_SCHEMA,
    CORPUS_SCHEMA,
    MANIFEST_SCHEMA,
    ParseOptions,
    empty_table,
    parse_jats,
    records_table,
    redirect_snakemake_log,
)


_thread_state = threading.local()


def http_session(retries: int, user_agent: str) -> requests.Session:
    session = getattr(_thread_state, "session", None)
    if session is not None:
        return session
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
    _thread_state.session = session
    return session


def fetch_object(
    url: str, *, timeout: int, retries: int, user_agent: str
) -> tuple[bytes, dict[str, str]]:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        return path.read_bytes(), {}
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported XML URL scheme: {parsed.scheme!r}")
    session = http_session(retries, user_agent)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content, dict(response.headers)


def https_s3_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https", "file"}:
        return url
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Unsupported PMC object URL: {url!r}")
    result = f"https://{parsed.netloc}.s3.amazonaws.com/{parsed.path.lstrip('/')}"
    if parsed.query:
        result += f"?{parsed.query}"
    return result


def verified_metadata(
    manifest: dict[str, Any], *, timeout: int, retries: int, user_agent: str
) -> dict[str, Any]:
    metadata_url = manifest.get("metadata_url")
    if not metadata_url:
        raise ValueError("Article manifest has no metadata_url for verification")
    content, headers = fetch_object(
        str(metadata_url), timeout=timeout, retries=retries, user_agent=user_agent
    )
    expected_etag = str(manifest.get("source_etag") or "").strip('"')
    observed_etag = headers.get("ETag", headers.get("etag", "")).strip('"')
    if expected_etag and observed_etag != expected_etag:
        raise ValueError(
            f"Metadata ETag changed after inventory: expected {expected_etag}, "
            f"received {observed_etag or '<missing>'}"
        )
    metadata = json.loads(content)
    if str(metadata.get("pmcid")) != str(manifest["pmcid"]):
        raise ValueError("PMC metadata PMCID does not match the article manifest")
    if int(metadata.get("version")) != int(manifest["version"]):
        raise ValueError("PMC metadata version does not match the article manifest")
    if metadata.get("is_pmc_openaccess") is not True:
        raise ValueError("Selected article version is no longer in the PMC Open Access Subset")
    if metadata.get("is_retracted") is True:
        raise ValueError("Selected article version is marked as retracted")
    return metadata


def process_article(
    manifest: dict[str, Any],
    *,
    options: ParseOptions,
    timeout: int,
    retries: int,
    user_agent: str,
    verify_metadata_json: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    try:
        metadata: dict[str, Any] | None = None
        xml_source = str(manifest["xml_url"])
        if verify_metadata_json:
            metadata = verified_metadata(
                manifest, timeout=timeout, retries=retries, user_agent=user_agent
            )
            xml_source = https_s3_url(str(metadata["xml_url"]))
        xml_bytes, _ = fetch_object(
            xml_source,
            timeout=timeout,
            retries=retries,
            user_agent=user_agent,
        )
        if metadata:
            expected_md5 = parse_qs(urlparse(str(metadata["xml_url"])).query).get(
                "md5", [None]
            )[0]
            if not expected_md5:
                raise ValueError("PMC metadata XML URL contains no MD5 digest")
            observed_md5 = hashlib.md5(xml_bytes, usedforsecurity=False).hexdigest()
            if observed_md5.lower() != expected_md5.lower():
                raise ValueError(
                    f"XML MD5 mismatch: expected {expected_md5}, received {observed_md5}"
                )
        parse_manifest = dict(manifest)
        parse_manifest["xml_url"] = xml_source
        article, rows = parse_jats(xml_bytes, parse_manifest, options)
        if metadata:
            for article_key, metadata_key in (
                ("pmid", "pmid"),
                ("doi", "doi"),
                ("title", "title"),
                ("citation", "citation"),
                ("license_code", "license_code"),
                ("is_pmc_openaccess", "is_pmc_openaccess"),
                ("is_retracted", "is_retracted"),
            ):
                value = metadata.get(metadata_key)
                if value is not None:
                    article[article_key] = str(value) if article_key == "pmid" else value
        return article, rows, None
    except Exception as exc:  # Each failed article is reported with its provenance.
        return None, [], f"{type(exc).__name__}: {exc}"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    manifest_path: Path,
    sentence_output: Path,
    article_output: Path,
    failure_output: Path,
    stats_output: Path,
    options: ParseOptions,
    workers: int,
    timeout: int,
    retries: int,
    verify_metadata_json: bool,
    user_agent: str,
) -> None:
    manifest_file = pq.ParquetFile(manifest_path)
    if not manifest_file.schema_arrow.equals(MANIFEST_SCHEMA, check_metadata=False):
        raise ValueError(f"Unexpected manifest schema in {manifest_path}")
    manifests = manifest_file.read().to_pylist()
    manifests.sort(key=lambda item: int(item["pmcid_num"]))

    for path in (sentence_output, article_output, failure_output, stats_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    sentence_temp = sentence_output.with_suffix(sentence_output.suffix + ".tmp")
    article_temp = article_output.with_suffix(article_output.suffix + ".tmp")
    failure_temp = failure_output.with_suffix(failure_output.suffix + ".tmp")

    sentence_writer = pq.ParquetWriter(
        sentence_temp,
        CORPUS_SCHEMA,
        compression="zstd",
        write_statistics=True,
    )
    article_writer = pq.ParquetWriter(
        article_temp,
        ARTICLE_SCHEMA,
        compression="zstd",
        write_statistics=True,
    )
    article_buffer: list[dict[str, Any]] = []
    sentence_buffer: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    exclusion_counts: Counter[str] = Counter()
    long_rows = 0
    included = 0
    corpus_rows = 0
    paragraph_count = 0

    def work(item: dict[str, Any]):
        return process_article(
            item,
            options=options,
            timeout=timeout,
            retries=retries,
            user_agent=user_agent,
            verify_metadata_json=verify_metadata_json,
        )

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            # executor.map preserves manifest order, making Parquet output deterministic.
            for manifest, result in zip(manifests, executor.map(work, manifests)):
                article, rows, error = result
                if error:
                    failures.append(
                        {
                            "pmcid": str(manifest["pmcid"]),
                            "article_version": str(manifest["article_version"]),
                            "xml_url": str(manifest["xml_url"]),
                            "error": error,
                        }
                    )
                    continue
                assert article is not None
                article_buffer.append(article)
                if article["included"]:
                    included += 1
                else:
                    exclusion_counts[str(article["exclusion_reason"])] += 1
                paragraph_count += int(article["paragraphs"])
                for row in rows:
                    if len(str(row["text"])) > options.max_text_chars:
                        long_rows += 1
                sentence_buffer.extend(rows)
                corpus_rows += len(rows)

                if len(article_buffer) >= 2_000:
                    article_writer.write_table(records_table(article_buffer, ARTICLE_SCHEMA))
                    article_buffer.clear()
                if len(sentence_buffer) >= 50_000:
                    sentence_writer.write_table(records_table(sentence_buffer, CORPUS_SCHEMA))
                    sentence_buffer.clear()

        if article_buffer:
            article_writer.write_table(records_table(article_buffer, ARTICLE_SCHEMA))
        if sentence_buffer:
            sentence_writer.write_table(records_table(sentence_buffer, CORPUS_SCHEMA))
    finally:
        article_writer.close()
        sentence_writer.close()

    # ParquetWriter produces valid empty files, but explicitly verify their schemas.
    if not manifests:
        pq.write_table(empty_table(CORPUS_SCHEMA), sentence_temp, compression="zstd")
        pq.write_table(empty_table(ARTICLE_SCHEMA), article_temp, compression="zstd")

    with failure_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pmcid", "article_version", "xml_url", "error"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(failures)

    sentence_temp.replace(sentence_output)
    article_temp.replace(article_output)
    failure_temp.replace(failure_output)

    attempted = len(manifests)
    failure_fraction = len(failures) / attempted if attempted else 0.0
    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk": manifest_path.stem,
        "articles": {
            "attempted": attempted,
            "parsed": attempted - len(failures),
            "included": included,
            "excluded": sum(exclusion_counts.values()),
            "failed": len(failures),
            "failure_fraction": failure_fraction,
        },
        "paragraphs": paragraph_count,
        "corpus_rows": corpus_rows,
        "rows_over_max_text_chars": long_rows,
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
    }
    atomic_json(stats_output, stats)
    print(
        f"{manifest_path.stem}: {attempted:,} attempted, {included:,} included, "
        f"{len(failures):,} failed, {corpus_rows:,} corpus rows"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--articles", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--verify-metadata-json", action="store_true")
    parser.add_argument("--user-agent", default="NLP4Pheno PMC corpus builder/2.0")
    parser.add_argument("--max-text-chars", type=int, default=512)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--publication-year-min", type=int, default=1950)
    parser.add_argument("--allowed-language", action="append", default=["en"])
    parser.add_argument("--exclude-section", action="append", default=[])
    parser.add_argument("--exclude-unknown-language", action="store_true")
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    options = ParseOptions(
        max_text_chars=args.max_text_chars,
        min_text_chars=args.min_text_chars,
        publication_year_min=args.publication_year_min,
        allowed_languages=tuple(args.allowed_language),
        include_unknown_language=not args.exclude_unknown_language,
        excluded_section_patterns=tuple(args.exclude_section),
    )
    run(
        manifest_path=Path(args.manifest),
        sentence_output=Path(args.sentences),
        article_output=Path(args.articles),
        failure_output=Path(args.failures),
        stats_output=Path(args.stats),
        options=options,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        verify_metadata_json=args.verify_metadata_json,
        user_agent=args.user_agent,
    )


def snakemake_entrypoint() -> None:
    options = ParseOptions.from_mapping(
        {
            "max_text_chars": snakemake.params.max_text_chars,
            "min_text_chars": snakemake.params.min_text_chars,
            "publication_year_min": snakemake.params.publication_year_min,
            "allowed_languages": list(snakemake.params.allowed_languages),
            "include_unknown_language": snakemake.params.include_unknown_language,
            "excluded_section_patterns": list(
                snakemake.params.excluded_section_patterns
            ),
        }
    )
    run(
        manifest_path=Path(str(snakemake.input.manifest)),
        sentence_output=Path(str(snakemake.output.sentences)),
        article_output=Path(str(snakemake.output.articles)),
        failure_output=Path(str(snakemake.output.failures)),
        stats_output=Path(str(snakemake.output.stats)),
        options=options,
        workers=int(snakemake.threads),
        timeout=int(snakemake.params.request_timeout),
        retries=int(snakemake.params.retries),
        verify_metadata_json=bool(snakemake.params.verify_metadata_json),
        user_agent=str(snakemake.params.user_agent),
    )


if "snakemake" in globals():
    redirect_snakemake_log(snakemake)
    snakemake_entrypoint()
elif __name__ == "__main__":
    cli()
