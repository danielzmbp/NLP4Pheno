#!/usr/bin/env python3
"""Resolve and download a PMC S3 inventory report."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from lxml import etree
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCRIPT_DIR = Path(snakemake.scriptdir) if "snakemake" in globals() else Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import redirect_snakemake_log  # noqa: E402


INVENTORY_VERSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}Z)/?$")


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


def object_url(bucket_url: str, key: str) -> str:
    return f"{bucket_url.rstrip('/')}/{quote(key.lstrip('/'), safe='/')}"


def list_inventory_versions(
    session: requests.Session,
    *,
    bucket_url: str,
    prefix: str,
    timeout: int,
) -> list[str]:
    continuation_token: str | None = None
    versions: set[str] = set()
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if continuation_token:
            params["continuation-token"] = continuation_token
        response = session.get(bucket_url, params=params, timeout=timeout)
        response.raise_for_status()
        root = etree.fromstring(response.content)
        for element in root.xpath("//*[local-name()='CommonPrefixes']/*[local-name()='Prefix']"):
            match = INVENTORY_VERSION_RE.search(element.text or "")
            if match:
                versions.add(match.group(1))
        truncated = root.xpath("string(//*[local-name()='IsTruncated'])").lower() == "true"
        if not truncated:
            break
        continuation_token = root.xpath(
            "string(//*[local-name()='NextContinuationToken'])"
        )
        if not continuation_token:
            raise RuntimeError("S3 inventory listing was truncated without a continuation token")
    return sorted(versions)


def choose_inventory_version(versions: list[str], requested: str, snapshot: str) -> str:
    if not versions:
        raise RuntimeError("No PMC inventory versions were found")
    if requested != "latest":
        normalized = requested.rstrip("/")
        if normalized not in versions:
            raise RuntimeError(
                f"Inventory version {normalized!r} is unavailable; newest is {versions[-1]}"
            )
        return normalized

    snapshot_day = date.fromisoformat(snapshot)
    eligible = [
        version
        for version in versions
        if date.fromisoformat(version[:10]) <= snapshot_day
    ]
    if not eligible:
        raise RuntimeError(f"No inventory version exists on or before {snapshot}")
    return eligible[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    session: requests.Session,
    *,
    url: str,
    destination: Path,
    timeout: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
    temporary.replace(destination)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_fixture_inventory(
    fixture_paths: list[Path], output_dir: Path, completion: Path, snapshot: str
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    for index, source in enumerate(fixture_paths):
        if not source.exists():
            raise FileNotFoundError(source)
        destination = data_dir / f"fixture_{index:04d}{''.join(source.suffixes)}"
        shutil.copy2(source, destination)
        files.append(
            {
                "key": str(source.resolve()),
                "local_path": str(destination.resolve()),
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    atomic_json(
        completion,
        {
            "source": "fixture",
            "snapshot_date": snapshot,
            "resolved_inventory_version": "fixture",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        },
    )


def fetch_inventory(
    *,
    output_dir: Path,
    completion: Path,
    bucket_url: str,
    inventory_prefix: str,
    requested_version: str,
    snapshot: str,
    timeout: int,
    retries: int,
) -> None:
    session = session_with_retries(retries)
    versions = list_inventory_versions(
        session, bucket_url=bucket_url, prefix=inventory_prefix, timeout=timeout
    )
    resolved_version = choose_inventory_version(versions, requested_version, snapshot)
    manifest_key = f"{inventory_prefix.rstrip('/')}/{resolved_version}/manifest.json"
    manifest_url = object_url(bucket_url, manifest_key)
    manifest_response = session.get(manifest_url, timeout=timeout)
    manifest_response.raise_for_status()
    manifest = manifest_response.json()

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []
    for index, file_entry in enumerate(manifest.get("files", [])):
        key = file_entry["key"]
        destination = data_dir / f"{index:04d}_{Path(key).name}"
        expected_size = file_entry.get("size")
        expected_md5 = file_entry.get("MD5checksum") or file_entry.get("md5Checksum")
        valid_existing = destination.exists()
        if valid_existing and expected_size is not None:
            valid_existing = destination.stat().st_size == int(expected_size)
        if valid_existing and expected_md5 and re.fullmatch(r"[0-9a-fA-F]{32}", expected_md5):
            valid_existing = md5_file(destination).lower() == expected_md5.lower()
        if not valid_existing:
            download_file(
                session,
                url=object_url(bucket_url, key),
                destination=destination,
                timeout=timeout,
            )
        if expected_size is not None and destination.stat().st_size != int(expected_size):
            raise RuntimeError(f"Size mismatch after downloading {key}")
        if expected_md5 and re.fullmatch(r"[0-9a-fA-F]{32}", expected_md5):
            if md5_file(destination).lower() != expected_md5.lower():
                raise RuntimeError(f"MD5 mismatch after downloading {key}")
        downloaded.append(
            {
                "key": key,
                "local_path": str(destination.resolve()),
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    if not downloaded:
        raise RuntimeError(f"Inventory manifest {manifest_key} contained no data files")

    local_manifest = output_dir / "source_manifest.json"
    atomic_json(local_manifest, manifest)
    atomic_json(
        completion,
        {
            "source": "pmc_s3_inventory",
            "snapshot_date": snapshot,
            "requested_inventory_version": requested_version,
            "resolved_inventory_version": resolved_version,
            "manifest_key": manifest_key,
            "manifest_url": manifest_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "files": downloaded,
        },
    )
    print(
        f"Downloaded {len(downloaded)} inventory parts for {resolved_version} "
        f"to {data_dir}"
    )


def run(
    *,
    mode: str,
    fixture_paths: list[Path],
    output_dir: Path,
    completion: Path,
    bucket_url: str,
    inventory_prefix: str,
    inventory_version: str,
    snapshot: str,
    timeout: int,
    retries: int,
) -> None:
    if mode == "fixture":
        copy_fixture_inventory(fixture_paths, output_dir, completion, snapshot)
    else:
        fetch_inventory(
            output_dir=output_dir,
            completion=completion,
            bucket_url=bucket_url,
            inventory_prefix=inventory_prefix,
            requested_version=inventory_version,
            snapshot=snapshot,
            timeout=timeout,
            retries=retries,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("s3", "fixture", "pmcid_list"), required=True)
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--bucket-url", required=True)
    parser.add_argument("--inventory-prefix", required=True)
    parser.add_argument("--inventory-version", default="latest")
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    run(
        mode=args.mode,
        fixture_paths=[Path(value) for value in args.fixture],
        output_dir=Path(args.output_dir),
        completion=Path(args.completion),
        bucket_url=args.bucket_url,
        inventory_prefix=args.inventory_prefix,
        inventory_version=args.inventory_version,
        snapshot=args.snapshot_date,
        timeout=args.timeout,
        retries=args.retries,
    )


def snakemake_entrypoint() -> None:
    run(
        mode=str(snakemake.params.mode),
        fixture_paths=[Path(str(value)) for value in snakemake.input.fixture],
        output_dir=Path(str(snakemake.output.complete)).parent,
        completion=Path(str(snakemake.output.complete)),
        bucket_url=str(snakemake.params.bucket_url),
        inventory_prefix=str(snakemake.params.inventory_prefix),
        inventory_version=str(snakemake.params.inventory_version),
        snapshot=str(snakemake.params.snapshot_date),
        timeout=int(snakemake.params.request_timeout),
        retries=int(snakemake.params.retries),
    )


if "snakemake" in globals():
    redirect_snakemake_log(snakemake)
    snakemake_entrypoint()
elif __name__ == "__main__":
    cli()
