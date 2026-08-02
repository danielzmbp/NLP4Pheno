#!/usr/bin/env python3
"""Download versioned terminology snapshots and build a compact alias index."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml

from ontology_grounding import (
    normalize_formula_surface,
    normalize_surface,
    relaxed_surface,
)


SYNONYM_RE = re.compile(
    r'^synonym: "(?P<value>(?:\\.|[^"])*)" '
    r"(?P<scope>EXACT|BROAD|NARROW|RELATED)\b"
)
RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDFS_LABEL = "{http://www.w3.org/2000/01/rdf-schema#}label"
OWL_CLASS = "{http://www.w3.org/2002/07/owl#}Class"
OWL_DEPRECATED = "{http://www.w3.org/2002/07/owl#}deprecated"
OBOINOWL = "http://www.geneontology.org/formats/oboInOwl#"
OWL_SYNONYM_SCOPES = {
    f"{{{OBOINOWL}}}hasExactSynonym": "EXACT",
    f"{{{OBOINOWL}}}hasBroadSynonym": "BROAD",
    f"{{{OBOINOWL}}}hasNarrowSynonym": "NARROW",
    f"{{{OBOINOWL}}}hasRelatedSynonym": "RELATED",
}
ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

TERM_SCHEMA = pa.schema(
    [
        ("entity_type", pa.string()),
        ("ontology", pa.string()),
        ("concept_id", pa.string()),
        ("label", pa.string()),
    ]
)
ALIAS_SCHEMA = pa.schema(
    [
        ("entity_type", pa.string()),
        ("ontology", pa.string()),
        ("concept_id", pa.string()),
        ("concept_label", pa.string()),
        ("alias", pa.string()),
        ("normalized_alias", pa.string()),
        ("relaxed_alias", pa.string()),
        ("formula_alias", pa.string()),
        ("scope", pa.string()),
        ("auto_eligible", pa.bool_()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(
    url: str, destination: Path, *, refresh: bool = False
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        return {
            "requested_url": url,
            "resolved_url": None,
            "downloaded": False,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        metadata = {
            "requested_url": url,
            "resolved_url": response.url,
            "downloaded": True,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    temporary.replace(destination)
    metadata.update(
        {
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    )
    return metadata


def _decode_obo_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\"", '"').replace(r"\\", "\\")


def read_obo_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                break
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in {"ontology", "data-version", "date", "remark"}:
                    header.setdefault(key, value)
    return header


def iter_obo_terms(path: Path, prefix: str) -> Iterator[dict[str, Any]]:
    current: dict[str, Any] | None = None

    def emit(term: dict[str, Any] | None) -> dict[str, Any] | None:
        if (
            term
            and term.get("id", "").startswith(prefix)
            and term.get("name")
            and not term.get("obsolete", False)
        ):
            return term
        return None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                completed = emit(current)
                if completed:
                    yield completed
                current = {"synonyms": []}
                continue
            if line.startswith("[") and line.endswith("]"):
                completed = emit(current)
                if completed:
                    yield completed
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line == "is_obsolete: true":
                current["obsolete"] = True
            elif line.startswith("synonym: "):
                match = SYNONYM_RE.match(line)
                if match:
                    current["synonyms"].append(
                        (
                            _decode_obo_string(match.group("value")),
                            match.group("scope"),
                        )
                    )

    completed = emit(current)
    if completed:
        yield completed


def iter_rdfxml_terms(path: Path, prefix: str) -> Iterator[dict[str, Any]]:
    """Stream named OWL classes from an RDF/XML ontology export."""
    iri_prefix = f"http://purl.obolibrary.org/obo/{prefix.replace(':', '_')}"
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag != OWL_CLASS:
            continue
        iri = element.attrib.get(RDF_ABOUT, "")
        if not iri.startswith(iri_prefix):
            element.clear()
            continue
        suffix = iri.removeprefix(iri_prefix)
        concept_id = f"{prefix}{suffix}"
        labels = [
            (child.text or "").strip()
            for child in element
            if child.tag == RDFS_LABEL and (child.text or "").strip()
        ]
        deprecated = any(
            child.tag == OWL_DEPRECATED
            and (child.text or "").strip().casefold() == "true"
            for child in element
        )
        if labels and not deprecated:
            synonyms = [
                ((child.text or "").strip(), OWL_SYNONYM_SCOPES[child.tag])
                for child in element
                if child.tag in OWL_SYNONYM_SCOPES and (child.text or "").strip()
            ]
            yield {"id": concept_id, "name": labels[0], "synonyms": synonyms}
        element.clear()


def iter_mediadive_terms(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload.get("data", []):
        identifier = row.get("id")
        label = html.unescape(str(row.get("name") or ""))
        label = re.sub(r"<[^>]+>", "", label).strip()
        if identifier is None or not label:
            continue
        synonyms: list[tuple[str, str]] = []
        without_medium = re.sub(r"\s+MEDIUM$", "", label, flags=re.IGNORECASE)
        if without_medium != label:
            synonyms.append((without_medium, "EXACT"))
        canonical_acronym = re.fullmatch(
            r"([A-Z][A-Z0-9-]{1,10})(?:\s+\(([^)]+)\))?\s+MEDIUM",
            label,
            flags=re.IGNORECASE,
        )
        if canonical_acronym:
            synonyms.append((canonical_acronym.group(1), "EXACT"))
            if canonical_acronym.group(2):
                synonyms.append((canonical_acronym.group(2), "EXACT"))
        parenthetical_acronym = re.search(
            r"\(([A-Z][A-Z0-9-]{1,10})\)$", label
        )
        if parenthetical_acronym:
            synonyms.append((parenthetical_acronym.group(1), "EXACT"))
        yield {
            "id": f"MEDIADIVE:{identifier}",
            "name": label,
            "synonyms": synonyms,
        }


NCBI_EXACT_NAME_CLASSES = {
    "anamorph",
    "common name",
    "equivalent name",
    "genbank common name",
    "genbank synonym",
    "synonym",
    "teleomorph",
}


def iter_ncbi_taxdump_terms(path: Path) -> Iterator[dict[str, Any]]:
    """Stream scientific names and conservative synonyms from NCBI taxdump."""
    with tarfile.open(path, mode="r:gz") as archive:
        member = archive.extractfile("names.dmp")
        if member is None:
            raise ValueError(f"NCBI taxdump is missing names.dmp: {path}")

        current_tax_id: int | None = None
        scientific_name: str | None = None
        synonyms: list[tuple[str, str]] = []

        def emit() -> dict[str, Any] | None:
            if current_tax_id is None or not scientific_name:
                return None
            return {
                "id": f"NCBITaxon:{current_tax_id}",
                "name": scientific_name,
                "synonyms": synonyms,
            }

        for raw_line in member:
            fields = [
                field.strip() for field in raw_line.decode("utf-8").split("|")
            ]
            if len(fields) < 4:
                continue
            tax_id = int(fields[0])
            name = fields[1]
            name_class = fields[3].casefold()
            if current_tax_id is not None and tax_id < current_tax_id:
                raise ValueError("NCBI names.dmp is not sorted by tax_id")
            if tax_id != current_tax_id:
                completed = emit()
                if completed:
                    yield completed
                current_tax_id = tax_id
                scientific_name = None
                synonyms = []
            if name_class == "scientific name":
                scientific_name = name
            elif name_class in NCBI_EXACT_NAME_CLASSES and name:
                synonyms.append((name, "EXACT"))

        completed = emit()
        if completed:
            yield completed


def looks_like_chemical_formula(alias: str) -> bool:
    """Recognize formula-like ChEBI aliases without admitting bare acronyms."""
    compact = re.sub(r"\s+", "", alias).strip("[]")
    if re.fullmatch(r"(?:iso-|anteiso-)?C\d+:\d+", compact, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Z][a-z]?\((?:I|V|X)+\)", compact):
        return True
    if not re.search(r"\d|[a-z]", compact):
        return False
    compact = re.sub(r"(?:\^?\d*[+-])$", "", compact)
    tokens = re.findall(r"([A-Z][a-z]?)(?:\(\d+\)|\d*)", compact)
    rebuilt = "".join(
        match.group(0)
        for match in re.finditer(r"([A-Z][a-z]?)(?:\(\d+\)|\d*)", compact)
    )
    if not tokens or rebuilt != compact or any(token not in ELEMENT_SYMBOLS for token in tokens):
        return False
    if len(tokens) == 1:
        return compact in {"H2", "N2", "O2", "F2", "Cl2", "Br2", "I2"}
    return True


def alias_is_auto_eligible(
    ontology: str,
    alias: str,
    scope: str,
    source: dict[str, Any],
) -> bool:
    eligible_scopes = set(source.get("auto_eligible_scopes", ["LABEL", "EXACT"]))
    if scope in eligible_scopes:
        return True
    return (
        ontology == "CHEBI"
        and scope == "RELATED"
        and source.get("related_alias_policy") == "chemical_formula"
        and looks_like_chemical_formula(alias)
    )


class BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 25_000):
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".part")
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                self.temporary, self.schema, compression="zstd"
            )
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            table = pa.Table.from_pylist([], schema=self.schema)
            pq.write_table(table, self.temporary, compression="zstd")
        else:
            self.writer.close()
        self.temporary.replace(self.path)


def build_index(
    sources_path: Path,
    output_dir: Path,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    with sources_path.open() as handle:
        config = yaml.safe_load(handle)
    sources = config.get("sources", {})
    if not sources:
        raise ValueError(f"No sources configured in {sources_path}")

    raw_dir = output_dir / "raw"
    term_writer = BufferedParquetWriter(output_dir / "terms.parquet", TERM_SCHEMA)
    alias_writer = BufferedParquetWriter(output_dir / "aliases.parquet", ALIAS_SCHEMA)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(sources_path),
        "sources": {},
    }

    for ontology, source in sources.items():
        path = raw_dir / source["filename"]
        download_metadata = download_source(source["url"], path, refresh=refresh)
        source_format = source["format"]
        if source_format == "obo":
            header = read_obo_header(path)
            terms = iter_obo_terms(path, source["prefix"])
        elif source_format == "owl_rdfxml":
            header = {}
            terms = iter_rdfxml_terms(path, source["prefix"])
        elif source_format == "mediadive":
            header = {}
            terms = iter_mediadive_terms(path)
        elif source_format == "ncbi_taxdump":
            header = {}
            terms = iter_ncbi_taxdump_terms(path)
        else:
            raise ValueError(f"Unsupported source format {source_format!r}")

        term_count = 0
        alias_count = 0
        seen_aliases: set[tuple[str, str, str]] = set()
        for term in terms:
            term_count += 1
            for entity_type in source["entity_types"]:
                term_writer.append(
                    {
                        "entity_type": entity_type,
                        "ontology": ontology,
                        "concept_id": term["id"],
                        "label": term["name"],
                    }
                )
                aliases = [(term["name"], "LABEL"), *term.get("synonyms", [])]
                for alias, scope in aliases:
                    alias = alias.strip()
                    if not alias:
                        continue
                    key = (entity_type, term["id"], normalize_surface(alias))
                    if key in seen_aliases:
                        continue
                    seen_aliases.add(key)
                    alias_writer.append(
                        {
                            "entity_type": entity_type,
                            "ontology": ontology,
                            "concept_id": term["id"],
                            "concept_label": term["name"],
                            "alias": alias,
                            "normalized_alias": normalize_surface(alias),
                            "relaxed_alias": relaxed_surface(alias),
                            "formula_alias": (
                                normalize_formula_surface(alias)
                                if ontology == "CHEBI"
                                and scope == "RELATED"
                                and looks_like_chemical_formula(alias)
                                else None
                            ),
                            "scope": scope,
                            "auto_eligible": alias_is_auto_eligible(
                                ontology, alias, scope, source
                            ),
                        }
                    )
                    alias_count += 1

        manifest["sources"][ontology] = {
            **download_metadata,
            "format": source_format,
            "entity_types": source["entity_types"],
            "auto_eligible_scopes": source.get(
                "auto_eligible_scopes", ["LABEL", "EXACT"]
            ),
            "related_alias_policy": source.get("related_alias_policy"),
            "ontology_header": header,
            "terms": term_count,
            "aliases": alias_count,
        }
        print(
            f"{ontology}: {term_count:,} terms, {alias_count:,} aliases "
            f"({download_metadata['bytes'] / 1024 / 1024:.1f} MiB)"
        )

    term_writer.close()
    alias_writer.close()
    manifest["term_rows"] = term_writer.count
    manifest["alias_rows"] = alias_writer.count
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("resources/ontologies/sources.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/ontologies/runtime"),
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args.sources, args.output_dir, refresh=args.refresh)


if __name__ == "__main__":
    main()
