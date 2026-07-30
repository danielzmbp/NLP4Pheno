"""Conservative matching of annotated strain mentions to StrainInfo aliases.

The matcher deliberately does not use edit-distance or partial-ratio fallback.
It accepts a normalized exact alias or a complete, token-bounded designation
contained in a longer mention. Candidate SI-IDs remain ambiguous unless the
evidence resolves to exactly one strain.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl


NON_ALNUM = re.compile(r"[^A-Z0-9]+")
SEROTYPE_LIKE = re.compile(r"^[OHK]\d+(?:[OHK]\d+)*$")
SCIENTIFIC_NAME = re.compile(
    r"\b(?P<genus>[A-Za-z][A-Za-z-]+|[A-Za-z])\.?\s+"
    r"(?P<species>[A-Za-z][A-Za-z-]+)\b",
    re.IGNORECASE,
)
GENUS_BEFORE_DESIGNATION = re.compile(
    r"\b(?P<genus>[A-Za-z][A-Za-z-]{2,})\s+"
    r"(?=(?:strain\s+)?[A-Za-z0-9][A-Za-z0-9._#/+-]*\b)",
    re.IGNORECASE,
)
NON_TAXONOMIC_GENUS_WORDS = {
    "atcc",
    "cbs",
    "ccm",
    "cctcc",
    "ccug",
    "cect",
    "cip",
    "clinical",
    "culture",
    "dsm",
    "environmental",
    "iam",
    "ifm",
    "isolate",
    "jcm",
    "kctc",
    "laboratory",
    "lmg",
    "mtcc",
    "mutant",
    "nbrc",
    "nctc",
    "pcm",
    "reference",
    "sample",
    "strain",
    "type",
    "unknown",
}
NON_SPECIFIC_SPECIES_WORDS = {"sp", "spp", "species"}


def designation_key(value: str) -> str:
    """Return the punctuation-insensitive key used by the catalog snapshot."""
    return NON_ALNUM.sub("", value.upper())


def eligible_contained_key(key: str) -> bool:
    """Reject aliases too weak to recognize safely inside a longer mention."""
    return (
        len(key) >= 6
        and any(character.isalpha() for character in key)
        and any(character.isdigit() for character in key)
    )


def eligible_exact_key(key: str) -> bool:
    """Require enough identifying structure for an uncontextualized exact match."""
    return len(key) >= 6 and any(character.isalpha() for character in key)


def serotype_like_key(key: str) -> bool:
    """Reject serogroup/serotype labels that do not identify one strain."""
    return SEROTYPE_LIKE.fullmatch(key) is not None


def strong_catalog_support(row: Mapping[str, Any]) -> bool:
    """Return whether a short alias has authoritative catalogue provenance."""
    return any(
        row.get(field) is True
        for field in ("type_strain", "in_compact", "in_detailed_deposit")
    )


def _normalized_positions(value: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(value):
        if character.isascii() and character.isalnum():
            normalized.append(character.upper())
            positions.append(index)
    return "".join(normalized), positions


def has_complete_occurrence(mention: str, key: str) -> bool:
    """Check that a normalized alias occupies complete original-text tokens.

    This prevents a catalog alias such as ``ATCC 1`` from matching the prefix
    of ``ATCC 17978`` after punctuation and whitespace have been removed.
    """
    normalized, positions = _normalized_positions(mention)
    start = normalized.find(key)
    while start >= 0:
        end = start + len(key)
        original_start = positions[start]
        original_end = positions[end - 1] + 1
        left_ok = original_start == 0 or not mention[original_start - 1].isalnum()
        right_ok = original_end == len(mention) or not mention[original_end].isalnum()
        if left_ok and right_ok:
            return True
        start = normalized.find(key, start + 1)
    return False


def scientific_name_hint(value: str) -> tuple[str, str] | None:
    """Extract the first binomial-looking genus/species hint from text."""
    for match in SCIENTIFIC_NAME.finditer(value):
        genus = match.group("genus").lower()
        if genus not in NON_TAXONOMIC_GENUS_WORDS:
            return genus, match.group("species").lower()
    return None


def scientific_genus_hint(value: str) -> str | None:
    """Extract a full genus immediately before a strain-like designation."""
    match = GENUS_BEFORE_DESIGNATION.search(value)
    if match is None:
        return None
    genus = match.group("genus").lower()
    return None if genus in NON_TAXONOMIC_GENUS_WORDS else genus


def taxon_compatible(mention: str, taxon: str | None) -> bool | None:
    """Compare an explicit scientific-name hint with a catalog taxon.

    ``None`` means the mention did not provide enough taxonomic evidence.
    """
    hint = scientific_name_hint(mention)
    genus_hint = scientific_genus_hint(mention) if hint is None else None
    if (hint is None and genus_hint is None) or not taxon:
        return None
    taxon_tokens = re.findall(r"[A-Za-z][A-Za-z-]*", taxon.lower())
    if len(taxon_tokens) < 2:
        return None
    taxon_genus, taxon_species = taxon_tokens[:2]
    if hint is None:
        return genus_hint == taxon_genus
    hint_genus, hint_species = hint
    genus_matches = (
        hint_genus == taxon_genus
        if len(hint_genus) > 1
        else hint_genus[0] == taxon_genus[0]
    )
    if not genus_matches:
        return False
    if hint_species in NON_SPECIFIC_SPECIES_WORDS:
        return True
    if hint_species == taxon_species:
        return True
    # Same-genus species disagreements are often nomenclature changes. They
    # are not positive support, but should not overrule an authoritative
    # collection or type-strain designation.
    return None


def _prefer_taxonomic_support(
    mention: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reject contradictions and prefer positively corroborated candidates."""
    evidence = [(row, taxon_compatible(mention, row.get("taxon"))) for row in rows]
    supported = [row for row, compatible in evidence if compatible is True]
    if supported:
        return supported
    return [row for row, compatible in evidence if compatible is not False]


def _prefer_compact_authority(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer aliases asserted by the compact catalog over detailed cross-references."""
    compact = [row for row in rows if row.get("in_compact") is True]
    return compact or rows


def resolve_candidates(
    mention: str,
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve catalog candidate rows for one mention without guessing."""
    mention_key = designation_key(mention)
    rows = [dict(candidate) for candidate in candidates]
    exact = [row for row in rows if row["designation_key"] == mention_key]
    if exact:
        accepted = _prefer_taxonomic_support(mention, exact)
        accepted = [
            row
            for row in accepted
            if not serotype_like_key(row["designation_key"])
            and (
                eligible_exact_key(row["designation_key"])
                or strong_catalog_support(row)
                or taxon_compatible(mention, row.get("taxon")) is True
            )
        ]
        if not accepted:
            return {
                "status": "unmatched",
                "method": "exact_weak_rejected",
                "si_id": None,
                "taxon": None,
                "si_ids": sorted({int(row["si_id"]) for row in exact}),
                "designation_keys": sorted({row["designation_key"] for row in exact}),
            }
        method = "exact"
    else:
        bounded = [
            row
            for row in rows
            if not serotype_like_key(row["designation_key"])
            and (
                eligible_contained_key(row["designation_key"])
                or (
                    len(row["designation_key"]) >= 4
                    and any(
                        character.isalpha()
                        for character in row["designation_key"]
                    )
                    and any(
                        character.isdigit()
                        for character in row["designation_key"]
                    )
                    and (
                        strong_catalog_support(row)
                        or taxon_compatible(mention, row.get("taxon")) is True
                    )
                )
            )
            and has_complete_occurrence(mention, row["designation_key"])
        ]
        bounded = _prefer_taxonomic_support(mention, bounded)
        if not bounded:
            return {
                "status": "unmatched",
                "method": None,
                "si_id": None,
                "taxon": None,
                "si_ids": [],
                "designation_keys": [],
            }
        longest = max(len(row["designation_key"]) for row in bounded)
        accepted = [row for row in bounded if len(row["designation_key"]) == longest]
        method = "contained_bounded"

    accepted = _prefer_compact_authority(accepted)
    si_ids = sorted({int(row["si_id"]) for row in accepted})
    taxa = sorted({str(row["taxon"]) for row in accepted if row.get("taxon")})
    return {
        "status": "unique" if len(si_ids) == 1 else "ambiguous",
        "method": method,
        "si_id": si_ids[0] if len(si_ids) == 1 else None,
        "taxon": taxa[0] if len(si_ids) == 1 and len(taxa) == 1 else None,
        "si_ids": si_ids,
        "designation_keys": sorted({row["designation_key"] for row in accepted}),
    }


def resolve_mentions(mentions: Iterable[str], aliases: pl.DataFrame) -> pl.DataFrame:
    """Resolve unique mentions against a StrainInfo designation DataFrame."""
    mention_values = sorted({str(value) for value in mentions if value is not None})
    output_schema = {
        "mention": pl.String,
        "straininfo_status": pl.String,
        "straininfo_method": pl.String,
        "straininfo_si_id": pl.Int64,
        "straininfo_taxon": pl.String,
        "straininfo_si_ids": pl.List(pl.Int64),
        "straininfo_designation_keys": pl.List(pl.String),
    }
    if not mention_values:
        return pl.DataFrame(schema=output_schema)
    required = {"designation_key", "designation", "si_id", "taxon", "type_strain"}
    missing = required - set(aliases.columns)
    if missing:
        raise ValueError(f"StrainInfo aliases are missing columns: {sorted(missing)}")

    optional = {"in_compact", "in_detailed_deposit", "in_detailed_other"}
    selected = sorted(required | (optional & set(aliases.columns)))
    alias_frame = aliases.select(*selected).filter(
        pl.col("designation_key").is_not_null()
    )
    alias_keys = alias_frame.get_column("designation_key").unique().to_list()
    contained_keys = [key for key in alias_keys if eligible_contained_key(key)]
    mention_frame = pl.DataFrame(
        {
            "mention": mention_values,
            "mention_key": [designation_key(value) for value in mention_values],
        }
    )
    exact_aliases = alias_frame.rename({"designation_key": "exact_key"})
    exact = (
        mention_frame.join(
            exact_aliases,
            left_on="mention_key",
            right_on="exact_key",
            how="left",
        )
        .filter(pl.col("si_id").is_not_null())
        .with_columns(pl.col("mention_key").alias("designation_key"))
    )
    if contained_keys:
        extracted = (
            mention_frame.with_columns(
                pl.col("mention_key")
                .str.extract_many(contained_keys, overlapping=True)
                .alias("designation_key")
            )
            .explode("designation_key", empty_as_null=True)
            .filter(pl.col("designation_key").is_not_null())
            .join(alias_frame, on="designation_key", how="inner")
        )
    else:
        extracted = exact.head(0)
    candidates = pl.concat([exact, extracted], how="diagonal_relaxed")
    grouped: dict[str, list[dict[str, Any]]] = {}
    candidate_columns = [
        "mention",
        "designation_key",
        "designation",
        "si_id",
        "taxon",
        "type_strain",
        *sorted(optional & set(candidates.columns)),
    ]
    for row in candidates.select(*candidate_columns).iter_rows(named=True):
        grouped.setdefault(row["mention"], []).append(row)

    records = []
    for mention in mention_values:
        result = resolve_candidates(mention, grouped.get(mention, []))
        records.append(
            {
                "mention": mention,
                "straininfo_status": result["status"],
                "straininfo_method": result["method"],
                "straininfo_si_id": result["si_id"],
                "straininfo_taxon": result["taxon"],
                "straininfo_si_ids": result["si_ids"],
                "straininfo_designation_keys": result["designation_keys"],
            }
        )
    return pl.DataFrame(records, schema=output_schema)
