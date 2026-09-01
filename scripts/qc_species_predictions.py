#!/usr/bin/env python3
"""Conservatively repair or quarantine SPECIES relation endpoints.

The raw grounded prediction table is never modified.  This stage writes a
separate accepted table, a quarantine table, a context-level audit, and a JSON
summary. Repairs are deliberately bounded to adjacent taxonomic tokens.
Uniquely grounded repairs receive NCBI Taxonomy identifiers; a uniquely
reconstructed, well-formed surface may instead remain an auditable text node.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pyarrow.dataset as ds

from ontology_grounding import (
    ground_mentions,
    load_candidate_indexes,
    normalize_surface,
    relaxed_surface,
)


SPECIES = "SPECIES"
NCBI_TAXON = "NCBITAXON"
GUARDED_SAME_TAXON_RELATIONS = {"INFECTS", "INHABITS", "SYMBIONT_OF"}
CONTEXT_KEYS = ["text", "start", "end", "word_qc_group"]
_GENUS = r"(?:Candidatus\s+)?[A-Z][a-z][A-Za-z-]*"
_ABBREVIATED_GENUS = r"[A-Z][a-z]{0,7}\."
_EPITHET = r"[A-Za-z][A-Za-z-]+"
_INFRA = (
    rf"(?:\s+(?:subsp\.?|ssp\.?|var\.?|pv\.?|ser\.?|serovar)\s+{_EPITHET}"
    rf"|\s+f\.\s*sp\.\s+{_EPITHET}"
    r"|\s+s\.\s*s\."
    rf"|\s+[A-Z][A-Za-z-]+)?"
)
_FULL_SPECIES_RE = re.compile(rf"^(?P<genus>{_GENUS})\s+(?P<epithet>{_EPITHET})(?P<infra>{_INFRA})$")
_ABBREVIATED_SPECIES_RE = re.compile(
    rf"^(?P<genus>{_ABBREVIATED_GENUS})\s*(?P<epithet>{_EPITHET})(?P<infra>{_INFRA})$"
)
_SAFE_ABBREVIATED_SPECIES_RE = re.compile(
    rf"^(?P<genus>{_ABBREVIATED_GENUS})\s*"
    rf"(?P<epithet>smeg\.)(?P<infra>{_INFRA})$",
    re.IGNORECASE,
)
_INITIAL_WITHOUT_PERIOD_SPECIES_RE = re.compile(
    rf"^(?P<genus>[A-Z])\s+(?P<epithet>{_EPITHET})(?P<infra>{_INFRA})$"
)
_UNSPECIFIED_SPECIES_RE = re.compile(rf"^(?:{_GENUS}|{_ABBREVIATED_GENUS})\s+spp?\.?$")
_GENUS_ONLY_RE = re.compile(rf"^{_GENUS}$")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]*|\.")


@dataclass(frozen=True)
class TaxonMatch:
    ontology: str
    concept_id: str
    concept_label: str
    alias: str
    scope: str
    method: str
    confidence: float


@dataclass(frozen=True)
class Expansion:
    surface: str
    start: int
    end: int


def canonical_taxon_surface(value: str) -> str:
    """Normalize spacing without lowercasing the biological surface form."""
    value = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:")
    value = re.sub(r"\s*\.\s*", ". ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:-1] if value.endswith(" .") else value


def parse_species_surface(value: str) -> tuple[str, str, str] | None:
    """Return (genus, epithet, infraspecific tail) for a binomial surface."""
    value = canonical_taxon_surface(value)
    match = (
        _FULL_SPECIES_RE.fullmatch(value)
        or _ABBREVIATED_SPECIES_RE.fullmatch(value)
        or _SAFE_ABBREVIATED_SPECIES_RE.fullmatch(value)
        or _INITIAL_WITHOUT_PERIOD_SPECIES_RE.fullmatch(value)
    )
    if not match:
        return None
    return match.group("genus"), match.group("epithet"), match.group("infra") or ""


def is_well_formed_species_surface(value: str) -> bool:
    value = canonical_taxon_surface(value)
    return bool(parse_species_surface(value) or _UNSPECIFIED_SPECIES_RE.fullmatch(value))


def abbreviation_key(value: str) -> tuple[str, str, str, bool] | None:
    parsed = parse_species_surface(value)
    if not parsed:
        return None
    genus, epithet, infra = parsed
    if not (genus.endswith(".") or len(genus) == 1):
        return None
    genus_prefix = genus.rstrip(".").casefold()
    epithet_is_prefix = epithet.endswith(".")
    return (
        genus_prefix,
        epithet.rstrip(".").casefold(),
        normalize_surface(infra),
        epithet_is_prefix,
    )


def is_safe_ungrounded_repair(value: str) -> bool:
    """Reject accidental repairs such as ordinary prose `E was`."""
    parsed = parse_species_surface(value)
    if not parsed:
        return bool(_UNSPECIFIED_SPECIES_RE.fullmatch(canonical_taxon_surface(value)))
    genus, _, _ = parsed
    return genus.endswith(".") or len(genus.split()[-1]) > 1


def candidate_expansions(text: str, start: int, end: int) -> list[Expansion]:
    """Generate taxonomic names by extending at most two tokens on either side."""
    if not text or start < 0 or end <= start or end > len(text):
        return []
    tokens = list(_TOKEN_RE.finditer(text))
    overlapping = [
        index
        for index, token in enumerate(tokens)
        if token.end() > start and token.start() < end
    ]
    if not overlapping:
        return []
    first, last = min(overlapping), max(overlapping)
    expansions: dict[tuple[str, int, int], Expansion] = {}
    for left in range(max(0, first - 2), first + 1):
        for right in range(last, min(len(tokens) - 1, last + 3) + 1):
            candidate_start = tokens[left].start()
            candidate_end = tokens[right].end()
            surface = canonical_taxon_surface(text[candidate_start:candidate_end])
            if not is_well_formed_species_surface(surface):
                continue
            if candidate_start > start or candidate_end < end:
                continue
            expansion = Expansion(surface, candidate_start, candidate_end)
            expansions[(surface, candidate_start, candidate_end)] = expansion
    return sorted(
        expansions.values(),
        key=lambda item: (item.end - item.start, item.start, item.surface.casefold()),
    )


def _direct_matches(
    surfaces: Iterable[str], aliases_file: Path, *, chunk_size: int = 5000
) -> dict[str, list[TaxonMatch]]:
    surfaces = sorted({canonical_taxon_surface(value) for value in surfaces if value})
    matches: dict[str, list[TaxonMatch]] = defaultdict(list)
    supported = {SPECIES}
    for offset in range(0, len(surfaces), chunk_size):
        selected = surfaces[offset : offset + chunk_size]
        mentions = [
            {
                "entity_type": SPECIES,
                "surface": surface,
                "normalized_surface": normalize_surface(surface),
                "relaxed_surface": relaxed_surface(surface),
                "formula_surface": "",
                "expanded_form": None,
            }
            for surface in selected
        ]
        exact, relaxed, formula = load_candidate_indexes(aliases_file, mentions)
        for row in ground_mentions(mentions, exact, relaxed, formula, supported):
            if row["status"] != "matched" or row["ontology"] != NCBI_TAXON:
                continue
            method = str(row["match_method"] or "direct_exact")
            confidence = 1.0 if method.endswith("_exact") else 0.97
            matches[row["surface"]].append(
                TaxonMatch(
                    ontology=row["ontology"],
                    concept_id=row["concept_id"],
                    concept_label=row["concept_label"],
                    alias=row["matched_alias"],
                    scope=row["alias_scope"],
                    method=method,
                    confidence=confidence,
                )
            )
    return matches


def _abbreviation_matches(
    surfaces: Iterable[str], aliases_file: Path
) -> dict[str, list[TaxonMatch]]:
    requested_exact: dict[
        tuple[str, str], list[tuple[str, str]]
    ] = defaultdict(list)
    requested_prefix: dict[
        tuple[str, str], list[tuple[str, str, str]]
    ] = defaultdict(list)
    requested_labels: dict[str, set[str]] = defaultdict(set)
    for surface in surfaces:
        canonical = canonical_taxon_surface(surface)
        key = abbreviation_key(canonical)
        if key:
            genus_prefix, epithet, infra, epithet_is_prefix = key
            if epithet_is_prefix:
                requested_prefix[(epithet[0], infra)].append(
                    (genus_prefix, epithet, canonical)
                )
            else:
                requested_exact[(epithet, infra)].append((genus_prefix, canonical))
        elif _GENUS_ONLY_RE.fullmatch(canonical):
            requested_labels[canonical.casefold()].add(canonical)
    if not requested_exact and not requested_prefix and not requested_labels:
        return {}

    expression = (
        ds.field("auto_eligible")
        & (ds.field("entity_type") == SPECIES)
        & (ds.field("ontology") == NCBI_TAXON)
    )
    scanner = ds.dataset(aliases_file, format="parquet").scanner(
        columns=["ontology", "concept_id", "concept_label", "alias", "scope"],
        filter=expression,
        batch_size=65536,
    )
    matches_by_surface: dict[str, dict[str, TaxonMatch]] = defaultdict(dict)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            abbreviation_match = TaxonMatch(
                ontology=row["ontology"],
                concept_id=row["concept_id"],
                concept_label=row["concept_label"],
                alias=row["alias"],
                scope=row["scope"],
                method="species_qc_abbreviation",
                confidence=0.98,
            )
            label = canonical_taxon_surface(row["concept_label"])
            if row["scope"] == "LABEL":
                for surface in requested_labels.get(label.casefold(), ()):
                    matches_by_surface[surface][abbreviation_match.concept_id] = TaxonMatch(
                        ontology=abbreviation_match.ontology,
                        concept_id=abbreviation_match.concept_id,
                        concept_label=abbreviation_match.concept_label,
                        alias=abbreviation_match.alias,
                        scope=abbreviation_match.scope,
                        method="species_qc_preferred_label",
                        confidence=1.0,
                    )
            parsed = parse_species_surface(canonical_taxon_surface(row["alias"]))
            if not parsed:
                continue
            genus, epithet, infra = parsed
            genus_word = genus.split()[-1].casefold()
            epithet_word = epithet.casefold()
            infra_key = normalize_surface(infra)
            for genus_prefix, surface in requested_exact.get(
                (epithet_word, infra_key), ()
            ):
                if genus_word.startswith(genus_prefix):
                    matches_by_surface[surface][abbreviation_match.concept_id] = (
                        abbreviation_match
                    )
            for genus_prefix, epithet_prefix, surface in requested_prefix.get(
                (epithet_word[0], infra_key), ()
            ):
                if genus_word.startswith(genus_prefix) and epithet_word.startswith(
                    epithet_prefix
                ):
                    matches_by_surface[surface][abbreviation_match.concept_id] = (
                        abbreviation_match
                    )
    return {
        surface: sorted(values.values(), key=lambda item: item.concept_id)
        for surface, values in matches_by_surface.items()
    }


def build_surface_matches(
    surfaces: Iterable[str], aliases_file: Path
) -> dict[str, list[TaxonMatch]]:
    surfaces = sorted({canonical_taxon_surface(value) for value in surfaces if value})
    print(f"Grounding {len(surfaces):,} direct or repaired species surfaces", flush=True)
    direct = _direct_matches(surfaces, aliases_file)
    print(
        f"Direct ontology lookup resolved {len(direct):,} species surfaces; "
        "scanning preferred labels for abbreviated genera",
        flush=True,
    )
    abbreviated = _abbreviation_matches(surfaces, aliases_file)
    combined: dict[str, dict[str, TaxonMatch]] = defaultdict(dict)
    for source in (direct, abbreviated):
        for surface, values in source.items():
            for value in values:
                combined[surface][value.concept_id] = value
    return {
        surface: sorted(values.values(), key=lambda item: item.concept_id)
        for surface, values in combined.items()
    }


def _decision_base(row: dict[str, Any]) -> dict[str, Any]:
    start = int(row.get("start") or 0)
    end = int(row.get("end") or 0)
    text = str(row.get("text") or "")
    original_surface = canonical_taxon_surface(
        text[start:end] if 0 <= start < end <= len(text) else row["word_qc_group"]
    )
    return {
        **{key: row[key] for key in CONTEXT_KEYS},
        "species_qc_original_surface": original_surface,
        "species_qc_surface": original_surface,
        "species_qc_original_start": start,
        "species_qc_original_end": end,
        "species_qc_start": start,
        "species_qc_end": end,
        "species_qc_status": None,
        "species_qc_reason": None,
        "species_qc_keep": False,
        "species_qc_repaired": False,
        "species_qc_candidate_count": 0,
        "species_qc_candidates_json": "[]",
        "species_qc_ontology": None,
        "species_qc_ontology_id": None,
        "species_qc_ontology_label": None,
        "species_qc_ontology_alias": None,
        "species_qc_ontology_scope": None,
        "species_qc_ontology_method": None,
        "species_qc_ontology_confidence": None,
        "prediction_rows": int(row["prediction_rows"]),
        "strain_taxonomy_id": row.get("strain_taxonomy_id"),
        "rel": row.get("rel"),
    }


def _apply_match(decision: dict[str, Any], match: TaxonMatch) -> None:
    decision.update(
        {
            "species_qc_ontology": match.ontology,
            "species_qc_ontology_id": match.concept_id,
            "species_qc_ontology_label": match.concept_label,
            "species_qc_ontology_alias": match.alias,
            "species_qc_ontology_scope": match.scope,
            "species_qc_ontology_method": match.method,
            "species_qc_ontology_confidence": match.confidence,
        }
    )


def decide_contexts(
    contexts: list[dict[str, Any]], aliases_file: Path
) -> list[dict[str, Any]]:
    expansions_by_context: list[list[Expansion]] = []
    candidate_surfaces: set[str] = set()
    for row in contexts:
        decision = _decision_base(row)
        original = decision["species_qc_original_surface"]
        candidates = [Expansion(original, decision["species_qc_start"], decision["species_qc_end"])]
        if not is_well_formed_species_surface(original):
            candidates.extend(
                candidate_expansions(
                    str(row.get("text") or ""),
                    int(row.get("start") or 0),
                    int(row.get("end") or 0),
                )
            )
        unique = {
            (item.surface, item.start, item.end): item for item in candidates if item.surface
        }
        ordered = sorted(
            unique.values(),
            key=lambda item: (item.end - item.start, item.start, item.surface.casefold()),
        )
        expansions_by_context.append(ordered)
        candidate_surfaces.update(item.surface for item in ordered)

    matches = build_surface_matches(candidate_surfaces, aliases_file)
    decisions: list[dict[str, Any]] = []
    for row, candidates in zip(contexts, expansions_by_context):
        decision = _decision_base(row)
        original = decision["species_qc_original_surface"]
        if row.get("ontology_status") == "matched" and row.get("ontology_id"):
            decision.update(
                {
                    "species_qc_status": "accepted_grounded",
                    "species_qc_reason": "existing_unique_ontology_match",
                    "species_qc_keep": True,
                }
            )
            decisions.append(decision)
            continue

        candidate_records: list[tuple[Expansion, TaxonMatch]] = []
        for candidate in candidates:
            for match in matches.get(candidate.surface, []):
                candidate_records.append((candidate, match))
        concepts = {match.concept_id for _, match in candidate_records}
        decision["species_qc_candidate_count"] = len(concepts)
        decision["species_qc_candidates_json"] = json.dumps(
            [
                {
                    "surface": candidate.surface,
                    "start": candidate.start,
                    "end": candidate.end,
                    "ontology": match.ontology,
                    "concept_id": match.concept_id,
                    "concept_label": match.concept_label,
                    "method": match.method,
                }
                for candidate, match in candidate_records
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

        if len(concepts) == 1:
            candidate, match = sorted(
                candidate_records,
                key=lambda item: (
                    item[0].end - item[0].start,
                    item[0].start,
                    item[0].surface.casefold(),
                ),
            )[0]
            repaired = (candidate.start, candidate.end) != (
                decision["species_qc_original_start"],
                decision["species_qc_original_end"],
            )
            relation_name = str(row.get("rel") or "").split(":")[-1]
            same_taxon = bool(
                row.get("strain_taxonomy_id")
                and row.get("strain_taxonomy_id") == match.concept_id
                and relation_name in GUARDED_SAME_TAXON_RELATIONS
            )
            decision.update(
                {
                    "species_qc_surface": candidate.surface,
                    "species_qc_start": candidate.start,
                    "species_qc_end": candidate.end,
                    "species_qc_repaired": repaired,
                    "species_qc_status": (
                        "rejected_same_taxon"
                        if same_taxon
                        else ("repaired_grounded" if repaired else "accepted_grounded")
                    ),
                    "species_qc_reason": (
                        "repaired_entity_matches_strain_taxon"
                        if same_taxon
                        else (
                            "unique_contextual_ontology_repair"
                            if repaired
                            else "unique_species_ontology_match"
                        )
                    ),
                    "species_qc_keep": not same_taxon,
                }
            )
            _apply_match(decision, match)
        elif len(concepts) > 1 or row.get("ontology_status") == "ambiguous":
            repaired_candidates = {
                (candidate.surface, candidate.start, candidate.end): candidate
                for candidate, _ in candidate_records
                if (candidate.start, candidate.end)
                != (
                    decision["species_qc_original_start"],
                    decision["species_qc_original_end"],
                )
            }
            most_specific = [
                candidate
                for candidate in repaired_candidates.values()
                if all(
                    candidate.start <= other.start and candidate.end >= other.end
                    for other in repaired_candidates.values()
                )
            ]
            if len(repaired_candidates) > 1 and len(most_specific) == 1:
                candidate = most_specific[0]
                candidate_matches = {
                    match.concept_id: match
                    for record_candidate, match in candidate_records
                    if record_candidate == candidate
                }
                if len(candidate_matches) == 1:
                    match = next(iter(candidate_matches.values()))
                    relation_name = str(row.get("rel") or "").split(":")[-1]
                    same_taxon = bool(
                        row.get("strain_taxonomy_id")
                        and row.get("strain_taxonomy_id") == match.concept_id
                        and relation_name in GUARDED_SAME_TAXON_RELATIONS
                    )
                    decision.update(
                        {
                            "species_qc_surface": candidate.surface,
                            "species_qc_start": candidate.start,
                            "species_qc_end": candidate.end,
                            "species_qc_repaired": True,
                            "species_qc_status": (
                                "rejected_same_taxon"
                                if same_taxon
                                else "repaired_grounded"
                            ),
                            "species_qc_reason": (
                                "repaired_entity_matches_strain_taxon"
                                if same_taxon
                                else "unique_most_specific_contextual_ontology_repair"
                            ),
                            "species_qc_keep": not same_taxon,
                        }
                    )
                    _apply_match(decision, match)
                else:
                    decision.update(
                        {
                            "species_qc_surface": candidate.surface,
                            "species_qc_start": candidate.start,
                            "species_qc_end": candidate.end,
                            "species_qc_repaired": True,
                            "species_qc_status": "repaired_ambiguous_surface",
                            "species_qc_reason": (
                                "most_specific_contextual_surface_with_multiple_"
                                "taxonomy_candidates"
                            ),
                            "species_qc_keep": True,
                        }
                    )
            elif len(repaired_candidates) == 1:
                candidate = next(iter(repaired_candidates.values()))
                decision.update(
                    {
                        "species_qc_surface": candidate.surface,
                        "species_qc_start": candidate.start,
                        "species_qc_end": candidate.end,
                        "species_qc_repaired": True,
                        "species_qc_status": "repaired_ambiguous_surface",
                        "species_qc_reason": (
                            "unique_contextual_surface_with_multiple_taxonomy_candidates"
                        ),
                        "species_qc_keep": True,
                    }
                )
            elif is_well_formed_species_surface(original):
                decision.update(
                    {
                        "species_qc_status": "accepted_ambiguous_surface",
                        "species_qc_reason": (
                            "well_formed_surface_with_multiple_taxonomy_candidates"
                        ),
                        "species_qc_keep": True,
                    }
                )
            else:
                decision.update(
                    {
                        "species_qc_status": "ambiguous",
                        "species_qc_reason": "multiple_contextual_repair_candidates",
                        "species_qc_keep": False,
                    }
                )
        elif is_well_formed_species_surface(original):
            decision.update(
                {
                    "species_qc_status": "accepted_surface",
                    "species_qc_reason": "well_formed_but_ungrounded_species_surface",
                    "species_qc_keep": True,
                }
            )
        else:
            repaired_candidates = {
                (candidate.surface, candidate.start, candidate.end): candidate
                for candidate in candidates
                if (candidate.start, candidate.end)
                != (
                    decision["species_qc_original_start"],
                    decision["species_qc_original_end"],
                )
                and candidate.start < decision["species_qc_original_start"]
                and is_safe_ungrounded_repair(candidate.surface)
            }
            if len(repaired_candidates) == 1:
                candidate = next(iter(repaired_candidates.values()))
                decision.update(
                    {
                        "species_qc_surface": candidate.surface,
                        "species_qc_start": candidate.start,
                        "species_qc_end": candidate.end,
                        "species_qc_repaired": True,
                        "species_qc_status": "repaired_surface",
                        "species_qc_reason": (
                            "unique_well_formed_contextual_surface_without_grounding"
                        ),
                        "species_qc_keep": True,
                    }
                )
            else:
                decision.update(
                    {
                        "species_qc_status": "rejected_fragment",
                        "species_qc_reason": (
                            "incomplete_species_surface_without_unique_repair"
                        ),
                        "species_qc_keep": False,
                    }
                )
        decisions.append(decision)
    return decisions


def _atomic_sink_parquet(frame: pl.LazyFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.sink_parquet(temporary, compression="zstd", engine="streaming")
    os.replace(temporary, destination)


def _decision_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        raise ValueError("No SPECIES prediction contexts were found")
    return pl.DataFrame(records)


def apply_species_qc(
    predictions_file: Path,
    aliases_file: Path,
    manifest_file: Path,
    accepted_output: Path,
    quarantine_output: Path,
    audit_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    source = pl.scan_parquet(predictions_file)
    columns = set(source.collect_schema().names())
    required = {
        "ner",
        "text",
        "start",
        "end",
        "word_qc_group",
        "ontology_status",
        "ontology_id",
    }
    missing = required - columns
    if missing:
        raise ValueError(f"Grounded predictions are missing columns: {sorted(missing)}")

    optional_context = [
        name
        for name in (
            "ontology_id",
            "rel",
            "strain_taxonomy_id",
        )
        if name in columns
    ]
    contexts = (
        source.filter(pl.col("ner") == SPECIES)
        .group_by(*CONTEXT_KEYS)
        .agg(
            *[pl.col(name).first().alias(name) for name in optional_context],
            pl.col("ontology_status").first().alias("ontology_status"),
            pl.len().alias("prediction_rows"),
        )
        .collect(engine="streaming")
        .sort(CONTEXT_KEYS)
        .to_dicts()
    )
    print(f"Loaded {len(contexts):,} unique SPECIES prediction contexts", flush=True)
    decisions = _decision_frame(decide_contexts(contexts, aliases_file))
    print(
        "Species QC context decisions: "
        + ", ".join(
            f"{status}={count:,}"
            for status, count in sorted(
                Counter(decisions.get_column("species_qc_status").to_list()).items()
            )
        ),
        flush=True,
    )
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    decisions.write_parquet(audit_output, compression="zstd")

    join_columns = decisions.select(
        *CONTEXT_KEYS,
        *[
            name
            for name in decisions.columns
            if name not in CONTEXT_KEYS
            and name not in {"prediction_rows", "rel", "strain_taxonomy_id"}
        ],
    ).lazy()
    annotated = source.join(join_columns, on=CONTEXT_KEYS, how="left")
    annotated = annotated.with_columns(
        pl.when(pl.col("ner") != SPECIES)
        .then(pl.lit("not_applicable"))
        .otherwise(pl.col("species_qc_status"))
        .alias("species_qc_status"),
        pl.when(pl.col("ner") != SPECIES)
        .then(pl.lit("non_species_endpoint"))
        .otherwise(pl.col("species_qc_reason"))
        .alias("species_qc_reason"),
        pl.when(pl.col("ner") != SPECIES)
        .then(pl.lit(True))
        .otherwise(pl.col("species_qc_keep"))
        .alias("species_qc_keep"),
        pl.when(pl.col("ner") != SPECIES)
        .then(pl.lit(False))
        .otherwise(pl.col("species_qc_repaired"))
        .alias("species_qc_repaired"),
    )
    has_qc_match = pl.col("species_qc_ontology_id").is_not_null()
    annotated = annotated.with_columns(
        pl.when(pl.col("species_qc_repaired"))
        .then(pl.col("species_qc_start"))
        .otherwise(pl.col("start"))
        .alias("start"),
        pl.when(pl.col("species_qc_repaired"))
        .then(pl.col("species_qc_end"))
        .otherwise(pl.col("end"))
        .alias("end"),
        pl.when(pl.col("species_qc_repaired"))
        .then(pl.col("species_qc_surface").map_elements(normalize_surface, return_dtype=pl.String))
        .otherwise(pl.col("word_qc_group"))
        .alias("word_qc_group"),
        pl.when(has_qc_match).then(pl.lit("matched")).otherwise(pl.col("ontology_status")).alias("ontology_status"),
        pl.when(has_qc_match).then(pl.lit(1)).otherwise(pl.col("ontology_candidate_count")).alias("ontology_candidate_count"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_method")).otherwise(pl.col("ontology_match_method")).alias("ontology_match_method"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_confidence")).otherwise(pl.col("ontology_match_confidence")).alias("ontology_match_confidence"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology")).otherwise(pl.col("ontology")).alias("ontology"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_id")).otherwise(pl.col("ontology_id")).alias("ontology_id"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_label")).otherwise(pl.col("ontology_label")).alias("ontology_label"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_alias")).otherwise(pl.col("ontology_matched_alias")).alias("ontology_matched_alias"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_scope")).otherwise(pl.col("ontology_alias_scope")).alias("ontology_alias_scope"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_id")).otherwise(pl.col("ontology_node_id")).alias("ontology_node_id"),
        pl.when(has_qc_match).then(pl.col("species_qc_ontology_label")).otherwise(pl.col("ontology_node_label")).alias("ontology_node_label"),
        pl.when(has_qc_match).then(pl.lit(True)).otherwise(pl.col("ontology_grounded")).alias("ontology_grounded"),
    )

    accepted = annotated.filter(pl.col("species_qc_keep"))
    quarantine = annotated.filter(
        (pl.col("ner") == SPECIES) & ~pl.col("species_qc_keep")
    )
    print(f"Writing accepted predictions to {accepted_output}", flush=True)
    _atomic_sink_parquet(accepted, accepted_output)
    print(f"Writing quarantined predictions to {quarantine_output}", flush=True)
    _atomic_sink_parquet(quarantine, quarantine_output)

    status_groups = Counter(decisions.get_column("species_qc_status").to_list())
    status_rows = {
        row["species_qc_status"]: int(row["prediction_rows"])
        for row in decisions.group_by("species_qc_status")
        .agg(pl.col("prediction_rows").sum())
        .iter_rows(named=True)
    }
    examples = {}
    for status in sorted(status_groups):
        examples[status] = (
            decisions.filter(pl.col("species_qc_status") == status)
            .sort("prediction_rows", descending=True)
            .select(
                "species_qc_original_surface",
                "species_qc_surface",
                "species_qc_reason",
                "species_qc_ontology_id",
                "species_qc_ontology_label",
                "prediction_rows",
                "text",
            )
            .head(25)
            .to_dicts()
        )
    manifest = json.loads(manifest_file.read_text())
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "Preserve raw predictions; accept existing unique grounding and "
            "well-formed surfaces; repair only unique adjacent-token expansions; "
            "ground repairs with one unique NCBI Taxonomy concept and otherwise "
            "retain a unique well-formed repair as a text node; quarantine "
            "unresolved fragments and competing contextual repairs"
        ),
        "inputs": {
            "predictions": str(predictions_file),
            "aliases": str(aliases_file),
            "manifest": str(manifest_file),
            "ontology_created_at": manifest.get("created_at"),
        },
        "species_contexts": decisions.height,
        "species_prediction_rows": int(decisions.get_column("prediction_rows").sum()),
        "status_context_counts": dict(sorted(status_groups.items())),
        "status_prediction_row_counts": dict(sorted(status_rows.items())),
        "kept_contexts": int(decisions.filter(pl.col("species_qc_keep")).height),
        "quarantined_contexts": int(decisions.filter(~pl.col("species_qc_keep")).height),
        "repaired_contexts": int(decisions.filter(pl.col("species_qc_repaired")).height),
        "examples": examples,
        "outputs": {
            "accepted": str(accepted_output),
            "quarantine": str(quarantine_output),
            "audit": str(audit_output),
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_output.with_suffix(summary_output.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("aliases", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--accepted-output", required=True, type=Path)
    parser.add_argument("--quarantine-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_species_qc(
        args.predictions,
        args.aliases,
        args.manifest,
        args.accepted_output,
        args.quarantine_output,
        args.audit_output,
        args.summary_output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
