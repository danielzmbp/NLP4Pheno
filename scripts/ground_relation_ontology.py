#!/usr/bin/env python3
"""Ground grouped relation entities to local, versioned ontology aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from ontology_grounding import (
    ground_mentions,
    load_candidate_indexes,
    normalize_formula_surface,
    normalize_surface,
    relaxed_surface,
)


MAPPING_SCHEMA = {
    "entity_type": pl.String,
    "grouped_surface": pl.String,
    "normalized_surface": pl.String,
    "relaxed_surface": pl.String,
    "formula_surface": pl.String,
    "prediction_rows": pl.Int64,
    "ontology_status": pl.String,
    "ontology_candidate_count": pl.Int64,
    "ontology_candidates_json": pl.String,
    "ontology_match_method": pl.String,
    "ontology_match_confidence": pl.Float64,
    "ontology": pl.String,
    "ontology_id": pl.String,
    "ontology_label": pl.String,
    "ontology_matched_alias": pl.String,
    "ontology_alias_scope": pl.String,
    "ontology_node_id": pl.String,
    "ontology_node_label": pl.String,
    "ontology_grounded": pl.Boolean,
}

OUTPUT_GROUNDING_COLUMNS = [
    "ontology_status",
    "ontology_candidate_count",
    "ontology_match_method",
    "ontology_match_confidence",
    "ontology",
    "ontology_id",
    "ontology_label",
    "ontology_matched_alias",
    "ontology_alias_scope",
    "ontology_node_id",
    "ontology_node_label",
    "ontology_grounded",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_confidence(status: str, method: str | None, scope: str | None) -> float | None:
    """Return a deterministic evidence strength, not a calibrated probability."""
    if status != "matched" or not method:
        return None
    if method.endswith("_formula"):
        return 0.99
    if method.endswith("_exact"):
        return 1.0 if scope == "LABEL" else 0.99
    if method.endswith("_normalized"):
        return 0.98 if scope == "LABEL" else 0.97
    return 0.95


def build_mapping(
    grouped_counts: pl.DataFrame,
    aliases_file: Path,
    manifest: dict[str, Any],
) -> pl.DataFrame:
    supported = {
        entity_type
        for metadata in manifest["sources"].values()
        for entity_type in metadata["entity_types"]
    }
    mentions = [
        {
            "entity_type": row["ner"],
            "surface": row["word_qc_group"],
            "normalized_surface": normalize_surface(row["word_qc_group"]),
            "relaxed_surface": relaxed_surface(row["word_qc_group"]),
            "formula_surface": normalize_formula_surface(row["word_qc_group"]),
            "expanded_form": None,
            "prediction_rows": int(row["prediction_rows"]),
        }
        for row in grouped_counts.iter_rows(named=True)
    ]
    exact, relaxed, formula = load_candidate_indexes(aliases_file, mentions)
    grounded = ground_mentions(mentions, exact, relaxed, formula, supported)

    records: list[dict[str, Any]] = []
    for row in grounded:
        matched = row["status"] == "matched"
        surface = row["surface"]
        records.append(
            {
                "entity_type": row["entity_type"],
                "grouped_surface": surface,
                "normalized_surface": row["normalized_surface"],
                "relaxed_surface": row["relaxed_surface"],
                "formula_surface": row["formula_surface"],
                "prediction_rows": row["prediction_rows"],
                "ontology_status": row["status"],
                "ontology_candidate_count": row["candidate_count"],
                "ontology_candidates_json": row["candidates_json"],
                "ontology_match_method": row["match_method"],
                "ontology_match_confidence": rule_confidence(
                    row["status"],
                    row["match_method"],
                    row["alias_scope"],
                ),
                "ontology": row["ontology"],
                "ontology_id": row["concept_id"],
                "ontology_label": row["concept_label"],
                "ontology_matched_alias": row["matched_alias"],
                "ontology_alias_scope": row["alias_scope"],
                "ontology_node_id": row["concept_id"] if matched else surface,
                "ontology_node_label": (
                    row["concept_label"] if matched else surface
                ),
                "ontology_grounded": matched,
            }
        )
    return pl.DataFrame(records, schema=MAPPING_SCHEMA)


def summarize_mapping(
    mapping: pl.DataFrame,
    manifest: dict[str, Any],
    aliases_file: Path,
    manifest_file: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "unique auto-eligible preferred label, exact synonym, conservative "
            "normalized alias, or case-preserving ChEBI formula; no fuzzy or "
            "embedding fallback; ambiguity is retained"
        ),
        "confidence_note": (
            "ontology_match_confidence is a rule-based evidence strength, not a "
            "calibrated probability"
        ),
        "aliases": {
            "path": str(aliases_file),
            "sha256": sha256_file(aliases_file),
            "rows": int(pl.scan_parquet(aliases_file).select(pl.len()).collect().item()),
        },
        "manifest": {
            "path": str(manifest_file),
            "sha256": sha256_file(manifest_file),
            "created_at": manifest.get("created_at"),
            "sources": {
                name: {
                    "entity_types": metadata["entity_types"],
                    "terms": metadata["terms"],
                    "aliases": metadata["aliases"],
                    "sha256": metadata["sha256"],
                    "version": metadata.get("ontology_header", {}).get("data-version"),
                }
                for name, metadata in manifest["sources"].items()
            },
        },
        "entity_types": {},
    }
    for entity_type in mapping.get_column("entity_type").unique().sort().to_list():
        selected = mapping.filter(pl.col("entity_type") == entity_type)
        group_status = Counter(selected.get_column("ontology_status").to_list())
        row_status = {
            row["ontology_status"]: int(row["prediction_rows"])
            for row in selected.group_by("ontology_status")
            .agg(pl.col("prediction_rows").sum())
            .iter_rows(named=True)
        }
        total_groups = selected.height
        total_rows = int(selected.get_column("prediction_rows").sum())
        matched_groups = group_status["matched"]
        matched_rows = row_status.get("matched", 0)
        by_ontology = {
            row["ontology"]: {
                "groups": int(row["groups"]),
                "prediction_rows": int(row["prediction_rows"]),
            }
            for row in selected.filter(pl.col("ontology_status") == "matched")
            .group_by("ontology")
            .agg(
                pl.len().alias("groups"),
                pl.col("prediction_rows").sum(),
            )
            .sort("ontology")
            .iter_rows(named=True)
        }
        by_method = {
            row["ontology_match_method"]: {
                "groups": int(row["groups"]),
                "prediction_rows": int(row["prediction_rows"]),
            }
            for row in selected.filter(pl.col("ontology_status") == "matched")
            .group_by("ontology_match_method")
            .agg(
                pl.len().alias("groups"),
                pl.col("prediction_rows").sum(),
            )
            .sort("ontology_match_method")
            .iter_rows(named=True)
        }
        unresolved = (
            selected.filter(
                pl.col("ontology_status").is_in(["ambiguous", "unmatched"])
            )
            .sort("prediction_rows", descending=True)
            .select(
                "grouped_surface",
                "ontology_status",
                "prediction_rows",
                "ontology_candidate_count",
            )
            .head(25)
            .to_dicts()
        )
        summary["entity_types"][entity_type] = {
            "unique_groups": total_groups,
            "prediction_rows": total_rows,
            "group_status_counts": dict(sorted(group_status.items())),
            "row_status_counts": dict(sorted(row_status.items())),
            "matched_group_coverage": (
                round(matched_groups / total_groups, 6) if total_groups else 0.0
            ),
            "matched_row_coverage": (
                round(matched_rows / total_rows, 6) if total_rows else 0.0
            ),
            "matched_by_ontology": by_ontology,
            "matched_by_method": by_method,
            "top_unresolved": unresolved,
        }
    return summary


def ground_relation_predictions(
    predictions_file: Path,
    aliases_file: Path,
    manifest_file: Path,
    grounded_output: Path,
    mapping_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    source = pl.scan_parquet(predictions_file)
    grouped_counts = (
        source.select("ner", "word_qc_group")
        .filter(pl.col("ner").is_not_null() & pl.col("word_qc_group").is_not_null())
        .group_by("ner", "word_qc_group")
        .len(name="prediction_rows")
        .collect(engine="streaming")
        .sort(["ner", "word_qc_group"])
    )
    print(
        f"Grounding {grouped_counts.height:,} unique grouped entity strings",
        flush=True,
    )
    with manifest_file.open() as handle:
        manifest = json.load(handle)
    mapping = build_mapping(grouped_counts, aliases_file, manifest)

    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping_temporary = mapping_output.with_suffix(mapping_output.suffix + ".tmp")
    mapping.write_parquet(mapping_temporary, compression="zstd")
    os.replace(mapping_temporary, mapping_output)

    join_mapping = pl.scan_parquet(mapping_output).select(
        pl.col("entity_type").alias("ner"),
        pl.col("grouped_surface").alias("word_qc_group"),
        *OUTPUT_GROUNDING_COLUMNS,
    )
    grounded = source.join(
        join_mapping,
        on=["ner", "word_qc_group"],
        how="left",
    )
    grounded_output.parent.mkdir(parents=True, exist_ok=True)
    grounded_temporary = grounded_output.with_suffix(grounded_output.suffix + ".tmp")
    grounded.sink_parquet(
        grounded_temporary,
        compression="zstd",
        engine="streaming",
    )
    os.replace(grounded_temporary, grounded_output)

    summary = summarize_mapping(mapping, manifest, aliases_file, manifest_file)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_output.with_suffix(summary_output.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_temporary, summary_output)
    matched = mapping.filter(pl.col("ontology_status") == "matched").height
    print(
        f"Matched {matched:,}/{mapping.height:,} unique groups; "
        f"wrote {grounded_output}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("aliases", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping-output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_relation_predictions(
        args.predictions,
        args.aliases,
        args.manifest,
        args.output,
        args.mapping_output,
        args.summary,
    )


if __name__ == "__main__":
    main()
