#!/usr/bin/env python3
"""Compare two NLP4Pheno PMC relation-network runs.

The script uses only the Python standard library so it can run on an offline
HPC compute node.  It distinguishes evidence rows, article-supported edges,
and unique biological edges; comparing only TSV line counts conflates these.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PMCID_PATTERN = re.compile(r"^PMC\d+$")
EXPECTED_PAIRS = {
    ("STRAIN", "ISOLATE", "INHABITS"),
    ("STRAIN", "MEDIUM", "GROWS_ON"),
    ("STRAIN", "PHENOTYPE", "PRESENTS"),
    ("STRAIN", "ORGANISM", "INHABITS"),
    ("STRAIN", "COMPOUND", "RESISTS"),
    ("STRAIN", "PHENOTYPE", "PROMOTES"),
    ("STRAIN", "ORGANISM", "INFECTS"),
    ("COMPOUND", "STRAIN", "INHIBITS"),
    ("STRAIN", "DISEASE", "ASSOCIATED_WITH"),
    ("STRAIN", "COMPOUND", "DEGRADES"),
    ("STRAIN", "ORGANISM", "SYMBIONT_OF"),
    ("STRAIN", "SPECIES", "INHIBITS"),
    ("STRAIN", "ORGANISM", "INHIBITS"),
    ("STRAIN", "PHENOTYPE", "INHIBITS"),
    ("STRAIN", "COMPOUND", "PRODUCES"),
    ("STRAIN", "DISEASE", "INHIBITS"),
}


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def summarize_network(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    edge_pmc: set[tuple[str, str, str, str]] = set()
    evidence: set[tuple[str, ...]] = set()
    typed_evidence: set[tuple[str, ...]] = set()
    relations: Counter[str] = Counter()
    relation_edges: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    relation_pmc: dict[str, set[str]] = defaultdict(set)
    type_pairs: Counter[str] = Counter()
    strains: set[str] = set()
    entities: set[tuple[str, str]] = set()
    pmcids: set[str] = set()
    endpoint_frequency: Counter[tuple[str, str]] = Counter()
    strain_frequency: Counter[str] = Counter()
    duplicate_edge_evidence_by_relation: Counter[str] = Counter()
    examples: dict[tuple[str, str, str], dict[str, str]] = {}
    rows = 0
    invalid_pmcids = 0
    missing_provenance = 0
    invalid_relation_signatures = 0
    self_loops = 0

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "source",
            "target",
            "rel",
            "source_ner",
            "target_ner",
            "pmcid",
            "article_version",
            "paragraph",
            "sentence_range",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        for row in reader:
            rows += 1
            source = row["source"]
            target = row["target"]
            relation = row["rel"]
            source_ner = row["source_ner"]
            target_ner = row["target_ner"]
            pmcid = row["pmcid"]
            edge = (source, target, relation)
            edge_counts[edge] += 1
            edge_pmc.add((*edge, pmcid))
            evidence_key = (
                *edge,
                pmcid,
                row["article_version"],
                row["paragraph"],
                row["sentence_range"],
            )
            if evidence_key in evidence:
                duplicate_edge_evidence_by_relation[relation] += 1
            evidence.add(evidence_key)
            typed_evidence.add(
                (
                    *edge,
                    source_ner,
                    target_ner,
                    pmcid,
                    row["article_version"],
                    row["paragraph"],
                    row["sentence_range"],
                )
            )
            relations[relation] += 1
            relation_edges[relation].add(edge)
            relation_pmc[relation].add(pmcid)
            type_pairs[f"{source_ner}->{target_ner}:{relation}"] += 1
            pmcids.add(pmcid)
            examples.setdefault(
                edge,
                {
                    "pmcid": pmcid,
                    "article_version": row["article_version"],
                    "paragraph": row["paragraph"],
                    "sentence_range": row["sentence_range"],
                },
            )
            for node, ner in ((source, source_ner), (target, target_ner)):
                if ner == "STRAIN":
                    strains.add(node)
                    strain_frequency[node] += 1
                else:
                    entities.add((ner, node))
                    endpoint_frequency[(ner, node)] += 1
            if not PMCID_PATTERN.fullmatch(pmcid):
                invalid_pmcids += 1
            if not all(
                (
                    pmcid,
                    row["article_version"],
                    row["paragraph"],
                    row["sentence_range"],
                )
            ):
                missing_provenance += 1
            if (source_ner, target_ner, relation) not in EXPECTED_PAIRS:
                invalid_relation_signatures += 1
            if source == target:
                self_loops += 1

    relation_summary = {
        relation: {
            "evidence_rows": count,
            "unique_edges": len(relation_edges[relation]),
            "unique_pmcids": len(relation_pmc[relation]),
        }
        for relation, count in sorted(relations.items())
    }
    summary = {
        "path": str(path),
        "evidence_rows": rows,
        "unique_untyped_evidence_records": len(evidence),
        "unique_typed_evidence_records": len(typed_evidence),
        "exact_duplicate_rows": rows - len(typed_evidence),
        "cross_type_duplicate_records": len(typed_evidence) - len(evidence),
        "duplicate_edge_evidence_rows_after_ignoring_type": rows - len(evidence),
        "duplicate_edge_evidence_rows_by_relation": dict(
            sorted(duplicate_edge_evidence_by_relation.items())
        ),
        "article_supported_edges": len(edge_pmc),
        "unique_edges": len(edge_counts),
        "unique_strains": len(strains),
        "unique_non_strain_entities": len(entities),
        "unique_pmcids": len(pmcids),
        "invalid_pmcids": invalid_pmcids,
        "missing_provenance_rows": missing_provenance,
        "invalid_relation_signature_rows": invalid_relation_signatures,
        "self_loops": self_loops,
        "relations": relation_summary,
        "relation_signatures": dict(sorted(type_pairs.items())),
        "entity_types": dict(
            sorted(Counter(entity_type for entity_type, _ in entities).items())
        ),
        "highest_frequency_entities": [
            {
                "entity_type": entity_type,
                "entity": entity,
                "evidence_rows": count,
            }
            for (entity_type, entity), count in endpoint_frequency.most_common(25)
        ],
        "highest_frequency_strains": [
            {"strain": strain, "evidence_rows": count}
            for strain, count in strain_frequency.most_common(25)
        ],
    }
    comparison_data = {
        "edge_counts": edge_counts,
        "edge_pmc": edge_pmc,
        "evidence": evidence,
        "examples": examples,
    }
    return summary, comparison_data


def summarize_ontology(path: Path) -> tuple[dict[str, Any], dict[str, set[Any]]]:
    rows = 0
    statuses: Counter[str] = Counter()
    status_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    ontologies: Counter[str] = Counter()
    concept_ids: set[str] = set()
    matched_edges: set[tuple[str, str, str]] = set()
    all_edges: set[tuple[str, str, str]] = set()
    matched_edge_pmc: set[tuple[str, str, str, str]] = set()
    all_edge_pmc: set[tuple[str, str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows += 1
            status = row.get("ontology_status") or "missing"
            entity_type = (
                row["target_ner"]
                if row["source_ner"] == "STRAIN"
                else row["source_ner"]
            )
            statuses[status] += 1
            status_by_type[entity_type][status] += 1
            edge = (row["source"], row["target"], row["rel"])
            all_edges.add(edge)
            all_edge_pmc.add((*edge, row["pmcid"]))
            if status == "matched":
                matched_edges.add(edge)
                matched_edge_pmc.add((*edge, row["pmcid"]))
                if row.get("ontology"):
                    ontologies[row["ontology"]] += 1
                if row.get("ontology_id"):
                    concept_ids.add(row["ontology_id"])
    matched_rows = statuses["matched"]
    summary = {
        "path": str(path),
        "evidence_rows": rows,
        "status_rows": dict(sorted(statuses.items())),
        "matched_row_percentage": percentage(matched_rows, rows),
        "unique_edges": len(all_edges),
        "matched_unique_edges": len(matched_edges),
        "matched_edge_percentage": percentage(len(matched_edges), len(all_edges)),
        "unique_ontology_concepts": len(concept_ids),
        "matched_rows_by_ontology": dict(sorted(ontologies.items())),
        "status_rows_by_entity_type": {
            entity_type: dict(sorted(counts.items()))
            for entity_type, counts in sorted(status_by_type.items())
        },
    }
    data = {
        "all_edges": all_edges,
        "matched_edges": matched_edges,
        "all_edge_pmc": all_edge_pmc,
        "matched_edge_pmc": matched_edge_pmc,
    }
    return summary, data


def compare_sets(old: set[Any], new: set[Any]) -> dict[str, Any]:
    overlap = len(old & new)
    union = len(old | new)
    return {
        "old": len(old),
        "new": len(new),
        "delta": len(new) - len(old),
        "delta_percentage": round(100.0 * (len(new) - len(old)) / len(old), 3)
        if old
        else None,
        "overlap": overlap,
        "old_only": len(old - new),
        "new_only": len(new - old),
        "jaccard": round(overlap / union, 6) if union else 1.0,
        "new_retains_old_percentage": percentage(overlap, len(old)),
        "new_overlap_percentage": percentage(overlap, len(new)),
    }


def ranked_edge_changes(
    old: dict[str, Any], new: dict[str, Any], limit: int
) -> dict[str, list[dict[str, Any]]]:
    old_counts = old["edge_counts"]
    new_counts = new["edge_counts"]

    def record(
        edge: tuple[str, str, str],
        old_count: int,
        new_count: int,
        example_source: dict[str, Any],
    ) -> dict[str, Any]:
        source, target, relation = edge
        return {
            "source": source,
            "target": target,
            "relation": relation,
            "old_evidence_rows": old_count,
            "new_evidence_rows": new_count,
            "delta": new_count - old_count,
            "example": example_source["examples"].get(edge, {}),
        }

    additions = sorted(
        (
            record(edge, old_counts.get(edge, 0), count, new)
            for edge, count in new_counts.items()
            if edge not in old_counts
        ),
        key=lambda item: (-item["new_evidence_rows"], item["source"], item["target"]),
    )[:limit]
    removals = sorted(
        (
            record(edge, count, new_counts.get(edge, 0), old)
            for edge, count in old_counts.items()
            if edge not in new_counts
        ),
        key=lambda item: (-item["old_evidence_rows"], item["source"], item["target"]),
    )[:limit]
    increases = sorted(
        (
            record(edge, old_counts[edge], new_counts[edge], new)
            for edge in old_counts.keys() & new_counts.keys()
            if new_counts[edge] > old_counts[edge]
        ),
        key=lambda item: (-item["delta"], item["source"], item["target"]),
    )[:limit]
    decreases = sorted(
        (
            record(edge, old_counts[edge], new_counts[edge], old)
            for edge in old_counts.keys() & new_counts.keys()
            if new_counts[edge] < old_counts[edge]
        ),
        key=lambda item: (item["delta"], item["source"], item["target"]),
    )[:limit]
    return {
        "largest_new_edges": additions,
        "largest_removed_edges": removals,
        "largest_evidence_increases": increases,
        "largest_evidence_decreases": decreases,
    }


def relation_deltas(
    old_summary: dict[str, Any], new_summary: dict[str, Any]
) -> dict[str, Any]:
    old_relations = old_summary["relations"]
    new_relations = new_summary["relations"]
    result: dict[str, Any] = {}
    for relation in sorted(old_relations.keys() | new_relations.keys()):
        result[relation] = {}
        for measure in ("evidence_rows", "unique_edges", "unique_pmcids"):
            old_value = old_relations.get(relation, {}).get(measure, 0)
            new_value = new_relations.get(relation, {}).get(measure, 0)
            result[relation][measure] = {
                "old": old_value,
                "new": new_value,
                "delta": new_value - old_value,
                "delta_percentage": round(
                    100.0 * (new_value - old_value) / old_value, 3
                )
                if old_value
                else None,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-network", required=True, type=Path)
    parser.add_argument("--new-network", required=True, type=Path)
    parser.add_argument("--old-ontology-network", type=Path)
    parser.add_argument("--new-ontology-network", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int, default=25)
    args = parser.parse_args()

    old_summary, old_data = summarize_network(args.old_network)
    new_summary, new_data = summarize_network(args.new_network)
    report: dict[str, Any] = {
        "old": old_summary,
        "new": new_summary,
        "comparison": {
            "evidence_records": compare_sets(
                old_data["evidence"], new_data["evidence"]
            ),
            "article_supported_edges": compare_sets(
                old_data["edge_pmc"], new_data["edge_pmc"]
            ),
            "unique_edges": compare_sets(
                set(old_data["edge_counts"]), set(new_data["edge_counts"])
            ),
            "relations": relation_deltas(old_summary, new_summary),
            "ranked_changes": ranked_edge_changes(
                old_data, new_data, args.sample_limit
            ),
        },
    }
    if args.old_ontology_network:
        old_ontology_summary, old_ontology_data = summarize_ontology(
            args.old_ontology_network
        )
        report["old"]["ontology"] = old_ontology_summary
    if args.new_ontology_network:
        new_ontology_summary, new_ontology_data = summarize_ontology(
            args.new_ontology_network
        )
        report["new"]["ontology"] = new_ontology_summary
    if args.old_ontology_network and args.new_ontology_network:
        report["comparison"]["ontology"] = {
            "all_unique_edges": compare_sets(
                old_ontology_data["all_edges"], new_ontology_data["all_edges"]
            ),
            "matched_unique_edges": compare_sets(
                old_ontology_data["matched_edges"],
                new_ontology_data["matched_edges"],
            ),
            "all_article_supported_edges": compare_sets(
                old_ontology_data["all_edge_pmc"],
                new_ontology_data["all_edge_pmc"],
            ),
            "matched_article_supported_edges": compare_sets(
                old_ontology_data["matched_edge_pmc"],
                new_ontology_data["matched_edge_pmc"],
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
