import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from qc_species_predictions import (  # noqa: E402
    apply_species_qc,
    build_surface_matches,
    candidate_expansions,
    is_well_formed_species_surface,
)


ONTOLOGY_DEFAULTS = {
    "ontology_status": "unmatched",
    "ontology_candidate_count": 0,
    "ontology_match_method": None,
    "ontology_match_confidence": None,
    "ontology": None,
    "ontology_id": None,
    "ontology_label": None,
    "ontology_matched_alias": None,
    "ontology_alias_scope": None,
    "ontology_node_id": None,
    "ontology_node_label": None,
    "ontology_grounded": False,
}


def prediction(
    row_id,
    ner,
    text,
    surface,
    *,
    rel="STRAIN-SPECIES:INHIBITS",
    strain_taxonomy_id=None,
    matched=None,
):
    start = text.index(surface)
    row = {
        "row_id": row_id,
        "ner": ner,
        "text": text,
        "start": start,
        "end": start + len(surface),
        "word_qc_group": surface.casefold(),
        "word_strain_qc": "strain x",
        "straininfo_si_id": 1,
        "strain_taxonomy_id": strain_taxonomy_id,
        "rel": rel,
        **ONTOLOGY_DEFAULTS,
    }
    if matched:
        concept_id, concept_label = matched
        row.update(
            {
                "ontology_status": "matched",
                "ontology_candidate_count": 1,
                "ontology_match_method": "direct_exact",
                "ontology_match_confidence": 1.0,
                "ontology": "NCBITAXON",
                "ontology_id": concept_id,
                "ontology_label": concept_label,
                "ontology_matched_alias": surface,
                "ontology_alias_scope": "LABEL",
                "ontology_node_id": concept_id,
                "ontology_node_label": concept_label,
                "ontology_grounded": True,
            }
        )
    return row


class SpeciesPredictionQCTests(unittest.TestCase):
    def test_surface_patterns_and_bounded_expansion(self):
        self.assertTrue(is_well_formed_species_surface("Escherichia coli"))
        self.assertTrue(is_well_formed_species_surface("E. coli"))
        self.assertTrue(is_well_formed_species_surface("Phyllosticta sp."))
        self.assertTrue(is_well_formed_species_surface("S. Typhimurium"))
        self.assertTrue(is_well_formed_species_surface("Sh. flexneri"))
        self.assertTrue(is_well_formed_species_surface("Staph. aureus"))
        self.assertTrue(is_well_formed_species_surface("M. smeg."))
        self.assertTrue(
            is_well_formed_species_surface("P. syringae pv. actinidiae")
        )
        self.assertTrue(
            is_well_formed_species_surface("F. oxysporum f. sp. vasinfectum")
        )
        self.assertTrue(is_well_formed_species_surface("P gingivalis"))
        self.assertFalse(is_well_formed_species_surface("E"))
        self.assertFalse(is_well_formed_species_surface(". coli"))
        self.assertFalse(is_well_formed_species_surface("coli"))
        self.assertFalse(is_well_formed_species_surface("A. alternate."))

        text = "The isolate inhibits S. mutans strongly."
        start = text.index("mutans")
        expansions = candidate_expansions(text, start, start + len("mutans"))
        self.assertIn("S. mutans", [item.surface for item in expansions])
        self.assertEqual(
            [item.surface for item in expansions].count("S. mutans"),
            1,
        )

    def test_prefix_safe_abbreviations_and_exact_genera(self):
        alias_rows = []
        for concept_id, label in (
            ("NCBITaxon:9001", "Didymella eupatorii"),
            ("NCBITaxon:9002", "Papaipema eupatorii"),
            ("NCBITaxon:1280", "Staphylococcus aureus"),
            ("NCBITaxon:1334", "Streptococcus dysgalactiae"),
            ("NCBITaxon:1245", "Leuconostoc mesenteroides"),
            ("NCBITaxon:1772", "Mycolicibacterium smegmatis"),
            ("NCBITaxon:1386", "Bacillus"),
            ("NCBITaxon:9003", "Argyresthia alternatella"),
        ):
            alias_rows.append(
                {
                    "entity_type": "SPECIES",
                    "ontology": "NCBITAXON",
                    "concept_id": concept_id,
                    "concept_label": label,
                    "alias": label,
                    "normalized_alias": label.casefold(),
                    "relaxed_alias": label.casefold(),
                    "formula_alias": None,
                    "scope": "LABEL",
                    "auto_eligible": True,
                }
            )
        alias_rows.append(
            {
                "entity_type": "SPECIES",
                "ontology": "NCBITAXON",
                "concept_id": "NCBITaxon:9001",
                "concept_label": "Didymella eupatorii",
                "alias": "Phoma eupatorii",
                "normalized_alias": "phoma eupatorii",
                "relaxed_alias": "phoma eupatorii",
                "formula_alias": None,
                "scope": "SYNONYM",
                "auto_eligible": True,
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            aliases_file = Path(temporary_directory) / "aliases.parquet"
            pl.DataFrame(alias_rows).write_parquet(aliases_file)
            matches = build_surface_matches(
                [
                    "Pho. eupatorii",
                    "P. eupatorii",
                    "Staph. aureus",
                    "Strep. dysgalactiae",
                    "Leuc. mesenteroides",
                    "M. smeg.",
                    "Bacillus",
                    "A. alternate.",
                ],
                aliases_file,
            )

        self.assertEqual(
            [match.concept_id for match in matches["Pho. eupatorii"]],
            ["NCBITaxon:9001"],
        )
        self.assertEqual(
            {match.concept_id for match in matches["P. eupatorii"]},
            {"NCBITaxon:9001", "NCBITaxon:9002"},
        )
        self.assertEqual(matches["Staph. aureus"][0].concept_id, "NCBITaxon:1280")
        self.assertEqual(
            matches["Strep. dysgalactiae"][0].concept_id, "NCBITaxon:1334"
        )
        self.assertEqual(
            matches["Leuc. mesenteroides"][0].concept_id, "NCBITaxon:1245"
        )
        self.assertEqual(matches["M. smeg."][0].concept_id, "NCBITaxon:1772")
        self.assertEqual(matches["Bacillus"][0].concept_id, "NCBITaxon:1386")
        self.assertNotIn("A. alternate.", matches)

    def test_qc_repairs_unique_taxa_and_quarantines_fragments(self):
        alias_rows = []
        for concept_id, label in (
            ("NCBITaxon:562", "Escherichia coli"),
            ("NCBITaxon:1309", "Streptococcus mutans"),
            ("NCBITaxon:1423", "Bacillus subtilis"),
            ("NCBITaxon:28901", "Salmonella enterica"),
            (
                "NCBITaxon:90371",
                "Salmonella enterica serovar Typhimurium",
            ),
            ("NCBITaxon:9001", "Alpha testii"),
            ("NCBITaxon:9002", "Another testii"),
        ):
            alias_rows.append(
                {
                    "entity_type": "SPECIES",
                    "ontology": "NCBITAXON",
                    "concept_id": concept_id,
                    "concept_label": label,
                    "alias": label,
                    "normalized_alias": label.casefold(),
                    "relaxed_alias": label.casefold(),
                    "formula_alias": None,
                    "scope": "LABEL",
                    "auto_eligible": True,
                }
            )
        aliases = pl.DataFrame(alias_rows)
        rows = [
            prediction(1, "SPECIES", "E. coli was inhibited.", "E. coli"),
            prediction(
                2,
                "SPECIES",
                "The isolate inhibits S. mutans strongly.",
                "mutans",
            ),
            prediction(3, "SPECIES", "E was measured.", "E"),
            prediction(
                4,
                "SPECIES",
                "Novibacter testii was recovered.",
                "Novibacter testii",
            ),
            prediction(5, "SPECIES", "A. testii was recovered.", "A. testii"),
            prediction(
                6,
                "SPECIES",
                "B. subtilis inhabited the culture.",
                "B. subtilis",
                rel="STRAIN-SPECIES:INHABITS",
                strain_taxonomy_id="NCBITaxon:1423",
            ),
            prediction(
                7,
                "SPECIES",
                "Phaeobacter was abundant.",
                "Phaeobacter",
                matched=("NCBITaxon:302485", "Phaeobacter"),
            ),
            prediction(8, "COMPOUND", "glucose was produced.", "glucose"),
            prediction(
                9,
                "SPECIES",
                "The isolate inhibits A. testii strongly.",
                "testii",
            ),
            prediction(
                10,
                "SPECIES",
                "Enteritidis was significantly higher.",
                "Enteritidis",
            ),
            prediction(
                11,
                "SPECIES",
                "Reductions occurred in S . enterica serovar Typhimurium.",
                "S .",
            ),
        ]
        predictions = pl.DataFrame(rows)
        manifest = {
            "created_at": "2026-01-01T00:00:00Z",
            "sources": {
                "NCBITAXON": {
                    "entity_types": ["SPECIES", "ORGANISM"],
                    "terms": len(alias_rows),
                    "aliases": len(alias_rows),
                    "sha256": "test",
                    "ontology_header": {"data-version": "test"},
                }
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_file = root / "predictions.parquet"
            aliases_file = root / "aliases.parquet"
            manifest_file = root / "manifest.json"
            accepted_file = root / "accepted.parquet"
            quarantine_file = root / "quarantine.parquet"
            audit_file = root / "audit.parquet"
            summary_file = root / "summary.json"
            predictions.write_parquet(predictions_file)
            aliases.write_parquet(aliases_file)
            manifest_file.write_text(json.dumps(manifest))

            summary = apply_species_qc(
                predictions_file,
                aliases_file,
                manifest_file,
                accepted_file,
                quarantine_file,
                audit_file,
                summary_file,
            )
            accepted = pl.read_parquet(accepted_file).sort("row_id")
            quarantine = pl.read_parquet(quarantine_file).sort("row_id")
            audit = pl.read_parquet(audit_file)

        self.assertEqual(
            accepted.get_column("row_id").to_list(), [1, 2, 4, 5, 7, 8, 9, 11]
        )
        self.assertEqual(quarantine.get_column("row_id").to_list(), [3, 6, 10])

        ecoli = accepted.filter(pl.col("row_id") == 1).row(0, named=True)
        self.assertEqual(ecoli["species_qc_status"], "accepted_grounded")
        self.assertEqual(ecoli["ontology_id"], "NCBITaxon:562")

        repaired = accepted.filter(pl.col("row_id") == 2).row(0, named=True)
        self.assertEqual(repaired["species_qc_status"], "repaired_grounded")
        self.assertEqual(repaired["word_qc_group"], "s. mutans")
        self.assertEqual(repaired["ontology_id"], "NCBITaxon:1309")
        self.assertEqual(
            repaired["text"][repaired["start"] : repaired["end"]], "S. mutans"
        )

        surface = accepted.filter(pl.col("row_id") == 4).row(0, named=True)
        self.assertEqual(surface["species_qc_status"], "accepted_surface")

        existing = accepted.filter(pl.col("row_id") == 7).row(0, named=True)
        self.assertEqual(existing["species_qc_status"], "accepted_grounded")

        nested = accepted.filter(pl.col("row_id") == 11).row(0, named=True)
        self.assertEqual(nested["species_qc_status"], "repaired_grounded")
        self.assertEqual(
            nested["word_qc_group"], "s. enterica serovar typhimurium"
        )
        self.assertEqual(nested["ontology_id"], "NCBITaxon:90371")

        statuses = dict(
            audit.select("species_qc_original_surface", "species_qc_status").rows()
        )
        self.assertEqual(statuses["E"], "rejected_fragment")
        self.assertEqual(statuses["A. testii"], "accepted_ambiguous_surface")
        self.assertEqual(statuses["testii"], "repaired_ambiguous_surface")
        self.assertEqual(statuses["Enteritidis"], "rejected_fragment")
        self.assertEqual(statuses["B. subtilis"], "rejected_same_taxon")
        self.assertEqual(summary["species_contexts"], 10)
        self.assertEqual(summary["quarantined_contexts"], 3)


if __name__ == "__main__":
    unittest.main()
