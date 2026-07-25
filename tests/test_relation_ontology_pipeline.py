import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from ground_relation_ontology import ground_relation_predictions  # noqa: E402


class RelationOntologyPipelineTests(unittest.TestCase):
    def test_grounding_preserves_rows_and_retains_ambiguity(self):
        aliases = pl.DataFrame(
            {
                "entity_type": [
                    "COMPOUND",
                    "DISEASE",
                    "DISEASE",
                ],
                "ontology": ["CHEBI", "MONDO", "MONDO"],
                "concept_id": ["CHEBI:17234", "MONDO:1", "MONDO:2"],
                "concept_label": ["glucose", "disease one", "disease two"],
                "alias": ["D-glucose", "shared disease", "shared disease"],
                "normalized_alias": [
                    "d-glucose",
                    "shared disease",
                    "shared disease",
                ],
                "relaxed_alias": [
                    "d glucose",
                    "shared disease",
                    "shared disease",
                ],
                "formula_alias": ["", "", ""],
                "scope": ["EXACT", "EXACT", "EXACT"],
                "auto_eligible": [True, True, True],
            }
        )
        predictions = pl.DataFrame(
            {
                "ner": ["COMPOUND", "COMPOUND", "DISEASE", "ORGANISM"],
                "word_qc_group": [
                    "D-glucose",
                    "D-glucose",
                    "shared disease",
                    "mouse",
                ],
                "row_id": [1, 2, 3, 4],
            }
        )
        manifest = {
            "created_at": "2026-01-01T00:00:00Z",
            "sources": {
                "CHEBI": {
                    "entity_types": ["COMPOUND"],
                    "terms": 1,
                    "aliases": 1,
                    "sha256": "chebi-hash",
                    "ontology_header": {"data-version": "test"},
                },
                "MONDO": {
                    "entity_types": ["DISEASE"],
                    "terms": 2,
                    "aliases": 2,
                    "sha256": "mondo-hash",
                    "ontology_header": {"data-version": "test"},
                },
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_file = root / "predictions.parquet"
            aliases_file = root / "aliases.parquet"
            manifest_file = root / "manifest.json"
            grounded_file = root / "grounded.parquet"
            mapping_file = root / "mapping.parquet"
            summary_file = root / "summary.json"
            predictions.write_parquet(predictions_file)
            aliases.write_parquet(aliases_file)
            manifest_file.write_text(json.dumps(manifest))

            ground_relation_predictions(
                predictions_file,
                aliases_file,
                manifest_file,
                grounded_file,
                mapping_file,
                summary_file,
            )

            grounded = pl.read_parquet(grounded_file).sort("row_id")
            mapping = pl.read_parquet(mapping_file)
            summary = json.loads(summary_file.read_text())

        self.assertEqual(grounded.height, predictions.height)
        self.assertEqual(grounded.get_column("row_id").to_list(), [1, 2, 3, 4])

        glucose = grounded.filter(pl.col("row_id") == 1).row(0, named=True)
        self.assertEqual(glucose["ontology_status"], "matched")
        self.assertEqual(glucose["ontology_id"], "CHEBI:17234")
        self.assertEqual(glucose["ontology_node_id"], "CHEBI:17234")
        self.assertEqual(glucose["ontology_node_label"], "glucose")
        self.assertEqual(glucose["ontology_match_method"], "direct_exact")
        self.assertEqual(glucose["ontology_match_confidence"], 0.99)

        ambiguous = grounded.filter(pl.col("row_id") == 3).row(0, named=True)
        self.assertEqual(ambiguous["ontology_status"], "ambiguous")
        self.assertEqual(ambiguous["ontology_candidate_count"], 2)
        self.assertEqual(ambiguous["ontology_node_id"], "shared disease")

        unsupported = grounded.filter(pl.col("row_id") == 4).row(0, named=True)
        self.assertEqual(unsupported["ontology_status"], "unsupported")
        self.assertEqual(unsupported["ontology_node_id"], "mouse")

        self.assertEqual(mapping.height, 3)
        ambiguous_mapping = mapping.filter(
            pl.col("grouped_surface") == "shared disease"
        ).row(0, named=True)
        self.assertEqual(
            len(json.loads(ambiguous_mapping["ontology_candidates_json"])),
            2,
        )
        self.assertEqual(
            summary["entity_types"]["COMPOUND"]["prediction_rows"],
            2,
        )
        self.assertEqual(
            summary["entity_types"]["COMPOUND"]["matched_row_coverage"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
