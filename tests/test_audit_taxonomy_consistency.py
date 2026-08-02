import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from audit_taxonomy_consistency import audit_taxonomy_consistency  # noqa: E402


class TaxonomyConsistencyAuditTests(unittest.TestCase):
    def test_reports_only_guarded_same_taxon_relations(self):
        predictions = pl.DataFrame(
            {
                "ner": ["ORGANISM", "ORGANISM", "ORGANISM"],
                "rel": [
                    "STRAIN-ORGANISM:INHABITS",
                    "STRAIN-ORGANISM:INHIBITS",
                    "STRAIN-ORGANISM:INHABITS",
                ],
                "straininfo_taxon": ["Bacillus subtilis"] * 3,
                "word_qc_group": ["B. subtilis", "B. subtilis", "mouse"],
                "ontology_status": ["matched", "matched", "matched"],
                "ontology_id": [
                    "NCBITaxon:1423",
                    "NCBITaxon:1423",
                    "NCBITaxon:10090",
                ],
                "ontology_label": [
                    "Bacillus subtilis",
                    "Bacillus subtilis",
                    "Mus musculus",
                ],
            }
        )
        mapping = pl.DataFrame(
            {
                "grouped_surface": ["Bacillus subtilis"],
                "ontology_status": ["matched"],
                "ontology_id": ["NCBITaxon:1423"],
                "ontology_label": ["Bacillus subtilis"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_file = root / "predictions.parquet"
            mapping_file = root / "mapping.parquet"
            rows_file = root / "rows.tsv"
            summary_file = root / "summary.json"
            predictions.write_parquet(predictions_file)
            mapping.write_parquet(mapping_file)
            summary = audit_taxonomy_consistency(
                predictions_file,
                mapping_file,
                rows_file,
                summary_file,
            )
            rows = pl.read_csv(rows_file, separator="\t")
            stored = json.loads(summary_file.read_text())

        self.assertEqual(rows.height, 1)
        self.assertEqual(summary, stored)
        self.assertEqual(summary["same_taxon_rows"], 1)
        self.assertEqual(
            summary["rows_by_relation"],
            {"STRAIN-ORGANISM:INHABITS": 1},
        )


if __name__ == "__main__":
    unittest.main()
