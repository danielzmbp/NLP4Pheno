import csv
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_straininfo_union import build_union  # noqa: E402


class StrainInfoUnionTests(unittest.TestCase):
    def test_union_preserves_compact_ids_and_deduplicates_alias_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact_path = root / "compact.parquet"
            detailed_path = root / "detailed.csv"
            pl.DataFrame(
                {
                    "designation_key": ["ATCC1", "COMPACTONLY"],
                    "designation": ["ATCC 1", "Compact only"],
                    "si_id": [1, 2],
                    "taxon": ["Example species", "Other species"],
                    "type_strain": [False, False],
                }
            ).write_parquet(compact_path)
            fields = [
                "SI_ID",
                "Taxon_Name",
                "Type_Strain",
                "Deposit_Designations",
                "Other_Designations",
            ]
            with detailed_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "SI_ID": 1,
                        "Taxon_Name": "Example species",
                        "Type_Strain": "False",
                        "Deposit_Designations": "ATCC-1",
                        "Other_Designations": "Lab A",
                    }
                )
                writer.writerow(
                    {
                        "SI_ID": 1,
                        "Taxon_Name": "Example species",
                        "Type_Strain": "False",
                        "Deposit_Designations": "ATCC-1",
                        "Other_Designations": "Lab A",
                    }
                )
            union, summary = build_union(compact_path, detailed_path)
            self.assertEqual(union.get_column("si_id").n_unique(), 2)
            self.assertEqual(
                union.filter(
                    (pl.col("designation_key") == "ATCC1") & (pl.col("si_id") == 1)
                ).height,
                1,
            )
            shared = union.filter(pl.col("designation_key") == "ATCC1").row(
                0, named=True
            )
            self.assertTrue(shared["in_compact"])
            self.assertTrue(shared["in_detailed_deposit"])
            self.assertEqual(summary["detailed_input"]["duplicated_si_id_count"], 1)
            self.assertEqual(
                summary["union"]["compact_si_ids_missing_from_detailed"], 1
            )


if __name__ == "__main__":
    unittest.main()
