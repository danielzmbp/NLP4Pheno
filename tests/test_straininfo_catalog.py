import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from fetch_straininfo_catalog import (  # noqa: E402
    designation_key,
    designation_records,
    parse_page_rows,
)


class StrainInfoCatalogTests(unittest.TestCase):
    def test_compact_rows_and_aliases_are_normalized(self):
        strains = parse_page_rows(
            [[44736, ["ATCC 25904", "NCTC 10833"], "Staphylococcus aureus", 0, "", 1]]
        )
        aliases = designation_records(strains)
        self.assertEqual(strains[0]["si_id"], 44736)
        self.assertEqual({row["designation_key"] for row in aliases}, {"ATCC25904", "NCTC10833"})
        self.assertEqual(designation_key("ATCC-25904"), "ATCC25904")

    def test_duplicate_alias_for_same_strain_is_collapsed(self):
        strains = parse_page_rows(
            [[1, ["DSM 1", "DSM-1"], "Example species", 1, "DE", 1]]
        )
        self.assertEqual(len(designation_records(strains)), 1)


if __name__ == "__main__":
    unittest.main()
