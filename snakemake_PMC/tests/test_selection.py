from __future__ import annotations

import sys
import unittest
from pathlib import Path


PMC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PMC_DIR / "scripts"))

from select_articles import load_latest_versions  # noqa: E402


class ArticleSelectionTests(unittest.TestCase):
    def test_newest_version_is_selected_and_unrequested_articles_are_ignored(self) -> None:
        inventory = PMC_DIR / "tests/fixtures/inventory.csv"
        selected, rows_scanned = load_latest_versions(
            {10_000_001, 10_000_002}, [inventory]
        )
        self.assertEqual(rows_scanned, 4)
        self.assertEqual(selected[10_000_001][0], 2)
        self.assertEqual(selected[10_000_002][0], 1)
        self.assertNotIn(10_000_003, selected)


if __name__ == "__main__":
    unittest.main()
