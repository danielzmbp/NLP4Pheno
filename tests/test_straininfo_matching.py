import sys
import unittest
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from straininfo_matching import (  # noqa: E402
    eligible_exact_key,
    has_complete_occurrence,
    resolve_candidates,
    resolve_mentions,
    scientific_name_hint,
    taxon_compatible,
)


class StrainInfoMatchingTests(unittest.TestCase):
    def test_rejects_prefix_of_a_longer_identifier(self):
        self.assertFalse(has_complete_occurrence("A. baumannii ATCC 17978", "ATCC1"))
        self.assertTrue(has_complete_occurrence("A. baumannii ATCC 17978", "ATCC17978"))

    def test_taxon_hint_accepts_abbreviated_genus(self):
        self.assertEqual(
            scientific_name_hint("A. baumannii ATCC 17978"), ("a", "baumannii")
        )
        self.assertTrue(
            taxon_compatible("A. baumannii ATCC 17978", "Acinetobacter baumannii")
        )
        self.assertFalse(
            taxon_compatible("A. baumannii ATCC 17978", "Bacillus pumilus")
        )

    def test_exact_alias_is_unique(self):
        result = resolve_candidates(
            "ATCC-25904",
            [
                {
                    "designation_key": "ATCC25904",
                    "si_id": 44736,
                    "taxon": "Staphylococcus aureus",
                }
            ],
        )
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["method"], "exact")

    def test_weak_exact_alias_is_not_resolved_without_taxonomy(self):
        self.assertFalse(eligible_exact_key("80"))
        result = resolve_candidates(
            "80",
            [{"designation_key": "80", "si_id": 1, "taxon": "Example species"}],
        )
        self.assertEqual(result["status"], "unmatched")
        self.assertEqual(result["method"], "exact_weak_rejected")

    def test_long_alphabetic_alias_can_match_exactly(self):
        result = resolve_candidates(
            "Sterne",
            [
                {
                    "designation_key": "STERNE",
                    "si_id": 51790,
                    "taxon": "Bacillus anthracis",
                }
            ],
        )
        self.assertEqual(result["status"], "unique")

    def test_taxonomy_resolves_alias_collision(self):
        candidates = [
            {
                "designation_key": "ATCC27853",
                "si_id": 1,
                "taxon": "Pseudomonas aeruginosa",
            },
            {"designation_key": "ATCC27853", "si_id": 2, "taxon": None},
        ]
        result = resolve_candidates("P. aeruginosa ATCC 27853", candidates)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["si_id"], 1)

    def test_compact_alias_wins_over_detailed_cross_reference(self):
        candidates = [
            {
                "designation_key": "ATCC25923",
                "si_id": 1,
                "taxon": "Staphylococcus aureus",
                "in_compact": True,
            },
            {
                "designation_key": "ATCC25923",
                "si_id": 2,
                "taxon": "Staphylococcus aureus",
                "in_compact": False,
            },
        ]
        result = resolve_candidates("ATCC 25923", candidates)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["si_id"], 1)

    def test_genus_hint_rejects_wrong_lab_alias(self):
        result = resolve_candidates(
            "Streptomyces PC 12",
            [
                {
                    "designation_key": "PC12",
                    "si_id": 1,
                    "taxon": "Emiliania huxleyi",
                }
            ],
        )
        self.assertEqual(result["status"], "unmatched")

    def test_contradictory_taxon_rejects_contained_alias(self):
        result = resolve_candidates(
            "A. baumannii ATCC 17978",
            [
                {
                    "designation_key": "ATCC17978",
                    "si_id": 1,
                    "taxon": "Bacillus pumilus",
                }
            ],
        )
        self.assertEqual(result["status"], "unmatched")

    def test_ambiguous_alias_is_not_guessed(self):
        candidates = [
            {"designation_key": "DSM123", "si_id": 1, "taxon": None},
            {"designation_key": "DSM123", "si_id": 2, "taxon": None},
        ]
        self.assertEqual(
            resolve_candidates("DSM 123", candidates)["status"], "ambiguous"
        )

    def test_batch_resolution_retains_unmatched_mentions(self):
        aliases = pl.DataFrame(
            {
                "designation_key": ["ATCC17978"],
                "designation": ["ATCC 17978"],
                "si_id": [13527],
                "taxon": ["Acinetobacter baumannii"],
                "type_strain": [False],
            }
        )
        result = resolve_mentions(
            ["A. baumannii ATCC 17978", "unknown isolate"], aliases
        ).sort("mention")
        self.assertEqual(result.height, 2)
        matched = result.filter(pl.col("straininfo_status") == "unique").row(
            0, named=True
        )
        self.assertEqual(matched["straininfo_si_id"], 13527)
        self.assertEqual(
            result.filter(pl.col("straininfo_status") == "unmatched").height, 1
        )

    def test_batch_resolution_handles_weak_exact_alias(self):
        aliases = pl.DataFrame(
            {
                "designation_key": ["80"],
                "designation": ["80"],
                "si_id": [1],
                "taxon": ["Example species"],
                "type_strain": [False],
            }
        )
        result = resolve_mentions(["80"], aliases).row(0, named=True)
        self.assertEqual(result["straininfo_status"], "unmatched")
        self.assertEqual(result["straininfo_method"], "exact_weak_rejected")


if __name__ == "__main__":
    unittest.main()
