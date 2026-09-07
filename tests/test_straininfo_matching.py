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
    serotype_like_key,
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

    def test_taxon_hint_accepts_lowercase_normalized_mention(self):
        self.assertEqual(
            scientific_name_hint("e. coli o157:h7"), ("e", "coli")
        )
        self.assertTrue(
            taxon_compatible("e. coli atcc 25922", "Escherichia coli")
        )
        self.assertFalse(
            taxon_compatible(
                "e. coli o157:h7",
                "Paraliobacillus ryukyuensis",
            )
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

    def test_serotype_alias_is_never_treated_as_a_strain_identifier(self):
        self.assertTrue(serotype_like_key("O157"))
        candidate = {
            "designation_key": "O157",
            "si_id": 362271,
            "taxon": "Paraliobacillus ryukyuensis",
            "type_strain": True,
        }
        self.assertEqual(
            resolve_candidates("O157", [candidate])["status"],
            "unmatched",
        )
        self.assertEqual(
            resolve_candidates("e. coli o157:h7", [candidate])["status"],
            "unmatched",
        )

    def test_short_other_designation_requires_taxonomic_support(self):
        result = resolve_candidates(
            "SA187",
            [
                {
                    "designation_key": "SA187",
                    "si_id": 63472,
                    "taxon": "Trichophyton mentagrophytes",
                    "type_strain": False,
                    "in_compact": False,
                    "in_detailed_deposit": False,
                    "in_detailed_other": True,
                }
            ],
        )
        self.assertEqual(result["status"], "unmatched")
        self.assertEqual(result["method"], "exact_weak_rejected")

    def test_short_type_strain_designation_remains_eligible(self):
        result = resolve_candidates(
            "B. amyloliquefaciens FZB42",
            [
                {
                    "designation_key": "FZB42",
                    "si_id": 378027,
                    "taxon": "Bacillus velezensis",
                    "type_strain": True,
                    "in_compact": False,
                    "in_detailed_deposit": False,
                }
            ],
        )
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["si_id"], 378027)

    def test_short_type_strain_alias_requires_compatible_genus(self):
        candidate = {
            "designation_key": "BL21",
            "si_id": 399905,
            "taxon": "Acidovorax soli",
            "type_strain": True,
            "in_compact": False,
            "in_detailed_deposit": False,
        }
        self.assertEqual(
            resolve_candidates("BL21", [candidate])["status"], "unmatched"
        )
        self.assertEqual(
            resolve_candidates("E. coli BL21", [candidate])["status"], "unmatched"
        )
        self.assertEqual(
            resolve_candidates("Acidovorax soli BL21", [candidate])["status"],
            "unique",
        )

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

    def test_batch_resolution_accepts_short_authoritative_contained_alias(self):
        aliases = pl.DataFrame(
            {
                "designation_key": ["FZB42"],
                "designation": ["FZB42"],
                "si_id": [378027],
                "taxon": ["Bacillus velezensis"],
                "type_strain": [True],
                "in_compact": [False],
                "in_detailed_deposit": [False],
            }
        )
        result = resolve_mentions(
            ["B. amyloliquefaciens strain FZB42"], aliases
        ).row(0, named=True)
        self.assertEqual(result["straininfo_status"], "unique")
        self.assertEqual(result["straininfo_method"], "contained_bounded")
        self.assertEqual(result["straininfo_si_id"], 378027)


if __name__ == "__main__":
    unittest.main()
