import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from apply_annotation_corrections import apply_corrections  # noqa: E402


class ApplyAnnotationCorrectionsTests(unittest.TestCase):
    def test_applies_entity_and_relation_changes_without_mutating_source(self):
        text = "Strain X inhibits yeast, not mold."
        source = [
            {
                "id": 1,
                "data": {
                    "candidate_id": "candidate-1",
                    "human_review_rank": 1,
                    "text": text,
                },
                "annotations": [
                    {
                        "id": 10,
                        "result": [
                            {
                                "id": "strain",
                                "type": "labels",
                                "value": {
                                    "start": 0,
                                    "end": 8,
                                    "text": "Strain X",
                                    "labels": ["STRAIN"],
                                },
                            },
                            {
                                "id": "yeast",
                                "type": "labels",
                                "value": {
                                    "start": 18,
                                    "end": 24,
                                    "text": "yeast,",
                                    "labels": ["ORGANISM"],
                                },
                            },
                            {
                                "id": "mold",
                                "type": "labels",
                                "value": {
                                    "start": 29,
                                    "end": 33,
                                    "text": "mold",
                                    "labels": ["ORGANISM"],
                                },
                            },
                            {
                                "from_id": "strain",
                                "to_id": "mold",
                                "type": "relation",
                                "labels": ["INHIBITS"],
                            },
                        ],
                    }
                ],
            }
        ]
        corrections = [
            {
                "candidate_id": "candidate-1",
                "notes": "Remove the negated relation and correct the span.",
                "drop_entity_ids": ["mold"],
                "edit_entities": [
                    {
                        "id": "yeast",
                        "text": "yeast",
                        "label": "ORGANISM",
                    }
                ],
                "add_relations": [["strain", "yeast", "INHIBITS"]],
            }
        ]

        corrected, report = apply_corrections(source, corrections)
        results = corrected[0]["annotations"][0]["result"]

        self.assertEqual(source[0]["annotations"][0]["result"][1]["value"]["text"], "yeast,")
        self.assertEqual(
            [result["id"] for result in results if result["type"] == "labels"],
            ["strain", "yeast"],
        )
        self.assertEqual(
            [result["value"]["text"] for result in results if result["type"] == "labels"],
            ["Strain X", "yeast"],
        )
        self.assertEqual(
            [
                (result["from_id"], result["to_id"], result["labels"][0])
                for result in results
                if result["type"] == "relation"
            ],
            [("strain", "yeast", "INHIBITS")],
        )
        self.assertTrue(corrected[0]["annotations"][0]["ground_truth"])
        self.assertTrue(corrected[0]["meta"]["second_pass_reviewed"])
        self.assertEqual(report["entities_dropped"], 1)
        self.assertEqual(report["entities_edited"], 1)
        self.assertEqual(report["relations_dropped"], 1)
        self.assertEqual(report["relations_added"], 1)


if __name__ == "__main__":
    unittest.main()
