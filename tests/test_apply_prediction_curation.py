import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from apply_prediction_curation import apply_decisions  # noqa: E402


class ApplyPredictionCurationTests(unittest.TestCase):
    def test_applies_corrections_and_selects_remaining_strata(self):
        tasks = []
        for index, tier in enumerate(["high_grounded", "high_grounded", "boundary"]):
            candidate_id = f"candidate-{index}"
            tasks.append(
                {
                    "data": {
                        "candidate_id": candidate_id,
                        "review_tier": tier,
                        "sampled_relation": "STRAIN-COMPOUND:PRODUCES",
                        "review_rank": index + 1,
                        "text": f"sentence {index}",
                    },
                    "meta": {"seeded_entities": 2, "seeded_relations": 1},
                    "predictions": [
                        {
                            "result": [
                                {
                                    "id": "strain",
                                    "type": "labels",
                                    "value": {
                                        "start": 0,
                                        "end": 1,
                                        "text": "s",
                                        "labels": ["STRAIN"],
                                    },
                                },
                                {
                                    "id": "compound",
                                    "type": "labels",
                                    "value": {
                                        "start": 2,
                                        "end": 3,
                                        "text": "c",
                                        "labels": ["COMPOUND"],
                                    },
                                },
                                {
                                    "from_id": "strain",
                                    "to_id": "compound",
                                    "type": "relation",
                                    "labels": ["PRODUCES"],
                                },
                            ]
                        }
                    ],
                }
            )
        decisions = [
            {
                "candidate_id": "candidate-0",
                "status": "hard_negative",
                "drop_all_relations": True,
            }
        ]

        curated, remaining, report = apply_decisions(
            tasks,
            decisions,
            curator="test",
            remaining_per_stratum=1,
        )

        self.assertEqual(len(curated), 1)
        self.assertEqual(
            [
                result
                for result in curated[0]["annotations"][0]["result"]
                if result["type"] == "relation"
            ],
            [],
        )
        self.assertEqual(len(remaining), 2)
        self.assertEqual(report["deferred_unselected_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
