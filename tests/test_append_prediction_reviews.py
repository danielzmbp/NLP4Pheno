import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from append_prediction_reviews import append_reviewed_tasks  # noqa: E402


class AppendPredictionReviewsTests(unittest.TestCase):
    def test_appends_reviewed_task_with_new_id_and_provenance(self):
        source = [{"id": 10, "data": {"text": "old"}, "annotations": []}]
        review = {
            "id": 4,
            "data": {"text": "new strain", "candidate_id": "candidate-1"},
            "annotations": [
                {
                    "id": 7,
                    "project": 4,
                    "ground_truth": True,
                    "result": [
                        {
                            "id": "entity",
                            "type": "labels",
                            "value": {
                                "start": 0,
                                "end": 10,
                                "text": "new strain",
                                "labels": ["STRAIN"],
                            },
                        }
                    ],
                }
            ],
        }

        output, report = append_reviewed_tasks(
            source,
            [("review.json", [review])],
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(output[1]["id"], 11)
        self.assertEqual(output[1]["annotations"][0]["task"], 11)
        self.assertEqual(
            output[1]["annotations"][0]["review_provenance"]["candidate_id"],
            "candidate-1",
        )
        self.assertEqual(report["added_tasks"], 1)

    def test_rejects_unsubmitted_reviews(self):
        with self.assertRaisesRegex(ValueError, "not submitted"):
            append_reviewed_tasks(
                [],
                [
                    (
                        "review.json",
                        [
                            {
                                "data": {
                                    "text": "new",
                                    "candidate_id": "candidate-1",
                                },
                                "annotations": [],
                            }
                        ],
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
