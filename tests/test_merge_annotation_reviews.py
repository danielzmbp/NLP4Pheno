import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from merge_annotation_reviews import merge_reviews, result_signature  # noqa: E402


def entity(result_id, text, label, start=0):
    return {
        "id": result_id,
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": start,
            "end": start + len(text),
            "text": text,
            "labels": [label],
        },
    }


class MergeAnnotationReviewsTests(unittest.TestCase):
    def test_signature_ignores_prediction_origin(self):
        manual = entity("e", "BC12", "STRAIN")
        prediction = copy.deepcopy(manual)
        manual["origin"] = "manual"
        prediction["origin"] = "prediction"

        self.assertEqual(result_signature([manual]), result_signature([prediction]))

    def test_merges_review_by_original_task_id_without_mutating_source(self):
        source = [
            {
                "id": 10,
                "data": {"text": "BC12 grew"},
                "annotations": [
                    {
                        "id": 4,
                        "project": 10,
                        "result": [entity("e", "BC12", "ORGANISM")],
                    }
                ],
            }
        ]
        review = [
            {
                "id": 1,
                "data": {"text": "BC12 grew", "original_task_id": "10"},
                "annotations": [
                    {
                        "id": 7,
                        "project": 1,
                        "completed_by": {"email": "reviewer@localhost"},
                        "result": [entity("e", "BC12", "STRAIN")],
                    }
                ],
            }
        ]
        original = copy.deepcopy(source)

        merged, report = merge_reviews(source, review, reviewer="reviewer@localhost")

        self.assertEqual(source, original)
        self.assertEqual(report["changed_tasks"], 1)
        annotation = merged[0]["annotations"][0]
        self.assertEqual(annotation["result"][0]["value"]["labels"], ["STRAIN"])
        self.assertEqual(annotation["completed_by"], "reviewer@localhost")
        self.assertTrue(annotation["ground_truth"])
        self.assertEqual(annotation["task"], 10)
        self.assertEqual(annotation["project"], 10)

    def test_rejects_relation_with_missing_endpoint(self):
        source = [{"id": 10, "data": {"text": "BC12 grew"}, "annotations": []}]
        review = [
            {
                "id": 1,
                "data": {"text": "BC12 grew", "original_task_id": "10"},
                "annotations": [
                    {
                        "id": 7,
                        "result": [
                            entity("e", "BC12", "STRAIN"),
                            {
                                "type": "relation",
                                "from_id": "e",
                                "to_id": "missing",
                                "labels": ["INHIBITS"],
                            },
                        ],
                    }
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "Missing reviewed relation endpoint"):
            merge_reviews(source, review)


if __name__ == "__main__":
    unittest.main()
