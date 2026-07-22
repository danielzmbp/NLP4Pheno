import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from annotation_utils import canonicalize_tasks, select_annotation, source_group  # noqa: E402


class AnnotationSelectionTests(unittest.TestCase):
    def test_latest_active_annotation_is_selected(self):
        task = {
            "annotations": [
                {"id": 1, "updated_at": "2025-01-01T00:00:00Z", "result": []},
                {"id": 2, "updated_at": "2025-01-02T00:00:00Z", "result": [1]},
            ]
        }
        self.assertEqual(select_annotation(task)["id"], 2)

    def test_ground_truth_annotation_takes_priority(self):
        task = {
            "annotations": [
                {
                    "id": 1,
                    "ground_truth": True,
                    "updated_at": "2025-01-01T00:00:00Z",
                },
                {"id": 2, "updated_at": "2025-01-02T00:00:00Z"},
            ]
        }
        self.assertEqual(select_annotation(task)["id"], 1)

    def test_cancelled_annotations_are_ignored(self):
        task = {
            "annotations": [
                {"id": 1, "was_cancelled": True, "updated_at": "2025-01-03"},
                {"id": 2, "updated_at": "2025-01-02"},
            ]
        }
        self.assertEqual(select_annotation(task)["id"], 2)

    def test_canonicalization_keeps_tasks_without_annotations(self):
        tasks = canonicalize_tasks([{"id": 1}, {"id": 2, "annotations": []}])
        self.assertEqual(tasks, [{"id": 1, "annotations": []}, {"id": 2, "annotations": []}])

    def test_source_groups_use_only_linked_pmcids(self):
        groups = {"1": "PMC123"}
        self.assertEqual(source_group(1, groups), "pmcid:PMC123")
        self.assertEqual(source_group(2, groups), "task:2")


if __name__ == "__main__":
    unittest.main()
