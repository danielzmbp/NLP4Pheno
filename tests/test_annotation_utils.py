import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from annotation_utils import (  # noqa: E402
    canonicalize_tasks,
    embedded_pmc_groups,
    merged_pmc_groups,
    select_annotation,
    source_group,
)


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

    def test_embedded_pmcids_group_new_annotation_tasks(self):
        tasks = [
            {"id": 10, "data": {"pmcid": "PMC123"}},
            {"id": 11, "data": {"pmcid": "PMC123"}},
            {"id": 12, "data": {}},
        ]
        groups = embedded_pmc_groups(tasks)
        self.assertEqual(groups, {"10": "PMC123", "11": "PMC123"})
        self.assertEqual(source_group(10, groups), source_group(11, groups))
        self.assertEqual(source_group(12, groups), "task:12")

    def test_embedded_pmcid_conflict_with_external_map_is_rejected(self):
        tasks = [{"id": 10, "data": {"pmcid": "PMC999"}}]
        with patch(
            "annotation_utils.load_unique_pmc_groups",
            return_value={"10": "PMC123"},
        ):
            with self.assertRaisesRegex(ValueError, "conflicting"):
                merged_pmc_groups(tasks, "matches.parquet")


if __name__ == "__main__":
    unittest.main()
