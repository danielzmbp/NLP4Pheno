import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from relation_data import (  # noqa: E402
    build_relation_rows,
    mark_entity_pair,
    relation_membership,
    split_by_task,
)


class RelationDataTests(unittest.TestCase):
    def test_markers_use_offsets_not_global_string_replacement(self):
        text = "X grew near X and agar."
        marked = mark_entity_pair(
            text,
            {"start": 0, "end": 1, "label": "STRAIN"},
            {"start": 18, "end": 22, "label": "MEDIUM"},
        )
        self.assertEqual(marked, "@STRAIN$ grew near X and @MEDIUM$.")

    def test_marking_preserves_hyphens_outside_entities(self):
        text = "S is beta-lactam resistant"
        marked = mark_entity_pair(
            text,
            {"start": 0, "end": 1, "label": "STRAIN"},
            {"start": 5, "end": 16, "label": "COMPOUND"},
        )
        self.assertEqual(marked, "@STRAIN$ is @COMPOUND$ resistant")

    def test_build_rows_preserves_task_and_relation_direction(self):
        task = {
            "id": 7,
            "data": {"text": "S grows on agar"},
            "annotations": [
                {
                    "result": [
                        {"id": "s", "type": "labels", "value": {"start": 0, "end": 1, "text": "S", "labels": ["STRAIN"]}},
                        {"id": "m", "type": "labels", "value": {"start": 11, "end": 15, "text": "agar", "labels": ["MEDIUM"]}},
                        {"type": "relation", "from_id": "s", "to_id": "m", "labels": ["GROWS_ON"]},
                    ]
                }
            ],
        }
        frame, stats = build_relation_rows([task])
        positive = frame[frame["relations"] == "GROWS_ON"].iloc[0]
        self.assertEqual(positive["task_id"], 7)
        self.assertEqual(positive["pair_type"], "STRAIN-MEDIUM")
        self.assertEqual(stats["skipped_overlapping_positive_pairs"], 0)

    def test_multi_label_pair_is_one_positive_for_each_classifier(self):
        task = {
            "id": 8,
            "data": {"text": "S affects agar"},
            "annotations": [
                {
                    "result": [
                        {"id": "s", "type": "labels", "value": {"start": 0, "end": 1, "text": "S", "labels": ["STRAIN"]}},
                        {"id": "m", "type": "labels", "value": {"start": 10, "end": 14, "text": "agar", "labels": ["MEDIUM"]}},
                        {"type": "relation", "from_id": "s", "to_id": "m", "labels": ["INHIBITS", "PROMOTES"]},
                    ]
                }
            ],
        }
        frame, _ = build_relation_rows([task])
        pair = frame[(frame["from_id"] == "s") & (frame["to_id"] == "m")]
        self.assertEqual(len(pair), 1)
        self.assertTrue(relation_membership(pair["relations"], "INHIBITS").iloc[0])
        self.assertTrue(relation_membership(pair["relations"], "PROMOTES").iloc[0])

    def test_split_keeps_tasks_disjoint(self):
        rows = []
        for task_id in range(20):
            rows.append({"task_id": task_id, "binary_label": task_id % 2, "sentence": str(task_id)})
            rows.append({"task_id": task_id, "binary_label": 0, "sentence": str(task_id)})
        splits = split_by_task(pd.DataFrame(rows), test_and_dev_size=0.3, seed=3)
        task_sets = {name: set(frame.task_id) for name, frame in splits.items()}
        self.assertFalse(task_sets["train"] & task_sets["test"])
        self.assertFalse(task_sets["train"] & task_sets["dev"])
        self.assertFalse(task_sets["test"] & task_sets["dev"])


if __name__ == "__main__":
    unittest.main()
