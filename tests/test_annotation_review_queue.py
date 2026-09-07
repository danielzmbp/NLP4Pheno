import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_annotation_review_queue import build_queue  # noqa: E402


def task(task_id, text, entities, relations=()):
    results = []
    for result_id, surface, label in entities:
        start = text.index(surface)
        results.append(
            {
                "id": result_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": start + len(surface),
                    "text": surface,
                    "labels": [label],
                },
            }
        )
    for source, target, label in relations:
        results.append(
            {
                "from_id": source,
                "to_id": target,
                "type": "relation",
                "direction": "right",
                "labels": [label],
            }
        )
    return {
        "id": task_id,
        "data": {"text": text},
        "annotations": [{"result": results}],
    }


class AnnotationReviewQueueTests(unittest.TestCase):
    def test_builds_editable_suggestions_for_each_high_confidence_category(self):
        tasks = []
        for index in range(5):
            tasks.append(task(index, "BC12 grew", [("s", "BC12", "STRAIN")]))
        tasks.append(task(10, "BC12 grew", [("s", "BC12", "ORGANISM")]))

        for index in range(5):
            tasks.append(
                task(
                    20 + index,
                    "S1 inhibits E. coli",
                    [("s", "S1", "STRAIN"), ("e", "E. coli", "SPECIES")],
                )
            )
        tasks.append(task(30, "E. coli grows", []))

        for index in range(4):
            tasks.append(
                task(
                    40 + index,
                    "S2 grows on agar",
                    [("s", "S2", "STRAIN"), ("m", "agar", "MEDIUM")],
                    [("s", "m", "GROWS_ON")],
                )
            )
        tasks.append(
            task(
                50,
                "S2 grows on agar",
                [("s", "S2", "STRAIN"), ("m", "agar", "MEDIUM")],
            )
        )

        queue, issues, summary = build_queue(
            tasks,
            configured_relations={"STRAIN-MEDIUM:GROWS_ON"},
            max_tasks=20,
            conflict_min_support=5,
            conflict_min_purity=0.8,
            omission_specific_support=5,
            omission_general_support=99,
            relation_min_positive=4,
            relation_min_rate=0.8,
        )

        categories = {item["category"] for item in issues}
        self.assertEqual(
            categories,
            {"entity_label_conflict", "entity_omission", "relation_omission"},
        )
        self.assertEqual(summary["selected_tasks"], len(queue))

        by_id = {item["data"]["original_task_id"]: item for item in queue}
        conflict_results = by_id["10"]["predictions"][0]["result"]
        self.assertEqual(conflict_results[0]["value"]["labels"], ["STRAIN"])
        omission_results = by_id["30"]["predictions"][0]["result"]
        self.assertEqual(omission_results[0]["value"]["labels"], ["SPECIES"])
        relation_results = by_id["50"]["predictions"][0]["result"]
        self.assertTrue(
            any(
                result.get("type") == "relation"
                and result.get("labels") == ["GROWS_ON"]
                for result in relation_results
            )
        )


if __name__ == "__main__":
    unittest.main()
