import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from audit_annotations import annotation_signature, audit_export  # noqa: E402


class AnnotationAuditTests(unittest.TestCase):
    def test_relation_signature_ignores_opaque_span_ids(self):
        def task(left_id, right_id):
            return {
                "annotations": [
                    {
                        "result": [
                            {
                                "id": left_id,
                                "type": "labels",
                                "value": {
                                    "start": 0,
                                    "end": 1,
                                    "text": "S",
                                    "labels": ["STRAIN"],
                                },
                            },
                            {
                                "id": right_id,
                                "type": "labels",
                                "value": {
                                    "start": 2,
                                    "end": 3,
                                    "text": "X",
                                    "labels": ["SPECIES"],
                                },
                            },
                            {
                                "type": "relation",
                                "from_id": left_id,
                                "to_id": right_id,
                                "labels": ["INHIBITS"],
                            },
                        ]
                    }
                ]
            }

        self.assertEqual(annotation_signature(task("a", "b")), annotation_signature(task("x", "y")))

    def test_reports_classifier_support_and_annotator_profiles(self):
        tasks = [
            {
                "id": 1,
                "data": {"text": "S inhibits X"},
                "annotations": [
                    {
                        "completed_by": 7,
                        "result": [
                            {
                                "id": "s",
                                "type": "labels",
                                "value": {
                                    "start": 0,
                                    "end": 1,
                                    "text": "S",
                                    "labels": ["STRAIN"],
                                },
                            },
                            {
                                "id": "x",
                                "type": "labels",
                                "value": {
                                    "start": 11,
                                    "end": 12,
                                    "text": "X",
                                    "labels": ["SPECIES"],
                                },
                            },
                            {
                                "type": "relation",
                                "from_id": "s",
                                "to_id": "x",
                                "labels": ["INHIBITS"],
                            },
                        ],
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "annotations.json"
            source.write_text(json.dumps(tasks))
            report = audit_export(
                source,
                {"STRAIN", "SPECIES"},
                {"STRAIN-SPECIES:INHIBITS"},
            )

        self.assertEqual(
            report["relations"]["configured_support"]["STRAIN-SPECIES:INHIBITS"],
            {"positive_labels": 1, "positive_tasks": 1},
        )
        self.assertEqual(report["annotations"]["annotator_profiles"]["7"]["tasks"], 1)
        self.assertEqual(report["integrity"]["same_entity_type_overlap_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
