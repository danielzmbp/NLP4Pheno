import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from ner_data import build_dataset, task_to_example  # noqa: E402


class NerDataTests(unittest.TestCase):
    def test_span_offsets_become_bio_tags(self):
        task = {
            "id": 9,
            "data": {"text": "E. coli ATCC 123 grows."},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "s",
                            "type": "labels",
                            "value": {
                                "start": 8,
                                "end": 16,
                                "text": "ATCC 123",
                                "labels": ["STRAIN"],
                            },
                        }
                    ]
                }
            ],
        }
        example = task_to_example(task, "STRAIN")
        self.assertEqual(example["tokens"], ["E", ".", "coli", "ATCC", "123", "grows", "."])
        self.assertEqual(example["ner_tags"], ["O", "O", "O", "B", "I", "O", "O"])

    def test_builder_writes_jsonl_and_bio(self):
        task = {"id": 1, "data": {"text": "plain text"}, "annotations": [{"result": []}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tasks.json"
            source.write_text(json.dumps([task]))
            stats = build_dataset(
                source,
                label="STRAIN",
                json_output=root / "train.json",
                bio_output=root / "train.txt",
            )
            self.assertEqual(stats, {"tasks": 1, "tokens": 2, "entity_tokens": 0})
            self.assertEqual(json.loads((root / "train.json").read_text()), {"id": "1", "tokens": ["plain", "text"], "ner_tags": ["O", "O"]})
            self.assertEqual((root / "train.txt").read_text(), "plain\tO\ntext\tO\n\n")

    def test_slash_separated_adjacent_entities_are_distinct_tokens(self):
        task = {
            "id": 2,
            "data": {"text": "RT027/CD196"},
            "annotations": [
                {
                    "result": [
                        {"id": "a", "type": "labels", "value": {"start": 0, "end": 5, "labels": ["STRAIN"]}},
                        {"id": "b", "type": "labels", "value": {"start": 6, "end": 11, "labels": ["STRAIN"]}},
                    ]
                }
            ],
        }
        example = task_to_example(task, "STRAIN")
        self.assertEqual(example["tokens"], ["RT027", "/", "CD196"])
        self.assertEqual(example["ner_tags"], ["B", "O", "B"])

    def test_hyphenated_source_text_is_preserved(self):
        task = {
            "id": 3,
            "data": {"text": "beta-lactam-resistant strain"},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "p",
                            "type": "labels",
                            "value": {
                                "start": 0,
                                "end": 21,
                                "text": "beta-lactam-resistant",
                                "labels": ["PHENOTYPE"],
                            },
                        }
                    ]
                }
            ],
        }
        example = task_to_example(task, "PHENOTYPE")
        self.assertEqual(
            example["tokens"],
            ["beta", "-", "lactam", "-", "resistant", "strain"],
        )
        self.assertEqual(example["ner_tags"], ["B", "I", "I", "I", "I", "O"])


if __name__ == "__main__":
    unittest.main()
