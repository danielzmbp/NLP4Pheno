import sys
import unittest
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from ner_postprocess import merge_entities  # noqa: E402
from relation_prediction_utils import add_formatted_text  # noqa: E402


class PredictionPostprocessTests(unittest.TestCase):
    def test_merge_entities_consumes_all_inside_tokens(self):
        entities = [
            {"entity_group": "B", "start": 0, "end": 2, "score": 0.9, "word": "A"},
            {"entity_group": "I", "start": 3, "end": 5, "score": 0.8, "word": "B"},
            {"entity_group": "I", "start": 6, "end": 8, "score": 0.7, "word": "C"},
        ]
        merged = merge_entities(entities)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (0, 8))

    def test_relation_format_does_not_duplicate_sentence(self):
        frame = pl.DataFrame(
            {
                "text": ["S grows on agar", "agar supports S"],
                "start_strain": [0, 14],
                "end_strain": [1, 15],
                "start": [11, 0],
                "end": [15, 4],
                "ner": ["MEDIUM", "MEDIUM"],
            }
        )
        formatted = add_formatted_text(frame)["formatted_text"].to_list()
        self.assertEqual(formatted[0], "@STRAIN$ grows on @MEDIUM$")
        self.assertEqual(formatted[1], "@MEDIUM$ supports @STRAIN$")


if __name__ == "__main__":
    unittest.main()
