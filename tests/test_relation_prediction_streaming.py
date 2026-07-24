import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from format_relation_sentences import format_relation_sentences  # noqa: E402
from rel_prediction import (  # noqa: E402
    other_entity_type,
    stream_relation_predictions,
)


class RelationPredictionStreamingTests(unittest.TestCase):
    def test_formats_lazy_input_and_writes_only_positive_relation_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_file = root / "preds.parquet"
            formatted_file = root / "ner_preds.parquet"
            output_file = root / "relations.parquet"

            pl.DataFrame(
                {
                    "text": [
                        "S grows on acid",
                        "S resists copper",
                        "S occurs in soil",
                    ],
                    "start_strain": [0, 0, 0],
                    "end_strain": [1, 1, 1],
                    "word_strain": ["S", "S", "S"],
                    "start": [11, 10, 12],
                    "end": [15, 16, 16],
                    "word": ["acid", "copper", "soil"],
                    "score": [0.9, 0.9, 0.9],
                    "ner": ["COMPOUND", "COMPOUND", "ISOLATE"],
                }
            ).write_parquet(input_file)

            format_relation_sentences(input_file, formatted_file)

            def predictor(texts):
                return [
                    {
                        "label": "LABEL_1" if "resists" in text else "LABEL_0",
                        "score": 0.95,
                    }
                    for text in texts
                ]

            candidates, positives = stream_relation_predictions(
                formatted_file,
                output_file,
                "STRAIN-COMPOUND:RESISTS",
                predictor,
                row_batch_size=1,
            )

            self.assertEqual((candidates, positives), (2, 1))
            result = pl.read_parquet(output_file)
            self.assertEqual(result.height, 1)
            self.assertEqual(result["word"].to_list(), ["copper"])
            self.assertEqual(result["label"].to_list(), [1])
            self.assertEqual(
                result["rel"].to_list(),
                ["STRAIN-COMPOUND:RESISTS"],
            )
            self.assertEqual(
                result["re_result"].struct.field("label").to_list(),
                ["LABEL_1"],
            )
            self.assertFalse(
                formatted_file.with_name(".ner_preds.parquet.tmp").exists()
            )
            self.assertFalse(output_file.with_name(".relations.parquet.tmp").exists())

    def test_relation_entity_type_requires_exactly_one_strain(self):
        self.assertEqual(
            other_entity_type("STRAIN-PHENOTYPE:PRESENTS"),
            "PHENOTYPE",
        )
        with self.assertRaisesRegex(ValueError, "STRAIN"):
            other_entity_type("SPECIES-PHENOTYPE:PRESENTS")
