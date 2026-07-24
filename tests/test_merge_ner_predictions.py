import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from merge_ner_predictions import merge_ner_predictions  # noqa: E402


class MergeNerPredictionsTests(unittest.TestCase):
    def test_streaming_join_matches_existing_merge_semantics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strains_file = root / "strains.parquet"
            others_file = root / "others.parquet"
            output_file = root / "preds.parquet"

            pl.DataFrame(
                {
                    "text": ["alpha", "alpha", "beta", "gamma"],
                    "entity_group": ["B", "B", "B", "B"],
                    "score": [0.99, 0.98, 0.97, 0.96],
                    "word": ["S1", "S2", "S3", "S4"],
                    "start": [0, 6, 0, 0],
                    "end": [2, 8, 2, 2],
                }
            ).write_parquet(strains_file)
            pl.DataFrame(
                {
                    "text": ["alpha", "alpha", "beta", "beta", "gamma"],
                    "entity_group": ["B", "B", "B", "B", "B"],
                    "score": [0.91, 0.40, 0.88, 0.92, 0.95],
                    "word": ["acid", "weak", None, "soil", "motile"],
                    "start": [10, 20, 10, 20, 10],
                    "end": [14, 24, 14, 24, 16],
                    "ner": [
                        "COMPOUND",
                        "COMPOUND",
                        "ISOLATE",
                        "ISOLATE",
                        "PHENOTYPE",
                    ],
                }
            ).write_parquet(others_file)

            merge_ner_predictions(
                strains_file,
                others_file,
                output_file,
                cutoff=0.5,
            )

            result = pl.read_parquet(output_file).sort(
                ["text", "word_strain", "word"]
            )
            self.assertEqual(result.height, 4)
            self.assertEqual(
                result.columns,
                [
                    "text",
                    "entity_group_strain",
                    "score_strain",
                    "word_strain",
                    "start_strain",
                    "end_strain",
                    "entity_group",
                    "score",
                    "word",
                    "start",
                    "end",
                    "ner",
                ],
            )
            self.assertEqual(
                result.select(["text", "word_strain", "word"]).rows(),
                [
                    ("alpha", "S1", "acid"),
                    ("alpha", "S2", "acid"),
                    ("beta", "S3", "soil"),
                    ("gamma", "S4", "motile"),
                ],
            )
            self.assertFalse(output_file.with_name(".preds.parquet.tmp").exists())

    def test_rejects_missing_join_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strains_file = root / "strains.parquet"
            others_file = root / "others.parquet"

            pl.DataFrame({"text": ["alpha"], "word": ["S1"]}).write_parquet(
                strains_file
            )
            pl.DataFrame({"text": ["alpha"], "score": [0.9]}).write_parquet(
                others_file
            )

            with self.assertRaisesRegex(ValueError, "word"):
                merge_ner_predictions(
                    strains_file,
                    others_file,
                    root / "preds.parquet",
                    cutoff=0.5,
                )
