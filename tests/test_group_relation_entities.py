import tempfile
import unittest
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from group_relation_entities import (  # noqa: E402
    build_consensus_mapping,
    group_relation_entities,
)


class GroupRelationEntitiesTests(unittest.TestCase):
    def test_mapping_uses_frequency_and_keeps_entity_types_separate(self):
        counts = pl.DataFrame(
            {
                "ner": ["PHENOTYPE", "PHENOTYPE", "PHENOTYPE", "DISEASE"],
                "word_qc": [
                    "biofilm formation",
                    "biofilm formations",
                    "sporulation",
                    "biofilm formations",
                ],
                "count": [5, 2, 1, 4],
            }
        )

        mapping = build_consensus_mapping(
            counts,
            cutoff=95,
            workers=1,
            matrix_mb=1,
        )

        self.assertEqual(
            mapping.rows(),
            [("PHENOTYPE", "biofilm formations", "biofilm formation")],
        )

    def test_grouping_streams_rows_and_preserves_unmatched_words(self):
        frame = pl.DataFrame(
            {
                "ner": ["PHENOTYPE"] * 4,
                "word_qc": [
                    "biofilm formation",
                    "biofilm formation",
                    "biofilm formations",
                    "sporulation",
                ],
                "score": [0.9, 0.8, 0.7, 0.6],
                "label": [1, 1, 1, 1],
                "label_rel": [1, 1, 1, 1],
                "text": ["a", "b", "c", "d"],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "input.parquet"
            output_file = Path(temporary_directory) / "output.parquet"
            frame.write_parquet(input_file)

            group_relation_entities(
                input_file,
                output_file,
                cutoff=95,
                workers=1,
                matrix_mb=1,
            )
            result = pl.read_parquet(output_file)

        self.assertNotIn("label", result.columns)
        self.assertNotIn("label_rel", result.columns)
        self.assertIn("ner_score", result.columns)
        self.assertEqual(
            result.get_column("word_qc_group").to_list(),
            [
                "biofilm formation",
                "biofilm formation",
                "biofilm formation",
                "sporulation",
            ],
        )


if __name__ == "__main__":
    unittest.main()
