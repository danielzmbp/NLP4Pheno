import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from reconcile_relation_predictions import (  # noqa: E402
    reconcile_relation_predictions,
)


def prediction(
    *,
    text: str,
    ner: str,
    rel: str,
    entity: str,
    ner_score: float,
    strain_id: int | None = 10,
    strain_taxon: str | None = "Bacillus subtilis",
) -> dict:
    return {
        "text": text,
        "ner": ner,
        "ner_score": ner_score,
        "score_rel": 0.95,
        "score_strain": 0.98,
        "rel": rel,
        "straininfo_si_id": strain_id,
        "straininfo_taxon": strain_taxon,
        "word_qc_group": entity,
    }


class ReconcileRelationPredictionsTests(unittest.TestCase):
    def test_keeps_best_entity_type_and_filters_impossible_same_taxon(self):
        rows = [
            prediction(
                text="host evidence",
                ner="ISOLATE",
                rel="STRAIN-ISOLATE:INHABITS",
                entity="human",
                ner_score=0.55,
            ),
            prediction(
                text="host evidence",
                ner="ORGANISM",
                rel="STRAIN-ORGANISM:INHABITS",
                entity="human",
                ner_score=0.95,
            ),
            prediction(
                text="self evidence",
                ner="ORGANISM",
                rel="STRAIN-ORGANISM:INHABITS",
                entity="b. subtilis",
                ner_score=0.99,
            ),
            prediction(
                text="conspecific inhibition can be real",
                ner="ORGANISM",
                rel="STRAIN-ORGANISM:INHIBITS",
                entity="Bacillus subtilis",
                ner_score=0.90,
            ),
            prediction(
                text="opposite direction",
                ner="COMPOUND",
                rel="COMPOUND-STRAIN:INHIBITS",
                entity="bacillus subtilis",
                ner_score=0.85,
            ),
            prediction(
                text="unmatched strain",
                ner="ORGANISM",
                rel="STRAIN-ORGANISM:INHABITS",
                entity="mouse",
                ner_score=0.80,
                strain_id=None,
                strain_taxon=None,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_file = root / "input.parquet"
            output_file = root / "output.parquet"
            summary_file = root / "summary.json"
            pl.DataFrame(rows).write_parquet(input_file)

            summary = reconcile_relation_predictions(
                input_file,
                output_file,
                summary_file,
            )
            result = pl.read_parquet(output_file).sort("text")
            stored_summary = json.loads(summary_file.read_text())

        self.assertEqual(summary, stored_summary)
        self.assertEqual(result.height, 4)
        self.assertEqual(
            result.filter(pl.col("text") == "host evidence")
            .get_column("ner")
            .to_list(),
            ["ORGANISM"],
        )
        self.assertEqual(
            result.filter(pl.col("text") == "self evidence").height,
            0,
        )
        self.assertEqual(
            result.filter(pl.col("text") == "conspecific inhibition can be real").height,
            1,
        )
        self.assertEqual(
            result.filter(pl.col("text") == "opposite direction").height,
            1,
        )
        self.assertEqual(
            result.filter(pl.col("text") == "unmatched strain").height,
            1,
        )
        self.assertEqual(summary["input_rows"], 6)
        self.assertEqual(summary["output_rows"], 4)
        self.assertEqual(summary["conflicting_edges"], 1)
        self.assertEqual(summary["removed_competing_type_rows"], 1)
        self.assertEqual(summary["removed_same_taxon_rows"], 1)


if __name__ == "__main__":
    unittest.main()
