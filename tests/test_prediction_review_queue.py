import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_prediction_review_queue import (  # noqa: E402
    build_queue,
    build_results,
    select_candidate_sentences,
)


class PredictionReviewQueueTests(unittest.TestCase):
    def test_selection_orders_high_confidence_tiers_first(self):
        rows = []
        tiers = [
            ("high_grounded", 0.97, 0.96, 0.98, "matched"),
            ("high_unresolved", 0.92, 0.96, 0.98, "unmatched"),
            ("relation_boundary", 0.60, 0.96, 0.98, "matched"),
            ("entity_tension", 0.90, 0.70, 0.98, "matched"),
        ]
        for index, (_, rel_score, ner_score, strain_score, ontology_status) in enumerate(tiers):
            text = f"Strain X evidence sentence {index}."
            rows.append(
                {
                    "text": text,
                    "pmcid": f"PMC{index + 1}",
                    "article_version": "v1",
                    "paragraph": index,
                    "sentence_range": "0:30",
                    "rel": "STRAIN-ORGANISM:SYMBIONT_OF",
                    "score_rel": rel_score,
                    "ner_score": ner_score,
                    "score_strain": strain_score,
                    "ner": "ORGANISM",
                    "word": "evidence",
                    "word_qc_group": "evidence",
                    "word_strain": "Strain X",
                    "word_strain_qc": "strain x",
                    "straininfo_si_id": 10,
                    "straininfo_taxon": "Example species",
                    "straininfo_status": "matched",
                    "straininfo_method": "exact",
                    "start": 9,
                    "end": 17,
                    "start_strain": 0,
                    "end_strain": 8,
                    "ontology_status": ontology_status,
                    "ontology_candidate_count": 1,
                    "ontology": "NCBITAXON",
                    "ontology_id": "NCBITaxon:1",
                    "ontology_label": "evidence",
                    "ontology_match_method": "direct_exact",
                    "ontology_match_confidence": 1.0,
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.parquet"
            pl.DataFrame(rows).write_parquet(path)
            selected = select_candidate_sentences(
                path,
                set(),
                per_relation_tier=1,
                seed=1,
                max_text_chars=100,
            )

        self.assertEqual(
            selected.get_column("review_tier").to_list(),
            [name for name, *_ in tiers],
        )

    def test_selection_can_focus_on_underrepresented_relations(self):
        rows = []
        for index, relation in enumerate(
            ["STRAIN-ORGANISM:INHIBITS", "STRAIN-COMPOUND:PRODUCES"]
        ):
            text = f"Strain X evidence sentence {index}."
            rows.append(
                {
                    "text": text,
                    "pmcid": f"PMC{index + 1}",
                    "article_version": "v1",
                    "paragraph": index,
                    "sentence_range": "0:30",
                    "rel": relation,
                    "score_rel": 0.97,
                    "ner_score": 0.96,
                    "score_strain": 0.98,
                    "ner": "ORGANISM" if index == 0 else "COMPOUND",
                    "word": "evidence",
                    "word_qc_group": "evidence",
                    "word_strain": "Strain X",
                    "word_strain_qc": "strain x",
                    "straininfo_si_id": 10,
                    "straininfo_taxon": "Example species",
                    "straininfo_status": "matched",
                    "straininfo_method": "exact",
                    "start": 9,
                    "end": 17,
                    "start_strain": 0,
                    "end_strain": 8,
                    "ontology_status": "matched",
                    "ontology_candidate_count": 1,
                    "ontology": "NCBITAXON",
                    "ontology_id": "NCBITaxon:1",
                    "ontology_label": "evidence",
                    "ontology_match_method": "direct_exact",
                    "ontology_match_confidence": 1.0,
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.parquet"
            pl.DataFrame(rows).write_parquet(path)
            selected = select_candidate_sentences(
                path,
                set(),
                per_relation_tier=2,
                seed=1,
                max_text_chars=100,
                include_relations={"STRAIN-ORGANISM:INHIBITS"},
            )

        self.assertEqual(
            selected.get_column("rel").to_list(),
            ["STRAIN-ORGANISM:INHIBITS"],
        )

    def test_results_include_all_entities_and_preserve_relation_direction(self):
        text = "Strain X produces glucose and inhibits yeast."
        strain_start = text.index("Strain X")
        glucose_start = text.index("glucose")
        yeast_start = text.index("yeast")
        ner_rows = [
            {
                "start_strain": strain_start,
                "end_strain": strain_start + len("Strain X"),
                "word_strain": "Strain X",
                "score_strain": 0.99,
                "start": glucose_start,
                "end": glucose_start + len("glucose"),
                "word": "glucose",
                "ner": "COMPOUND",
                "ner_score": 0.98,
            },
            {
                "start_strain": strain_start,
                "end_strain": strain_start + len("Strain X"),
                "word_strain": "Strain X",
                "score_strain": 0.99,
                "start": yeast_start,
                "end": yeast_start + len("yeast"),
                "word": "yeast",
                "ner": "ORGANISM",
                "ner_score": 0.97,
            },
        ]
        relation_rows = [
            {
                **ner_rows[0],
                "rel": "STRAIN-COMPOUND:PRODUCES",
            },
            {
                **ner_rows[1],
                "rel": "STRAIN-ORGANISM:INHIBITS",
            },
        ]
        results, counts = build_results(text, ner_rows, relation_rows)
        entities = [result for result in results if result["type"] == "labels"]
        relations = [result for result in results if result["type"] == "relation"]

        self.assertEqual(counts["entities"], 3)
        self.assertEqual(counts["relations"], 2)
        self.assertEqual(
            sorted(result["value"]["labels"][0] for result in entities),
            ["COMPOUND", "ORGANISM", "STRAIN"],
        )
        self.assertEqual(
            sorted(result["labels"][0] for result in relations),
            ["INHIBITS", "PRODUCES"],
        )

    def test_queue_keeps_provenance_and_seed_counts(self):
        text = "Strain X produces glucose."
        selected = pl.DataFrame(
            {
                "text": [text],
                "pmcid": ["PMC1"],
                "article_version": ["v1"],
                "paragraph": [2],
                "sentence_range": ["0:26"],
                "rel": ["STRAIN-COMPOUND:PRODUCES"],
                "score_rel": [0.99],
                "ner_score": [0.98],
                "score_strain": [0.97],
                "review_tier": ["high_grounded"],
                "ontology_status": ["matched"],
                "ontology": ["CHEBI"],
                "ontology_id": ["CHEBI:17234"],
                "ontology_label": ["glucose"],
                "straininfo_si_id": [10],
                "straininfo_taxon": ["Example species"],
            }
        )
        ner = pl.DataFrame(
            {
                "text": [text],
                "start_strain": [0],
                "end_strain": [8],
                "word_strain": ["Strain X"],
                "score_strain": [0.97],
                "start": [18],
                "end": [25],
                "word": ["glucose"],
                "ner": ["COMPOUND"],
                "ner_score": [0.98],
            }
        )
        relations = ner.with_columns(
            pl.lit("STRAIN-COMPOUND:PRODUCES").alias("rel"),
            pl.lit(0.99).alias("score_rel"),
            pl.lit("glucose").alias("word_qc_group"),
            pl.lit(10).alias("straininfo_si_id"),
            pl.lit("Example species").alias("straininfo_taxon"),
            pl.lit("matched").alias("ontology_status"),
            pl.lit("CHEBI").alias("ontology"),
            pl.lit("CHEBI:17234").alias("ontology_id"),
            pl.lit("glucose").alias("ontology_label"),
        )

        tasks, issues = build_queue(selected, ner, relations)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["data"]["pmcid"], "PMC1")
        self.assertEqual(tasks[0]["meta"]["seeded_entities"], 2)
        self.assertEqual(tasks[0]["meta"]["seeded_relations"], 1)
        self.assertEqual(issues[0]["candidate_id"], tasks[0]["data"]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
