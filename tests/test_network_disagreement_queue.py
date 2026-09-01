import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_network_disagreement_queue import (  # noqa: E402
    build_queue,
    select_disagreements,
)


def relation_row(text: str, si_id: int, organism: str, rel: str) -> dict:
    strain = f"strain {si_id}"
    sentence = f"{strain} {rel.split(':')[1].lower()} {organism}."
    return {
        "text": sentence,
        "pmcid": f"PMC{si_id}",
        "article_version": "1",
        "paragraph": 1,
        "sentence_range": "0:1",
        "rel": rel,
        "score_rel": 0.97,
        "ner_score": 0.96,
        "score_strain": 0.99,
        "ner": "ORGANISM",
        "word": organism,
        "word_qc_group": organism.casefold(),
        "word_strain": strain,
        "word_strain_qc": strain,
        "straininfo_si_id": si_id,
        "straininfo_taxon": "test taxon",
        "start": sentence.index(organism),
        "end": sentence.index(organism) + len(organism),
        "start_strain": 0,
        "end_strain": len(strain),
        "ontology_status": "matched",
        "ontology": "NCBITAXON",
        "ontology_id": f"NCBITaxon:{si_id}",
        "ontology_label": organism,
        "ontology_node_id": f"NCBITaxon:{si_id}",
    }


class NetworkDisagreementQueueTests(unittest.TestCase):
    def test_selects_and_preannotates_both_directions(self):
        relation = "STRAIN-ORGANISM:INFECTS"
        common = relation_row("", 1, "common host", relation)
        old_only = relation_row("", 2, "old host", relation)
        new_only = relation_row("", 3, "new host", relation)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_file = root / "old.parquet"
            new_file = root / "new.parquet"
            pl.DataFrame([common, old_only]).write_parquet(old_file)
            pl.DataFrame([common, new_only]).write_parquet(new_file)
            selected, counts = select_disagreements(
                old_file,
                new_file,
                set(),
                relations={relation},
                per_relation_direction=10,
                max_text_chars=600,
            )

        self.assertEqual(counts, {"old_only_edges": 1, "new_only_edges": 1})
        self.assertEqual(
            set(selected.get_column("review_direction")),
            {"old_only", "new_only"},
        )
        tasks, _ = build_queue(selected)
        self.assertEqual(len(tasks), 2)
        self.assertTrue(
            all(task["predictions"][0]["result"] for task in tasks)
        )
        self.assertTrue(
            all(
                any(
                    result.get("type") == "relation"
                    for result in task["predictions"][0]["result"]
                )
                for task in tasks
            )
        )


if __name__ == "__main__":
    unittest.main()
