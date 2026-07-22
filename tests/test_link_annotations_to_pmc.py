import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from link_annotations_to_pmc import match_annotations  # noqa: E402


class AnnotationPmcLinkTests(unittest.TestCase):
    def test_literal_matches_keep_ambiguity_and_containment(self):
        tasks = [
            {"id": 1, "data": {"text": "A strain grows."}},
            {"id": 2, "data": {"text": "Unique result."}},
            {"id": 3, "data": {"text": "Not present."}},
        ]
        corpus = pl.DataFrame(
            {
                "pmcid": ["PMC1", "PMC2", "PMC3"],
                "article_version": ["PMC1.1", "PMC2.1", "PMC3.1"],
                "section": ["body", "body", "body"],
                "paragraph": [1, 1, 1],
                "sentence_range": ["1-2", "1", "1"],
                "text": [
                    "Before. A strain grows.",
                    "A strain grows.",
                    "Unique result.",
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.parquet"
            corpus.write_parquet(path)
            matches, summary = match_annotations(tasks, [path])

        self.assertEqual(summary["tasks"], {
            "total": 3,
            "matched": 2,
            "unmatched": 1,
            "unique_pmcid": 1,
            "ambiguous_pmcid": 1,
        })
        task_one = matches.filter(pl.col("task_id") == "1")
        self.assertEqual(set(task_one["status"]), {"ambiguous_pmcid"})
        self.assertEqual(set(task_one["match_kind"]), {"contained_exact", "row_exact"})


if __name__ == "__main__":
    unittest.main()
