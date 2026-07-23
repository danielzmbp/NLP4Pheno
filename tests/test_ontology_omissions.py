import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from audit_ontology_omissions import (  # noqa: E402
    AliasCandidate,
    find_omission_issues,
    context_is_credible,
    iter_ngram_spans,
    score_candidate,
)


class OntologyOmissionTests(unittest.TestCase):
    def test_ngram_spans_preserve_offsets(self):
        text = "Cells produced hydrogen peroxide."
        spans = list(iter_ngram_spans(text, max_tokens=2))
        self.assertIn((15, 32, "hydrogen peroxide"), spans)

    def test_annotation_and_ontology_support_scores_highly(self):
        candidate = AliasCandidate(
            "COMPOUND",
            "CHEBI",
            "CHEBI:16240",
            "hydrogen peroxide",
            "hydrogen peroxide",
            "LABEL",
            "normalized",
            False,
        )
        result = score_candidate(
            "hydrogen peroxide",
            candidate,
            {"hydrogen peroxide": Counter({"COMPOUND": 3})},
            has_strain=True,
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result[0], 0.95)

    def test_existing_spans_are_not_suggested_again(self):
        task = {
            "id": 1,
            "data": {"text": "The strain produced hydrogen peroxide."},
            "annotations": [
                {
                    "id": 2,
                    "result": [
                        {
                            "id": "s",
                            "type": "labels",
                            "value": {
                                "start": 4,
                                "end": 10,
                                "text": "strain",
                                "labels": ["STRAIN"],
                            },
                        },
                        {
                            "id": "c",
                            "type": "labels",
                            "value": {
                                "start": 20,
                                "end": 37,
                                "text": "hydrogen peroxide",
                                "labels": ["COMPOUND"],
                            },
                        }
                    ],
                }
            ],
        }
        candidate = AliasCandidate(
            "COMPOUND",
            "CHEBI",
            "CHEBI:16240",
            "hydrogen peroxide",
            "hydrogen peroxide",
            "LABEL",
            "normalized",
            False,
        )
        issues = find_omission_issues(
            [task],
            {"hydrogen peroxide": candidate},
            {},
            max_tokens=3,
        )
        self.assertEqual(issues, [])

    def test_uncovered_unique_alias_is_suggested(self):
        task = {
            "id": 1,
            "data": {"text": "The strain produced hydrogen peroxide."},
            "annotations": [{"id": 2, "result": []}],
        }
        candidate = AliasCandidate(
            "COMPOUND",
            "CHEBI",
            "CHEBI:16240",
            "hydrogen peroxide",
            "hydrogen peroxide",
            "LABEL",
            "normalized",
            False,
        )
        issues = find_omission_issues(
            [task],
            {"hydrogen peroxide": candidate},
            {},
            max_tokens=3,
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["action"]["text"], "hydrogen peroxide")

    def test_local_abbreviation_definition_blocks_wrong_ontology_expansion(self):
        candidate = AliasCandidate(
            "DISEASE",
            "MONDO",
            "MONDO:0009061",
            "cystic fibrosis",
            "CF",
            "EXACT",
            "normalized",
            True,
        )
        self.assertFalse(
            context_is_credible(
                "cell-free culture filtrate (CF)",
                "CF",
                candidate,
                {"cf": "cell-free culture filtrate"},
            )
        )

    def test_blood_requires_isolation_context(self):
        candidate = AliasCandidate(
            "ISOLATE",
            "UBERON",
            "UBERON:0000178",
            "blood",
            "blood",
            "LABEL",
            "normalized",
            False,
        )
        self.assertFalse(
            context_is_credible(
                "The strain grew on sheep blood agar.",
                "blood",
                candidate,
                {},
            )
        )
        self.assertTrue(
            context_is_credible(
                "The strain was isolated from blood.",
                "blood",
                candidate,
                {},
            )
        )

    def test_amino_acid_sequence_phrase_is_not_a_compound_mention(self):
        candidate = AliasCandidate(
            "COMPOUND",
            "CHEBI",
            "CHEBI:33709",
            "amino acid",
            "amino acid",
            "LABEL",
            "normalized",
            False,
        )
        self.assertFalse(
            context_is_credible(
                "The proteins shared 80% amino acid sequence identity.",
                "amino acid",
                candidate,
                {},
            )
        )


if __name__ == "__main__":
    unittest.main()
