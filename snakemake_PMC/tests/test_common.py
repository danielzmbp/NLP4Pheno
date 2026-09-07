from __future__ import annotations

import sys
import unittest
from pathlib import Path


PMC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PMC_DIR / "scripts"))

from common import ParseOptions, parse_jats, split_sentences  # noqa: E402


class SentenceSplittingTests(unittest.TestCase):
    def test_genus_abbreviation_is_not_split(self) -> None:
        text = "E. coli cells were inhibited. Growth continued in the control."
        self.assertEqual(
            split_sentences(text),
            ["E. coli cells were inhibited.", "Growth continued in the control."],
        )


class JatsParsingTests(unittest.TestCase):
    def test_parser_preserves_provenance_and_filters_sections(self) -> None:
        xml_path = PMC_DIR / "tests/fixtures/xml/PMC10000001.2.xml"
        manifest = {
            "pmcid": "PMC10000001",
            "article_version": "PMC10000001.2",
            "version": 2,
            "inventory_last_modified": "2026-07-19T01:00:00.000Z",
            "source_etag": "etag-new",
            "xml_url": xml_path.resolve().as_uri(),
            "snapshot_date": "2026-07-20",
        }
        options = ParseOptions(
            max_text_chars=100,
            min_text_chars=10,
            excluded_section_patterns=("acknowledg",),
        )
        article, rows = parse_jats(xml_path.read_bytes(), manifest, options)

        self.assertTrue(article["included"])
        self.assertEqual(article["doi"], "10.1234/fixture.2")
        self.assertEqual(article["publication_year"], 2026)
        self.assertEqual(article["license_code"], "CC BY")
        self.assertEqual(article["corpus_rows"], len(rows))
        combined = " ".join(row["text"] for row in rows)
        self.assertIn("E. coli indicator cells", combined)
        self.assertNotIn("acknowledgement", combined.lower())
        self.assertNotIn("figure caption", combined.lower())
        self.assertTrue(all(row["article_version"] == "PMC10000001.2" for row in rows))

    def test_non_english_article_is_recorded_but_not_emitted(self) -> None:
        xml_path = PMC_DIR / "tests/fixtures/xml/PMC10000002.1.xml"
        manifest = {
            "pmcid": "PMC10000002",
            "article_version": "PMC10000002.1",
            "version": 1,
            "inventory_last_modified": "2026-07-18T01:00:00.000Z",
            "source_etag": "etag-french",
            "xml_url": xml_path.resolve().as_uri(),
            "snapshot_date": "2026-07-20",
        }
        options = ParseOptions(allowed_languages=("en",), include_unknown_language=False)
        article, rows = parse_jats(xml_path.read_bytes(), manifest, options)
        self.assertFalse(article["included"])
        self.assertEqual(article["exclusion_reason"], "language_fr")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
