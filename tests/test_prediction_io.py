import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from prediction_io import load_corpus_lines  # noqa: E402


class PredictionIOTests(unittest.TestCase):
    def test_keeps_final_line_without_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("first\nsecond", encoding="utf-8")
            self.assertEqual(load_corpus_lines(path), ["first", "second"])

    def test_does_not_create_a_spurious_line_for_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("first\nsecond\n", encoding="utf-8")
            self.assertEqual(load_corpus_lines(path), ["first", "second"])


if __name__ == "__main__":
    unittest.main()
