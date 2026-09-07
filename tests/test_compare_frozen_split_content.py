import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_frozen_split_content import compare  # noqa: E402


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class CompareFrozenSplitContentTests(unittest.TestCase):
    def test_reports_matching_drifted_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / "NER" / "STRAIN"
            path.mkdir(parents=True)
            path.joinpath("dev.jsonls").write_bytes(b"same\n")
            path.joinpath("test.jsonls").write_bytes(b"new\n")
            manifest = {
                "name": "baseline",
                "ner": {
                    "STRAIN": {
                        "dev_sha256": digest(b"same\n"),
                        "test_sha256": digest(b"old\n"),
                    }
                },
                "rel": {
                    "STRAIN-PHENOTYPE:PRESENTS": {
                        "dev_sha256": digest(b"missing\n"),
                        "test_sha256": digest(b"missing\n"),
                    }
                },
            }

            report = compare(run, manifest)

        self.assertEqual(report["files"], 4)
        self.assertEqual(report["matching_files"], 1)
        self.assertEqual(report["drifted_files"], 1)
        self.assertEqual(report["missing_files"], 2)
        self.assertEqual(report["drifted"][0]["split"], "test")
