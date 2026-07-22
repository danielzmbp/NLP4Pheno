import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from assembly_manifest import load_assembly_manifest  # noqa: E402


class AssemblyManifestTests(unittest.TestCase):
    def test_loads_and_deduplicates_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assemblies.txt"
            path.write_text("SI-1/GCF_1.1\nSI-1/GCF_1.1\nSI-2/GCA_2.1\n")
            self.assertEqual(
                load_assembly_manifest(path),
                [("SI-1", "GCF_1.1"), ("SI-2", "GCA_2.1")],
            )

    def test_rejects_empty_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assemblies.txt"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_assembly_manifest(path)

    def test_rejects_malformed_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assemblies.txt"
            path.write_text("GCF_1.1\n")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_assembly_manifest(path)


if __name__ == "__main__":
    unittest.main()
