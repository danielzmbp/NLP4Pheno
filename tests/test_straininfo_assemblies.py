import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from fetch_straininfo_assemblies import parse_strain_response  # noqa: E402


class StrainInfoAssemblyTests(unittest.TestCase):
    def test_prefers_complete_then_newest_genome(self):
        payload = [
            {
                "strain": {
                    "siID": 2805,
                    "sequence": [
                        {
                            "accessionNumber": "GCA_1",
                            "type": "genome",
                            "assemblyLevel": "scaffold",
                            "year": 2025,
                        },
                        {
                            "accessionNumber": "GCA_2",
                            "type": "genome",
                            "assemblyLevel": "complete",
                            "year": 2020,
                        },
                        {
                            "accessionNumber": "GCA_3",
                            "type": "genome",
                            "assemblyLevel": "complete",
                            "year": 2023,
                        },
                        {"accessionNumber": "AB123", "type": "gene"},
                    ],
                }
            }
        ]
        rows = parse_strain_response(2805, payload, "abc")
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["accession"] for row in rows if row["selected"]], ["GCA_3"]
        )

    def test_rejects_wrong_si_id(self):
        with self.assertRaisesRegex(ValueError, "wrong SI-ID"):
            parse_strain_response(1, [{"strain": {"siID": 2}}], "abc")


if __name__ == "__main__":
    unittest.main()
