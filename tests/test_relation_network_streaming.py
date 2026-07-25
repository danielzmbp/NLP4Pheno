import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from create_relation_network import create_relation_network  # noqa: E402
from link_relation_evidence import (  # noqa: E402
    link_network_to_evidence,
    link_predictions_to_pmc,
)


class RelationNetworkStreamingTests(unittest.TestCase):
    def test_network_filters_bad_strains_and_preserves_direction(self):
        predictions = pl.DataFrame(
            {
                "word_strain_qc": ["strain x", "strain y", "covid strain"],
                "straininfo_si_id": [10, 20, 30],
                "word_qc_group": ["glucose", "host a", "ignored"],
                "rel": [
                    "STRAIN-COMPOUND:PRODUCES",
                    "ORGANISM-STRAIN:INHIBITS",
                    "STRAIN-PHENOTYPE:PRESENTS",
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_file = root / "predictions.parquet"
            network_file = root / "network.tsv"
            strains_file = root / "strains.txt"
            predictions.write_parquet(predictions_file)

            create_relation_network(predictions_file, network_file, strains_file)

            network = pl.read_csv(network_file, separator="\t").sort("source")
            strains = strains_file.read_text().splitlines()

        self.assertEqual(strains, ["SI-ID10", "SI-ID20"])
        self.assertEqual(
            network.select(
                "source", "target", "rel", "source_ner", "target_ner"
            ).rows(),
            [
                ("SI-ID10", "glucose", "PRODUCES", "STRAIN", "COMPOUND"),
                ("host a", "SI-ID20", "INHIBITS", "ORGANISM", "STRAIN"),
            ],
        )

    def test_pmc_and_network_evidence_are_linked(self):
        predictions = pl.DataFrame(
            {
                "text": ["sentence a"],
                "straininfo_si_id": [10],
                "word_qc_group": ["glucose"],
                "rel": ["STRAIN-COMPOUND:PRODUCES"],
            }
        )
        pmc = pl.DataFrame(
            {
                "text": ["sentence a"],
                "pmcid": ["PMC1"],
                "article_version": ["v1"],
                "paragraph": [2],
                "sentence_range": ["0:10"],
            }
        )
        network = pl.DataFrame(
            {
                "source": ["SI-ID10"],
                "target": ["glucose"],
                "rel": ["PRODUCES"],
                "source_ner": ["STRAIN"],
                "target_ner": ["COMPOUND"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_file = root / "predictions.parquet"
            pmc_file = root / "pmc.parquet"
            linked_file = root / "linked.parquet"
            network_file = root / "network.tsv"
            network_output = root / "network_pmc.tsv"
            predictions.write_parquet(predictions_file)
            pmc.write_parquet(pmc_file)
            network.write_csv(network_file, separator="\t")

            link_predictions_to_pmc(predictions_file, pmc_file, linked_file)
            link_network_to_evidence(network_file, linked_file, network_output)
            result = pl.read_csv(network_output, separator="\t")

        self.assertEqual(result.get_column("pmcid").to_list(), ["PMC1"])
        self.assertEqual(result.get_column("sentence_range").to_list(), ["0:10"])


if __name__ == "__main__":
    unittest.main()
