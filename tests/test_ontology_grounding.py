import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_ontology_index import (  # noqa: E402
    iter_mediadive_terms,
    iter_ncbi_taxdump_terms,
    iter_obo_terms,
    iter_rdfxml_terms,
    looks_like_chemical_formula,
)
from ontology_grounding import (  # noqa: E402
    Candidate,
    extract_abbreviation_definitions,
    extract_mentions,
    ground_mentions,
    normalize_formula_surface,
    normalize_surface,
    relaxed_surface,
)


class OntologyParsingTests(unittest.TestCase):
    def test_obo_parser_keeps_requested_active_terms_and_synonyms(self):
        content = """format-version: 1.2
data-version: test

[Term]
id: OMP:0001
name: motility
synonym: "microbial motility" EXACT []
synonym: "movement phenotype" RELATED []

[Term]
id: PATO:0001
name: imported quality

[Term]
id: OMP:0002
name: obsolete phenotype
is_obsolete: true
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.obo"
            path.write_text(content)
            terms = list(iter_obo_terms(path, "OMP:"))

        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0]["id"], "OMP:0001")
        self.assertEqual(
            terms[0]["synonyms"],
            [("microbial motility", "EXACT"), ("movement phenotype", "RELATED")],
        )

    def test_rdfxml_parser_keeps_requested_active_classes(self):
        content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#"
 xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
 <owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_00001234">
  <rdfs:label>kimchi</rdfs:label>
  <oboInOwl:hasExactSynonym>kimchee</oboInOwl:hasExactSynonym>
 </owl:Class>
 <owl:Class rdf:about="http://purl.obolibrary.org/obo/ENVO_00000001">
  <rdfs:label>not food</rdfs:label>
 </owl:Class>
</rdf:RDF>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.owl"
            path.write_text(content)
            terms = list(iter_rdfxml_terms(path, "FOODON:"))
        self.assertEqual(
            terms,
            [
                {
                    "id": "FOODON:00001234",
                    "name": "kimchi",
                    "synonyms": [("kimchee", "EXACT")],
                }
            ],
        )

    def test_mediadive_parser_adds_resource_derived_aliases(self):
        payload = {
            "data": [
                {"id": 11, "name": "MRS MEDIUM"},
                {"id": 381, "name": "LB (Luria-Bertani) MEDIUM"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.json"
            path.write_text(json.dumps(payload))
            terms = list(iter_mediadive_terms(path))
        self.assertIn(("MRS", "EXACT"), terms[0]["synonyms"])
        self.assertIn(("LB", "EXACT"), terms[1]["synonyms"])
        self.assertIn(("Luria-Bertani", "EXACT"), terms[1]["synonyms"])

    def test_ncbi_taxdump_parser_keeps_names_but_not_authorities(self):
        content = """9606\t|\tHomo sapiens\t|\t\t|\tscientific name\t|
9606\t|\thuman\t|\t\t|\tgenbank common name\t|
9606\t|\tHomo sapiens Linnaeus, 1758\t|\t\t|\tauthority\t|
10090\t|\tMus musculus\t|\t\t|\tscientific name\t|
10090\t|\thouse mouse\t|\t\t|\tcommon name\t|
""".encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taxdump.tar.gz"
            with tarfile.open(path, mode="w:gz") as archive:
                info = tarfile.TarInfo("names.dmp")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            terms = list(iter_ncbi_taxdump_terms(path))

        self.assertEqual(terms[0]["id"], "NCBITaxon:9606")
        self.assertEqual(terms[0]["name"], "Homo sapiens")
        self.assertEqual(terms[0]["synonyms"], [("human", "EXACT")])
        self.assertEqual(terms[1]["synonyms"], [("house mouse", "EXACT")])

    def test_chemical_formula_policy_rejects_bare_acronyms(self):
        for formula in ("NaCl", "H2O2", "MgSO4", "Fe(III)", "iso-C15:0"):
            self.assertTrue(looks_like_chemical_formula(formula), formula)
        for acronym in ("HM", "LTA", "Cm", "GABA", "C12", "N3"):
            self.assertFalse(looks_like_chemical_formula(acronym), acronym)


class GroundingTests(unittest.TestCase):
    def test_surface_normalization_is_conservative(self):
        self.assertEqual(normalize_surface("  Gram–Positive. "), "gram-positive")
        self.assertEqual(relaxed_surface("β-lactam"), "beta lactam")
        self.assertEqual(relaxed_surface("H _2 O _2"), "h2 o2")
        self.assertEqual(normalize_formula_surface("H _2 O _2"), "H2O2")
        self.assertNotEqual(
            normalize_formula_surface("Co^2+"),
            normalize_formula_surface("CO(2)"),
        )

    def test_extracts_parenthetical_abbreviation(self):
        definitions = extract_abbreviation_definitions(
            "Cells were grown in brain heart infusion (BHI) broth."
        )
        self.assertEqual(definitions["bhi"], "brain heart infusion")

    def test_extract_mentions_uses_local_abbreviation_definition(self):
        task = {
            "id": 1,
            "data": {"text": "cystic fibrosis (CF) disease"},
            "annotations": [
                {
                    "id": 2,
                    "result": [
                        {
                            "id": "x",
                            "type": "labels",
                            "value": {
                                "start": 17,
                                "end": 19,
                                "text": "CF",
                                "labels": ["DISEASE"],
                            },
                        }
                    ],
                }
            ],
        }
        mentions = extract_mentions([task])
        self.assertEqual(mentions[0]["expanded_form"], "cystic fibrosis")

    def test_unique_and_ambiguous_candidates_are_kept_separate(self):
        candidate = Candidate("OMP", "OMP:1", "motility", "motility", "LABEL")
        collision = Candidate("OMP", "OMP:2", "other motility", "movement", "EXACT")
        mentions = [
            {
                "entity_type": "PHENOTYPE",
                "normalized_surface": "motility",
                "relaxed_surface": "motility",
                "expanded_form": None,
            },
            {
                "entity_type": "PHENOTYPE",
                "normalized_surface": "movement",
                "relaxed_surface": "movement",
                "expanded_form": None,
            },
        ]
        exact = {
            ("PHENOTYPE", "motility"): [candidate],
            ("PHENOTYPE", "movement"): [candidate, collision],
        }
        rows = ground_mentions(mentions, exact, {})
        self.assertEqual(rows[0]["status"], "matched")
        self.assertEqual(rows[0]["concept_id"], "OMP:1")
        self.assertEqual(rows[1]["status"], "ambiguous")
        self.assertEqual(json.loads(rows[1]["candidates_json"])[0]["ontology"], "OMP")

    def test_unsupported_entity_types_are_explicit(self):
        mentions = [
            {
                "entity_type": "STRAIN",
                "normalized_surface": "abc",
                "relaxed_surface": "abc",
                "expanded_form": None,
            }
        ]
        rows = ground_mentions(
            mentions,
            {},
            {},
            supported_entity_types={"PHENOTYPE"},
        )
        self.assertEqual(rows[0]["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
