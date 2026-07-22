import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_nervaluate.py"
SPEC = importlib.util.spec_from_file_location("run_nervaluate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_conll_types_bio_tags():
    source = "strain\tB\nname\tI\n.\tO\n\n"

    assert MODULE.normalize_conll(source, "STRAIN") == (
        "strain\tB-STRAIN\nname\tI-STRAIN\n.\tO\n\n"
    )


def test_normalize_conll_preserves_already_typed_tags():
    source = "E.\tB-SPECIES\ncoli\tI-SPECIES\n"

    assert MODULE.normalize_conll(source, "SPECIES") == source


def test_normalize_conll_rejects_space_separated_input():
    with pytest.raises(ValueError, match="tab-separated"):
        MODULE.normalize_conll("strain B\n", "STRAIN")
