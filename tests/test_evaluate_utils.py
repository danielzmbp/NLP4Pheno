import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_utils import cached_metric_script, load_metric  # noqa: E402


class EvaluateUtilsTests(unittest.TestCase):
    def test_finds_hashed_cached_metric_script(self):
        with tempfile.TemporaryDirectory() as directory:
            script = (
                Path(directory)
                / "evaluate_modules"
                / "metrics"
                / "evaluate-metric--seqeval"
                / "abc123"
                / "seqeval.py"
            )
            script.parent.mkdir(parents=True)
            script.touch()
            with patch.dict(os.environ, {"HF_MODULES_CACHE": directory}, clear=False):
                self.assertEqual(cached_metric_script("seqeval"), script)

    def test_offline_load_uses_cached_script_path(self):
        with tempfile.TemporaryDirectory() as directory:
            script = (
                Path(directory)
                / "evaluate_modules"
                / "metrics"
                / "evaluate-metric--accuracy"
                / "abc123"
                / "accuracy.py"
            )
            script.parent.mkdir(parents=True)
            script.touch()
            environment = {"HF_MODULES_CACHE": directory, "HF_HUB_OFFLINE": "1"}
            mocked = Mock(return_value="metric")
            fake_evaluate = Mock(load=mocked)
            with patch.dict(os.environ, environment, clear=False), patch.dict(
                sys.modules, {"evaluate": fake_evaluate}
            ):
                self.assertEqual(load_metric("accuracy"), "metric")
                mocked.assert_called_once_with(str(script))

    def test_offline_load_fails_fast_when_metric_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {"HF_MODULES_CACHE": directory, "HF_HUB_OFFLINE": "1"}
            fake_evaluate = Mock()
            with patch.dict(os.environ, environment, clear=False), patch.dict(
                sys.modules, {"evaluate": fake_evaluate}
            ):
                with self.assertRaisesRegex(RuntimeError, "not cached"):
                    load_metric("missing")


if __name__ == "__main__":
    unittest.main()
