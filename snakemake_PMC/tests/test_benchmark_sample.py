from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PMC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PMC_DIR / "scripts"))

from build_benchmark_sample import verified_sample  # noqa: E402


class BenchmarkSamplingTests(unittest.TestCase):
    def test_metadata_requests_stop_after_one_bounded_batch(self) -> None:
        candidates = [(value, 1, f"metadata/PMC{value}.1.json") for value in range(100)]
        with patch(
            "build_benchmark_sample.fetch_metadata", return_value=({}, None)
        ) as fetch:
            accepted, rejected = verified_sample(
                candidates,
                count=5,
                bucket_url="https://example.invalid",
                workers=2,
                publication_year_min=None,
            )
        self.assertEqual(len(accepted), 5)
        self.assertEqual(rejected, {})
        self.assertLessEqual(fetch.call_count, 8)


if __name__ == "__main__":
    unittest.main()
