import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from frozen_splits import frozen_split_indices  # noqa: E402


class FrozenSplitTests(unittest.TestCase):
    def test_keeps_eval_and_excludes_new_task_from_eval_article(self):
        indices, stats = frozen_split_indices(
            ["old_train", "old_dev", "old_test", "new", "new_dev_article"],
            ["pmc:A", "pmc:B", "pmc:C", "pmc:D", "pmc:B"],
            dev_task_ids=["old_dev"],
            test_task_ids=["old_test"],
        )
        self.assertEqual(indices["train"], [0, 3])
        self.assertEqual(indices["dev"], [1])
        self.assertEqual(indices["test"], [2])
        self.assertEqual(stats["excluded_task_ids"], ["new_dev_article"])

    def test_rejects_eval_group_crossing_dev_and_test(self):
        with self.assertRaisesRegex(ValueError, "crosses dev and test"):
            frozen_split_indices(
                ["dev", "test"],
                ["pmc:A", "pmc:A"],
                dev_task_ids=["dev"],
                test_task_ids=["test"],
            )

    def test_rejects_missing_frozen_task(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            frozen_split_indices(
                ["train"],
                ["pmc:A"],
                dev_task_ids=["missing"],
                test_task_ids=[],
            )


if __name__ == "__main__":
    unittest.main()
