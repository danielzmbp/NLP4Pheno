import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from split_utils import three_way_group_split  # noqa: E402


class GroupSplitTests(unittest.TestCase):
    def test_source_group_never_crosses_partitions(self):
        labels = []
        groups = []
        for index in range(20):
            labels.extend([index % 2, 0])
            groups.extend([f"group-{index}", f"group-{index}"])
        splits = three_way_group_split(
            labels, groups, test_and_dev_size=0.3, seed=5
        )
        group_sets = {
            split: {groups[index] for index in indices}
            for split, indices in splits.items()
        }
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["train"] & group_sets["dev"])
        self.assertFalse(group_sets["test"] & group_sets["dev"])
        self.assertEqual(sum(map(len, splits.values())), len(labels))


if __name__ == "__main__":
    unittest.main()
