"""Utilities for extending a training corpus without changing evaluation sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_frozen_split_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported frozen split manifest: {path}")
    return manifest


def frozen_split_indices(
    task_ids: Iterable[Any],
    source_groups: Iterable[Any],
    *,
    dev_task_ids: Iterable[Any],
    test_task_ids: Iterable[Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Keep frozen dev/test tasks fixed and add safe unseen groups to train."""
    ids = [str(value) for value in task_ids]
    groups = [str(value) for value in source_groups]
    if len(ids) != len(groups):
        raise ValueError("task_ids and source_groups must have equal length")

    dev_ids = {str(value) for value in dev_task_ids}
    test_ids = {str(value) for value in test_task_ids}
    overlap = dev_ids & test_ids
    if overlap:
        raise ValueError(f"Frozen dev/test task IDs overlap: {sorted(overlap)[:10]}")

    observed_ids = set(ids)
    missing_dev = dev_ids - observed_ids
    missing_test = test_ids - observed_ids
    if missing_dev or missing_test:
        raise ValueError(
            "Frozen evaluation tasks are absent from the current data: "
            f"dev={sorted(missing_dev)[:10]}, test={sorted(missing_test)[:10]}"
        )

    evaluation_group_split: dict[str, str] = {}
    for task_id, group in zip(ids, groups):
        split = "dev" if task_id in dev_ids else "test" if task_id in test_ids else None
        if split is None:
            continue
        previous = evaluation_group_split.get(group)
        if previous is not None and previous != split:
            raise ValueError(
                f"Frozen source group {group!r} crosses dev and test splits"
            )
        evaluation_group_split[group] = split

    indices = {"train": [], "dev": [], "test": []}
    excluded_indices: list[int] = []
    excluded_tasks: set[str] = set()
    for index, (task_id, group) in enumerate(zip(ids, groups)):
        if task_id in dev_ids:
            indices["dev"].append(index)
        elif task_id in test_ids:
            indices["test"].append(index)
        elif group in evaluation_group_split:
            excluded_indices.append(index)
            excluded_tasks.add(task_id)
        else:
            indices["train"].append(index)

    split_groups = {
        split: {groups[index] for index in split_indices}
        for split, split_indices in indices.items()
    }
    if (
        split_groups["train"] & split_groups["dev"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["dev"] & split_groups["test"]
    ):
        raise RuntimeError("Source group leakage remains after frozen split assignment")

    return indices, {
        "policy": "frozen_dev_test_new_nonoverlapping_groups_to_train",
        "frozen_dev_tasks": len(dev_ids),
        "frozen_test_tasks": len(test_ids),
        "excluded_rows": len(excluded_indices),
        "excluded_tasks": len(excluded_tasks),
        "excluded_task_ids": sorted(excluded_tasks),
    }
