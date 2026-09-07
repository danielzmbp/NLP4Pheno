"""Build relation-classification examples from canonical Label Studio tasks."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import pandas as pd

from split_utils import three_way_group_split


def mark_entity_pair(
    text: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str | None:
    """Replace two non-overlapping entity spans with their typed markers."""
    entities = sorted((left, right), key=lambda entity: (entity["start"], entity["end"]))
    first, second = entities
    if first["end"] > second["start"]:
        return None
    marked = "".join(
        (
            text[: first["start"]],
            f"@{first['label']}$",
            text[first["end"] : second["start"]],
            f"@{second['label']}$",
            text[second["end"] :],
        )
    )
    return marked


def build_relation_rows(tasks: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create one row per ordered entity pair with all of its relation labels."""
    rows: list[dict[str, Any]] = []
    skipped_overlapping_pairs = 0
    skipped_overlapping_positive_pairs = 0

    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            continue
        results = annotations[0].get("result", [])
        entities: dict[str, dict[str, Any]] = {}
        relations: dict[tuple[str, str], list[str]] = {}

        for result in results:
            if result.get("type") != "labels":
                continue
            value = result.get("value", {})
            labels = value.get("labels", [])
            if len(labels) != 1:
                continue
            entities[result["id"]] = {
                "id": result["id"],
                "start": int(value["start"]),
                "end": int(value["end"]),
                "label": labels[0],
                "text": value.get("text", ""),
            }

        for result in results:
            if result.get("type") != "relation":
                continue
            key = (result.get("from_id"), result.get("to_id"))
            relations.setdefault(key, []).extend(result.get("labels", []))

        text = task.get("data", {}).get("text", "")
        for from_id, to_id in permutations(entities, 2):
            source = entities[from_id]
            target = entities[to_id]
            relation_labels = sorted(set(relations.get((from_id, to_id), [])))
            marked = mark_entity_pair(text, source, target)
            if marked is None:
                skipped_overlapping_pairs += 1
                if relation_labels:
                    skipped_overlapping_positive_pairs += 1
                continue
            pair_type = f"{source['label']}-{target['label']}"
            rows.append(
                {
                    "task_id": task.get("id"),
                    "from_id": from_id,
                    "to_id": to_id,
                    "pair_type": pair_type,
                    # Keep legitimate multi-label relations on one pair. This
                    # prevents a label from becoming an identical negative in
                    # another binary classifier.
                    "relations": "|".join(relation_labels),
                    "sentence": marked,
                }
            )

    frame = pd.DataFrame.from_records(
        rows,
        columns=[
            "task_id",
            "from_id",
            "to_id",
            "pair_type",
            "relations",
            "sentence",
        ],
    )
    stats = {
        "rows": len(frame),
        "skipped_overlapping_pairs": skipped_overlapping_pairs,
        "skipped_overlapping_positive_pairs": skipped_overlapping_positive_pairs,
    }
    return frame, stats


def relation_membership(values: pd.Series, relation_label: str) -> pd.Series:
    """Return whether each pipe-delimited relation set contains a label."""
    return values.fillna("").map(
        lambda value: relation_label in {
            item for item in str(value).split("|") if item
        }
    )


def split_by_task(
    frame: pd.DataFrame,
    *,
    test_and_dev_size: float,
    seed: int,
    group_column: str = "task_id",
) -> dict[str, pd.DataFrame]:
    """Split relation rows without allowing a source group to cross partitions."""
    if frame.empty:
        raise ValueError("Cannot split an empty relation dataset")
    indices = three_way_group_split(
        frame["binary_label"].astype(int).tolist(),
        frame[group_column].astype(str).tolist(),
        test_and_dev_size=test_and_dev_size,
        seed=seed,
    )
    return {
        split: frame.iloc[row_indices].copy()
        for split, row_indices in indices.items()
    }
