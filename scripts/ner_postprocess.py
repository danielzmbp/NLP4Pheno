"""Post-processing helpers shared by NER prediction workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def merge_entities(entity_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge complete contiguous B-I token sequences into entity spans."""
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(entity_list):
        current = entity_list[index].copy()
        current.pop("word", None)
        next_index = index + 1
        if current.get("entity_group") == "B":
            scores = [current["score"]]
            while (
                next_index < len(entity_list)
                and entity_list[next_index].get("entity_group") == "I"
                and entity_list[next_index]["start"] - current["end"] <= 4
            ):
                scores.append(entity_list[next_index]["score"])
                current["end"] = entity_list[next_index]["end"]
                next_index += 1
            current["score"] = float(np.mean(scores))
        merged.append(current)
        index = next_index
    return merged
