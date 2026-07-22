"""Leakage-resistant stratified split helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def three_way_group_split(
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    test_and_dev_size: float,
    seed: int,
) -> dict[str, list[int]]:
    """Split row indices while keeping every group in exactly one partition."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")
    if not labels:
        raise ValueError("Cannot split an empty dataset")
    frame = pd.DataFrame(
        {
            "row_index": range(len(labels)),
            "label": [int(value) for value in labels],
            "group": [str(value) for value in groups],
        }
    )
    group_labels = frame.groupby("group", sort=True)["label"].max()
    if group_labels.nunique() != 2:
        raise ValueError("Grouped dataset must contain positive and negative groups")
    if group_labels.value_counts().min() < 4:
        raise ValueError("Grouped dataset has too few groups for a stratified 3-way split")

    train_groups, holdout_groups = train_test_split(
        group_labels.index.to_numpy(),
        test_size=test_and_dev_size,
        random_state=seed,
        stratify=group_labels.to_numpy(),
    )
    holdout_labels = group_labels.loc[holdout_groups]
    test_groups, dev_groups = train_test_split(
        holdout_groups,
        test_size=0.5,
        random_state=seed,
        stratify=holdout_labels.to_numpy(),
    )
    group_sets = {
        "train": set(train_groups),
        "test": set(test_groups),
        "dev": set(dev_groups),
    }
    return {
        split: frame.loc[frame["group"].isin(group_set), "row_index"].tolist()
        for split, group_set in group_sets.items()
    }
