"""Shared helpers for consuming Label Studio annotation exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _annotation_order(annotation: dict[str, Any]) -> tuple[str, str, int]:
    """Sort annotations deterministically, preferring the most recent record."""
    return (
        str(annotation.get("updated_at") or ""),
        str(annotation.get("created_at") or ""),
        int(annotation.get("id") or 0),
    )


def select_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    """Choose one authoritative annotation for a Label Studio task.

    A marked ground-truth annotation wins. Otherwise, use the latest active
    annotation. This prevents duplicate task annotations from being counted as
    separate training examples while retaining legitimate empty annotations.
    """
    active = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]
    if not active:
        return None
    ground_truth = [annotation for annotation in active if annotation.get("ground_truth")]
    candidates = ground_truth or active
    return max(candidates, key=_annotation_order)


def canonicalize_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tasks with at most one authoritative annotation per task."""
    canonical = []
    for task in tasks:
        annotation = select_annotation(task)
        normalized = dict(task)
        normalized["annotations"] = [annotation] if annotation is not None else []
        canonical.append(normalized)
    return canonical


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load and canonicalize a Label Studio JSON export."""
    with Path(path).open() as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a list of Label Studio tasks in {path}")
    return canonicalize_tasks(tasks)


def load_unique_pmc_groups(path: str | Path | None) -> dict[str, str]:
    """Load task-to-PMC groups that the provenance audit marked unambiguous."""
    if path is None:
        return {}
    import pyarrow.parquet as pq

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Configured annotation PMC map does not exist: {source}")
    table = pq.read_table(source, columns=["task_id", "pmcid", "status"])
    groups: dict[str, set[str]] = {}
    for row in table.to_pylist():
        if row["status"] != "unique_pmcid":
            continue
        groups.setdefault(str(row["task_id"]), set()).add(str(row["pmcid"]))
    invalid = {task_id: values for task_id, values in groups.items() if len(values) != 1}
    if invalid:
        raise ValueError(f"Tasks marked unique have multiple PMCIDs: {invalid}")
    return {task_id: next(iter(values)) for task_id, values in groups.items()}


def source_group(task_id: Any, pmc_groups: dict[str, str]) -> str:
    """Group a linked task by article and keep each unlinked task independent."""
    task_key = str(task_id)
    pmcid = pmc_groups.get(task_key)
    return f"pmcid:{pmcid}" if pmcid else f"task:{task_key}"
