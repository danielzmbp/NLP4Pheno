"""Convert Label Studio span annotations directly to NER training files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def task_to_example(task: dict[str, Any], label: str) -> dict[str, Any]:
    text = task.get("data", {}).get("text", "")
    spans = []
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") != "labels":
                continue
            value = result.get("value", {})
            if label not in value.get("labels", []):
                continue
            spans.append((int(value["start"]), int(value["end"]), result.get("id")))
    spans.sort()

    tokens = []
    tags = []
    previous_span = None
    for token, start, end in tokenize_with_offsets(text):
        overlaps = [span for span in spans if start < span[1] and end > span[0]]
        if len(overlaps) > 1:
            raise ValueError(f"Token overlaps multiple {label} spans in task {task.get('id')}")
        if overlaps:
            span = overlaps[0]
            prefix = "I" if previous_span == span else "B"
            # Each entity type has its own binary model, so generic B/I tags
            # preserve the prediction schema expected by ner_pred.smk.
            tag = prefix
            previous_span = span
        else:
            tag = "O"
            previous_span = None
        tokens.append(token)
        tags.append(tag)
    return {"id": str(task.get("id")), "tokens": tokens, "ner_tags": tags}


def build_dataset(
    source: str | Path,
    *,
    label: str,
    json_output: str | Path,
    bio_output: str | Path,
) -> dict[str, int]:
    with Path(source).open() as handle:
        tasks = json.load(handle)
    examples = [task_to_example(task, label) for task in tasks]

    json_path = Path(json_output)
    bio_path = Path(bio_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    bio_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    with bio_path.open("w") as handle:
        for example in examples:
            for token, tag in zip(example["tokens"], example["ner_tags"]):
                handle.write(f"{token}\t{tag}\n")
            handle.write("\n")
    return {
        "tasks": len(examples),
        "tokens": sum(len(example["tokens"]) for example in examples),
        "entity_tokens": sum(
            tag != "O" for example in examples for tag in example["ner_tags"]
        ),
    }
