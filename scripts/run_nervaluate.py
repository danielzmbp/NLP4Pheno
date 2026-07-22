"""Evaluate the per-entity NER predictions with nervaluate.

The training datasets intentionally use untyped BIO tags (``B``, ``I``, and
``O``), because each model predicts exactly one entity type.  Nervaluate 1.x
requires typed tags, so this script adds the entity name before evaluation.
"""

from dataclasses import asdict, is_dataclass
import json
import os

import yaml


def load_predictions(label):
    file_path = f"NER_output/{label}/test_predictions.txt"
    with open(file_path) as handle:
        return handle.read()


def load_ground_truth(label):
    file_path = f"NER/{label}/test.txt"
    with open(file_path) as handle:
        return handle.read()


def load_labels_from_config(config_path):
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    return config.get("ner_labels", [])


def normalize_conll(text, label):
    """Return tab-separated CoNLL with bare B/I tags made entity-specific."""
    normalized = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            normalized.append("")
            continue

        columns = line.split("\t")
        if len(columns) < 2:
            raise ValueError(
                f"Invalid CoNLL line {line_number}: expected tab-separated token and tag"
            )

        tag = columns[-1]
        if tag in {"B", "I"}:
            columns[-1] = f"{tag}-{label}"
        elif tag != "O" and not tag.startswith(("B-", "I-")):
            raise ValueError(f"Invalid BIO tag {tag!r} on line {line_number}")
        normalized.append("\t".join(columns))

    return "\n".join(normalized) + "\n"


def jsonable(value):
    """Convert nervaluate dataclasses into stable, rounded JSON values."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_error_sentences(output_dir, predictions, indices):
    sentences = predictions.strip().split("\n\n")
    for strategy, categories in indices.items():
        category_values = asdict(categories) if is_dataclass(categories) else categories
        for category, pairs in category_values.items():
            document_ids = sorted({pair[0] for pair in pairs})
            selected = [sentences[index] for index in document_ids if index < len(sentences)]
            output_path = os.path.join(output_dir, f"{strategy}_{category}_sentences.txt")
            with open(output_path, "w") as handle:
                handle.write("\n\n".join(selected))


def main():
    from nervaluate import Evaluator

    labels = load_labels_from_config("./config.yaml")
    for label in labels:
        ground_truth = load_ground_truth(label)
        ground_truth = ground_truth.replace("-DOCSTART- -X- O\n", "").replace(
            " -X- _ ", "\t"
        )
        predictions = load_predictions(label)

        ground_truth = normalize_conll(ground_truth, label)
        predictions = normalize_conll(predictions, label)
        evaluation = Evaluator(
            ground_truth, predictions, tags=[label], loader="conll"
        ).evaluate()

        output_dir = f"./NER_output/{label}"
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "overall_results.json"), "w") as handle:
            json.dump(jsonable(evaluation["overall"]), handle, indent=4)

        with open(os.path.join(output_dir, "results_per_tag.json"), "w") as handle:
            json.dump(jsonable(evaluation["entities"]), handle, indent=4)

        with open(os.path.join(output_dir, "comparison_report.json"), "w") as handle:
            json.dump(jsonable(evaluation["overall_indices"]), handle, indent=4)

        write_error_sentences(
            output_dir, predictions, evaluation["overall_indices"]
        )


if __name__ == "__main__":
    main()
