#!/usr/bin/env python3
"""Apply one relation model with bounded-memory Parquet I/O."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


Prediction = dict[str, Any]
Predictor = Callable[[list[str]], Sequence[Prediction]]


def other_entity_type(relation: str) -> str:
    """Return the non-STRAIN entity type encoded in a relation label."""
    pair = relation.split(":", 1)[0].split("-")
    non_strain = [entity for entity in pair if entity != "STRAIN"]
    if len(pair) != 2 or len(non_strain) != 1:
        raise ValueError(
            f"Relation must contain STRAIN and one other entity type: {relation}"
        )
    return non_strain[0]


def _prediction_label(value: Any) -> int:
    label = str(value)
    try:
        return int(label.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected relation prediction label: {label}") from exc


def _output_schema(input_schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [
            *input_schema,
            pa.field(
                "re_result",
                pa.struct(
                    [
                        pa.field("label", pa.string()),
                        pa.field("score", pa.float64()),
                    ]
                ),
            ),
            pa.field("rel_score", pa.float64()),
            pa.field("label", pa.int64()),
            pa.field("rel", pa.string()),
        ]
    )


def stream_relation_predictions(
    input_file: str | Path,
    output_file: str | Path,
    relation: str,
    predictor: Predictor,
    row_batch_size: int = 8192,
) -> tuple[int, int]:
    """Predict one relation type and incrementally write positive rows."""
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be positive.")

    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(f".{output_file.name}.tmp")
    temporary_output.unlink(missing_ok=True)

    parquet = pq.ParquetFile(input_file)
    input_schema = parquet.schema_arrow
    for required in ("ner", "formatted_text"):
        if required not in input_schema.names:
            raise ValueError(f"Input table is missing required column: {required}")

    entity_type = other_entity_type(relation)
    output_schema = _output_schema(input_schema)
    total_rows = parquet.metadata.num_rows
    rows_scanned = 0
    report_interval = max(row_batch_size * 16, 100_000)
    next_report = report_interval
    candidate_count = 0
    positive_count = 0

    writer = pq.ParquetWriter(
        temporary_output,
        output_schema,
        compression="snappy",
    )
    try:
        for record_batch in parquet.iter_batches(batch_size=row_batch_size):
            rows_scanned += record_batch.num_rows
            table = pa.Table.from_batches([record_batch])
            candidate_mask = pc.and_(
                pc.equal(table["ner"], entity_type),
                pc.and_(
                    pc.is_valid(table["formatted_text"]),
                    pc.not_equal(table["formatted_text"], ""),
                ),
            )
            candidates = table.filter(candidate_mask)
            if candidates.num_rows == 0:
                continue

            texts = candidates["formatted_text"].to_pylist()
            predictions = list(predictor(texts))
            if len(predictions) != len(texts):
                raise ValueError(
                    "Predictor returned a different number of results than inputs."
                )

            candidate_count += len(predictions)
            labels = [_prediction_label(result.get("label")) for result in predictions]
            positive_indices = [
                index for index, label in enumerate(labels) if label == 1
            ]
            if positive_indices:
                positive_predictions = [
                    predictions[index] for index in positive_indices
                ]
                positives = candidates.take(
                    pa.array(positive_indices, type=pa.int64())
                )
                scores = [
                    float(prediction["score"])
                    for prediction in positive_predictions
                ]
                result_struct = pa.array(
                    [
                        {
                            "label": str(prediction["label"]),
                            "score": float(prediction["score"]),
                        }
                        for prediction in positive_predictions
                    ],
                    type=output_schema.field("re_result").type,
                )
                output_table = pa.Table.from_arrays(
                    [
                        *positives.columns,
                        result_struct,
                        pa.array(scores, type=pa.float64()),
                        pa.array(
                            [labels[index] for index in positive_indices],
                            type=pa.int64(),
                        ),
                        pa.array(
                            [relation] * len(positive_indices),
                            type=pa.string(),
                        ),
                    ],
                    schema=output_schema,
                )
                writer.write_table(output_table)
                positive_count += len(positive_indices)

            if rows_scanned >= next_report or rows_scanned == total_rows:
                print(
                    f"{relation}: scanned {rows_scanned:,}/{total_rows:,} rows; "
                    f"evaluated {candidate_count:,} candidates; "
                    f"found {positive_count:,} positives.",
                    flush=True,
                )
                next_report = rows_scanned + report_interval
    except BaseException:
        writer.close()
        temporary_output.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_output, output_file)

    return candidate_count, positive_count


def build_predictor(
    relation: str,
    device: int,
    half_precision: bool,
    inference_batch_size: int,
) -> Predictor:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    model_path = f"REL_output/{relation}/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_kwargs: dict[str, Any] = {}
    if half_precision:
        if device < 0:
            raise ValueError(
                "Half precision requires a GPU device. "
                "Use --no-half-precision for CPU runs."
            )
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if half_precision:
        model = model.to(f"cuda:{device}")

    classifier_kwargs: dict[str, Any] = {
        "task": "text-classification",
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "truncation": True,
        "max_length": 512,
    }
    if half_precision:
        classifier_kwargs["torch_dtype"] = torch.float16
    classifier = pipeline(**classifier_kwargs)

    def predict(texts: list[str]) -> Sequence[Prediction]:
        with torch.inference_mode():
            return classifier(texts, batch_size=inference_batch_size)

    return predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply one trained relation model to entity pairs."
    )
    parser.add_argument("--model", required=True, help="Relation type identifier")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-batch-size", type=int, default=8192)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument(
        "--no-half-precision",
        dest="half_precision",
        action="store_false",
    )
    parser.set_defaults(half_precision=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = build_predictor(
        args.model,
        args.device,
        args.half_precision,
        args.inference_batch_size,
    )
    candidates, positives = stream_relation_predictions(
        args.input,
        args.output,
        args.model,
        predictor,
        row_batch_size=args.row_batch_size,
    )
    print(
        f"Processed {candidates:,} candidate pairs for {args.model}; "
        f"wrote {positives:,} positives."
    )


if __name__ == "__main__":
    main()
