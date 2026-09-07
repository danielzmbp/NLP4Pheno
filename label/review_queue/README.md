# High-confidence annotation review queue

`queue.json` contains proposed corrections as Label Studio predictions. They are
review suggestions, not accepted ground truth. The queue is sorted by category
and confidence: dominant-label conflicts first, exact entity omissions second,
and strongly repeated relation omissions third.

The original task ID and supporting evidence are retained in each task's data
and metadata. Export completed human reviews to a new file; do not overwrite the
2025 source export.

Regenerate the queue from the repository root with:

```bash
python scripts/build_annotation_review_queue.py \
  label/project-10-at-2025-08-21-21-08-cb43bf25.json \
  --config config.yaml \
  --queue-output label/review_queue/queue.json \
  --issues-output label/review_queue/issues.tsv \
  --summary-output label/review_queue/summary.json
```

The local Label Studio database lives in `label/label_studio_data/` and is
intentionally ignored by Git. The source queue, configuration, audit table and
summary are reproducible project artifacts.

## Local review server

The isolated Label Studio environment is installed at `.envs/label-studio/`.
Start the existing review project from the repository root with:

```bash
label/review_queue/start_server.sh
```

Then open <http://127.0.0.1:8080>. Local login details are stored in the ignored
file `label/review_queue/credentials.txt`.

The initialized project uses sequential sampling and contains the queue's 86
ranked tasks. Predictions are visible to reviewers, but remain suggestions
until a reviewer submits an annotation. Keep exports of reviewed annotations
separate from the 2025 source export so changes can be audited before merging.

The submitted first batch was exported and merged into the versioned file
`label/project-10-reviewed-2026-07-21.json`. The follow-up queue and its audit
artifacts live under `label/review_queue/batch2/`. All 21 follow-up tasks were
subsequently reviewed and merged into
`label/project-10-reviewed-2026-07-22.json`.

The ontology-backed third batch under `ontology_batch/` contained 128 tasks
with 138 proposed entity omissions corroborated both by the ontology index and
by at least two existing annotations of the same surface. All tasks were
reviewed on 2026-07-23: 134 omissions were accepted, four were rejected, and 15
relations were added by the reviewer. The result is merged into
`label/project-10-reviewed-2026-07-23.json`.

The full-PMC model-assisted audit under `prediction_batch1/` is the first
active-learning batch generated after full-corpus inference. Codex curated 27
of its short cases with tracked rationales; project 4 contains a balanced
55-task human queue drawn from the remaining relation and error strata.

The follow-up `weak_relations_20260802/` queue targets the five relation
classifiers that remain below 50 positive source tasks after the 4,061-task
gold update. Its 68 pre-annotated tasks are available in Label Studio project
5, with high-confidence candidates presented before boundary/tension cases.
