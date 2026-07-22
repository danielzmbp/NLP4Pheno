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
