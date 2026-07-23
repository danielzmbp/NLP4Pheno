# Ontology-backed omission review

This queue audits `label/project-10-reviewed-2026-07-22.json` for entity spans
that may have been missed. It contains 128 tasks with 138 suggestions:

- 100 `COMPOUND`
- 25 `ISOLATE`
- 12 `PHENOTYPE`
- 1 `MEDIUM`

These are review suggestions, not edits to the ground truth. A candidate enters
this high-confidence batch only when:

1. it does not overlap an existing annotated span;
2. the surface resolves to one eligible ontology concept;
3. the same surface has already been annotated with the suggested label in at
   least two other occurrences; and
4. basic context checks do not detect a conflicting local acronym expansion or
   a known semantic trap.

An existing `STRAIN` annotation adds a small ranking bonus, but is not required.

For example, the audit rejects `CF` when the sentence defines it as
`cell-free culture filtrate`, rejects `blood` in `blood agar`, and rejects
`circular` when it describes a chromosome rather than colony morphology.

The broader audit found 948 possible omissions. The 810 lower-confidence
ontology-only or singly corroborated matches are reported only as counts in
`summary.json`; they are not placed in the review queue.

Files:

- `queue.json`: Label Studio tasks with proposed spans as predictions
- `issues.tsv`: one row per selected suggestion with concept ID and evidence
- `summary.json`: thresholds and audit counts

Regenerate the queue from the repository root:

```bash
python scripts/audit_ontology_omissions.py \
  label/project-10-reviewed-2026-07-22.json \
  --queue-output label/review_queue/ontology_batch/queue.json \
  --issues-output label/review_queue/ontology_batch/issues.tsv \
  --summary-output label/review_queue/ontology_batch/summary.json \
  --max-tasks 200 \
  --min-score 0.93
```

Use `label/review_queue/label_config.xml` when creating a Label Studio project
and import `queue.json`. The predictions contain all existing annotations plus
the proposed spans. Accept, edit, or remove each suggestion before submitting.

After exporting the submitted reviews, merge them into a new version rather
than overwriting the current source:

```bash
python scripts/merge_annotation_reviews.py \
  label/project-10-reviewed-2026-07-22.json \
  /path/to/label-studio-export.json \
  --output label/project-10-reviewed-YYYY-MM-DD.json \
  --report label/review_queue/ontology_batch/merge-report.json
```
