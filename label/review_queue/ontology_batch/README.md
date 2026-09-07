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

## Review outcome

All 128 tasks were submitted in Label Studio on 2026-07-23. The reviewer
accepted 134 of the 138 proposed entity omissions:

- 100 `COMPOUND`
- 24 `ISOLATE`
- 9 `PHENOTYPE`
- 1 `MEDIUM`

The four rejected suggestions were three uses of `fluorescence` as a generic
measurement rather than a phenotype and one use of `dairy` that did not denote
an isolate source. The reviewer also added 15 relations. No source entities or
relations were removed. The merged corpus is
`label/project-10-reviewed-2026-07-23.json`; `review-summary.json` records the
decision counts and rejected cases.

Files:

- `queue.json`: Label Studio tasks with proposed spans as predictions
- `issues.tsv`: one row per selected suggestion with concept ID and evidence
- `summary.json`: thresholds and audit counts
- `reviewed-export-2026-07-23.json`: completed Label Studio export
- `review-summary.json`: accepted/rejected decision audit
- `merge-report.json`: merge counts and output hash

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

The completed review was merged without overwriting its source:

```bash
python scripts/merge_annotation_reviews.py \
  label/project-10-reviewed-2026-07-22.json \
  label/review_queue/ontology_batch/reviewed-export-2026-07-23.json \
  --output label/project-10-reviewed-2026-07-23.json \
  --report label/review_queue/ontology_batch/merge-report.json
```
