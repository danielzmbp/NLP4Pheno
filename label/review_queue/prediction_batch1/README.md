# Full-PMC model prediction audit — batch 1

This batch mines the completed full-PMC NER/RE predictions for informative new
training examples. It is separate from the earlier annotation-correction
projects: these are new PMC texts that can be appended to the ground truth only
after every entity and relation in the selected text has been checked.

The reproducible mining run evaluated four strata for every predicted relation:

- `high_grounded`: strong NER/RE evidence and a unique ontology match;
- `high_unresolved`: strong NER/RE evidence but no unique ontology match;
- `relation_boundary`: strong entities with an RE score near the decision
  threshold;
- `entity_tension`: strong RE evidence but a lower-confidence entity or strain.

Only texts with at most 450 characters and unique PMC provenance were selected.
`queue.json` contains 151 balanced source candidates. Codex manually inspected
32 of the simplest candidates and recorded every decision and rationale in
`codex-decisions.json`:

- 9 accepted;
- 9 corrected;
- 9 hard negatives;
- 5 deferred for project-owner judgment.

The accepted, corrected, and hard-negative tasks are stored in
`codex-curated.json`. The human queue contains the five explicitly deferred
cases plus one low-complexity representative from each remaining
relation/error stratum, for 55 tasks total. The other 69 candidates remain in
the source queue for later rounds.

The local Label Studio project was **PMC model prediction audit — batch 1**
(project 4):

<http://127.0.0.1:8080/projects/4/data>

Predictions are suggestions. Before submitting a task, check all displayed
entities and relations, remove false predictions, add missed entities, and add
any other supported configured relations within the text.

## Completed review

All 55 human-review tasks were submitted on 2026-07-26. The untouched Label
Studio export is `reviewed-export-2026-07-26.json`. A complete second semantic
pass corrected 33 tasks without modifying that source export:

- 8 missed entities were added, 12 false entities removed, and 22 entity
  boundaries or labels corrected;
- 17 missed relations were added and 22 false or malformed relations removed;
- the corrected 55-task export has no invalid or mismatched spans, duplicate
  spans or relations, missing endpoints, unlabeled relations, or relation types
  outside `config.yaml`.

Every second-pass change and its rationale is recorded in
`second-pass-corrections.json`; `second-pass-report.json` is the generated
summary. Reproduce the corrected export with:

```bash
python scripts/apply_annotation_corrections.py \
  label/review_queue/prediction_batch1/reviewed-export-2026-07-26.json \
  label/review_queue/prediction_batch1/second-pass-corrections.json \
  --output label/review_queue/prediction_batch1/reviewed-export-2026-07-26-second-pass.json \
  --report label/review_queue/prediction_batch1/second-pass-report.json
```

The 27 Codex-curated tasks and 55 corrected human-reviewed tasks were appended
to the previous 3,979-task gold export. The new versioned training file is
`label/project-10-reviewed-2026-07-26.json` with 4,061 tasks:

```bash
python scripts/append_prediction_reviews.py \
  label/project-10-reviewed-2026-07-23.json \
  --reviewed label/review_queue/prediction_batch1/codex-curated.json \
  --reviewed label/review_queue/prediction_batch1/reviewed-export-2026-07-26-second-pass.json \
  --output label/project-10-reviewed-2026-07-26.json \
  --report label/review_queue/prediction_batch1/append-report.json
```

Regenerate the source batch on a machine containing the full prediction files:

```bash
python scripts/build_prediction_review_queue.py \
  results/preds2509/REL_output/preds_straininfo_grounded_pmc.pqt \
  results/preds2509/NER_output/preds.parquet \
  label/project-10-reviewed-2026-07-23.json \
  --queue-output label/review_queue/prediction_batch1/queue.json \
  --issues-output label/review_queue/prediction_batch1/issues.tsv \
  --summary-output label/review_queue/prediction_batch1/summary.json \
  --per-relation-tier 3 \
  --max-text-chars 450 \
  --seed 2509
```

Apply the tracked manual curation and select the compact human queue:

```bash
python scripts/apply_prediction_curation.py \
  label/review_queue/prediction_batch1/queue.json \
  label/review_queue/prediction_batch1/codex-decisions.json \
  --curated-output label/review_queue/prediction_batch1/codex-curated.json \
  --remaining-output label/review_queue/prediction_batch1/human-queue.json \
  --report label/review_queue/prediction_batch1/curation-report.json
```

Never overwrite the 2026-07-23 source export; it is the reproducible parent of
the new version.
