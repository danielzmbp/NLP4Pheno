# Weak-relation review queue — 2026-08-02

This is a targeted, fully pre-annotated active-learning queue built from the
strict full-PMC network run. It focuses on the five configured classifiers that
still have fewer than 50 positive training tasks:

- `STRAIN-DISEASE:INHIBITS`: 16 tasks;
- `STRAIN-ORGANISM:INHIBITS`: 15 tasks;
- `STRAIN-ORGANISM:SYMBIONT_OF`: 13 tasks;
- `STRAIN-PHENOTYPE:INHIBITS`: 12 tasks;
- `STRAIN-SPECIES:INHIBITS`: 12 tasks.

The 68 tasks are ordered by review tier: 16 high-confidence ontology-grounded,
16 high-confidence unresolved, 17 relation-boundary, and 19 entity-tension
examples. Predictions are suggestions, not accepted ground truth. Check every
entity and relation in the sentence, including omissions, before submitting.

The local Label Studio project is **PMC weak relation audit — 2026-08-02**
(project 5):

<http://127.0.0.1:8080/projects/5/data>

`queue.json` is the immutable source queue, `issues.tsv` is its audit table,
`summary.json` records the reproducible selection policy, and
`label-studio-project.json` records the local import. Completed human reviews
are exported to a dated file in this directory and are appended to the gold
corpus only after semantic and structural audit.

The queue was generated with `scripts/build_prediction_review_queue.py` using
four examples per relation/tier, seed 2509, and one repeated
`--include-relation` argument for each relation listed above.

## Completed review

All 68 tasks were submitted on 2026-08-15. The immutable Label Studio export is
`reviewed-export-2026-08-15.json`. A semantic second pass corrected 25 tasks
without modifying that source export: 2 entities were added, 6 removed, 26
boundaries or labels edited, 15 relations added, and 36 relations removed.

The corrected batch is `reviewed-export-2026-08-15-second-pass.json`; every
change and rationale is recorded in `second-pass-corrections-2026-08-15.json`
and `second-pass-report-2026-08-15.json`. It has no invalid or mismatched spans,
duplicate spans or relations, missing endpoints, unlabeled relations, or typed
relations outside `config.yaml`.

The corrected 68 tasks were appended to the prior 4,061-task export as
`label/project-10-reviewed-2026-08-15.json` (4,129 tasks total). The combined
audit is `label/annotation_audit_2026-08-15.md`.
