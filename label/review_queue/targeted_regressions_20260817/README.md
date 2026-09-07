# Targeted regression review queue — 2026-08-17

This 55-task queue targets the entity and relation classes that declined in the
frozen-holdout comparison after the 4,129-task retrain:

- `STRAIN-COMPOUND:PRODUCES`: 12 tasks;
- `STRAIN-COMPOUND:RESISTS`: 12 tasks;
- `STRAIN-DISEASE:ASSOCIATED_WITH`: 12 tasks;
- `STRAIN-ORGANISM:SYMBIONT_OF`: 12 tasks;
- `STRAIN-PHENOTYPE:INHIBITS`: 7 tasks.

The source predictions come from the completed strict full-PMC run, while the
exclusion set is the corrected 4,129-task gold corpus. Therefore every sentence
is new to the annotation corpus. The queue is balanced across high-confidence
grounded, high-confidence unresolved, relation-boundary, and entity-tension
tiers where candidates were available. High-confidence tasks appear first.

`queue.json` contains prediction suggestions, not accepted ground truth. Review
all entity spans and all relations in each sentence before submitting. In
particular, distinguish `SYMBIONT_OF` from the broader `INHABITS`, explicit
disease association from sample-source descriptions, and compound resistance
from growth or biofilm context that merely mentions a compound.

`issues.tsv` records the sampled edge and confidence metadata, while
`summary.json` records the reproducible selection policy. Reviewed annotations
must be exported and audited before they are appended to the gold corpus.

## Completed review — 2026-08-27

All 55 tasks were submitted once in Label Studio project 7. The raw export is
`reviewed-export-2026-08-27.json`. A tracked second pass corrected 18 tasks:
six entity spans were removed, four were edited, eleven relations were added,
and sixteen were removed. The changes include negation and endpoint errors,
two blank relation labels, unsupported cross-sentence edges, entity-boundary
fixes, and exact cross-type duplicates.

Reproduce the corrected review export with:

```bash
python scripts/apply_annotation_corrections.py \
  label/review_queue/targeted_regressions_20260817/reviewed-export-2026-08-27.json \
  label/review_queue/targeted_regressions_20260817/second-pass-corrections-2026-08-27.json \
  --output label/review_queue/targeted_regressions_20260817/reviewed-export-2026-08-27-second-pass.json \
  --report label/review_queue/targeted_regressions_20260817/second-pass-report-2026-08-27.json
```

The corrected export has no invalid or mismatched spans, missing endpoints,
unlabelled relations, duplicate relations, or typed relations outside
`config.yaml`. Five intentional multi-label relations retain both
`SYMBIONT_OF` and its broader `INHABITS` label.

The 55 tasks were appended to the 4,129-task parent as
`label/project-10-reviewed-2026-08-27-targeted-pass.json`, yielding 4,184 gold
tasks. `append-report-2026-08-27.json` and
`integrated-audit-2026-08-27.{json,md}` record provenance and validation.
