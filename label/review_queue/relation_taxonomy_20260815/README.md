# Relation-taxonomy audit (2026-08-15)

The combined 4,129-task gold corpus contains 716 relation labels whose typed
source/target/predicate combination is not configured as a classifier. Most
(654) do not involve a `STRAIN` endpoint and are outside the current network
scope. The remaining 62 labels span 13 strain-involving patterns in 37 tasks.

This pass deliberately separates clear annotation errors from plausible new
biology. `clear-corrections.json` fixes eight tasks with direct textual
evidence: a reversed edge, disease/phenotype confusion, taxonomic membership
encoded as inhabitation, the wrong relation predicate, and unsupported links.
It is applied reproducibly by `scripts/apply_annotation_corrections.py`.

The resulting gold file is
`label/project-10-reviewed-2026-08-15-taxonomy-pass.json`. It retains all 4,129
tasks and 17,433 entity spans. The correction pass changes five entity labels,
drops or replaces 18 relation records, adds 10 corrected edges, and leaves
7,128 relation records. Strain-involving unconfigured relations fall from 62
labels in 13 patterns to 50 labels in eight plausible patterns. The full
before/after integrity result is in `audit-after-corrections.md` and JSON.

Three corrections intentionally alter relation gold labels inside frozen
evaluation tasks: task 21216 affects the `STRAIN-ORGANISM:INHABITS` dev set,
task 22778 affects its test set, and tasks 20132/22778 affect the
`STRAIN-SPECIES:INHIBITS` test set. For these classifiers, compare models only
after evaluating both against the corrected files. Use
`scripts/compare_frozen_split_content.py` to detect all byte-level holdout
drift; unchanged classifiers should continue to match the original manifest
hashes exactly.

## Plausible taxonomy-expansion candidates

These classes should be retained and mined for more examples before deciding
whether to add a classifier:

| Typed relation | Current labels | Recommendation |
|---|---:|---|
| STRAIN-PHENOTYPE:ASSOCIATED_WITH | 13 | High-value candidate; adjudicate against `PRESENTS` |
| STRAIN-STRAIN:INHIBITS | 11 | High-value ecological interaction; prefer direct evidence or retain a mechanistic compound edge too |
| STRAIN-COMPOUND:PROMOTES | 9 | High-value candidate; define separately from `PRODUCES` |
| ORGANISM-STRAIN:RESISTS | 7 | Directionally unusual; adjudicate each example before expansion |
| STRAIN-PHENOTYPE:RESISTS | 4 | High-value phenotype relation; mine more positives |
| STRAIN-COMPOUND:INHIBITS | 4 | High-value candidate; distinguish inhibition of production/activity from degradation |
| STRAIN-STRAIN:RESISTS | 3 | Potential ecological interaction; mine more direct positives |
| STRAIN-ORGANISM:RESISTS | 2 | Directionally plausible but sparse; adjudicate with organism context |
| STRAIN-SPECIES:INFECTS | 2 | The audited examples were errors; do not expand from current support |
| STRAIN-DISEASE:PROMOTES | 1 | Plausible but too sparse; mine more evidence |
| COMPOUND-STRAIN:PROMOTES | 1 | Plausible and useful; mine more evidence |
| STRAIN-PHENOTYPE:PRODUCES | 1 | The audited example was an error; do not expand from current support |
| STRAIN-SPECIES:INHABITS | 4 | The audited examples were taxonomic-membership errors; do not expand from current support |

## Decision rule for the next queue

Prefer sentences with explicit trigger language and ontology-grounded entity
spans. Prioritize the first seven plausible classes above, but do not activate
a new binary classifier until it has at least 50 independently reviewed
positive tasks and a frozen holdout containing positives. Sparse retained
relations remain gold annotations; they are not silently treated as errors.
