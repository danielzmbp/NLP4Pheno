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
