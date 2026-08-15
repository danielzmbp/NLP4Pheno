# Ground-truth annotation audit

Latest export: `label/review_queue/weak_relations_20260802/reviewed-export-2026-08-15.json` (`03b1d37f9697…`)

## Summary

- Tasks: 68; raw active annotation records: 68; canonical annotations: 68
- Canonical annotators: {'1': 68}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 442; relation labels: 182
- Exact duplicate texts: 0; tasks with multiple annotations: 0
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'review_tier', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 54 |
| DISEASE | 30 |
| ISOLATE | 5 |
| MEDIUM | 3 |
| ORGANISM | 92 |
| PHENOTYPE | 72 |
| SPECIES | 42 |
| STRAIN | 144 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| 1 | 68 | 0 | 6.50 | 2.68 |

### Entity labels per 100 tasks

| Label | Annotator 1 |
|---|---:|
| COMPOUND | 79.4 |
| DISEASE | 44.1 |
| ISOLATE | 7.4 |
| MEDIUM | 4.4 |
| ORGANISM | 135.3 |
| PHENOTYPE | 105.9 |
| SPECIES | 61.8 |
| STRAIN | 211.8 |

### Relation labels per 100 tasks

| Label | Annotator 1 |
|---|---:|
| INHABITS | 25.0 |
| INHIBITS | 120.6 |
| PRESENTS | 44.1 |
| PRODUCES | 19.1 |
| PROMOTES | 7.4 |
| SYMBIONT_OF | 51.5 |

## Typed relation counts

| Relation | Count |
|---|---:|
| COMPOUND-STRAIN:INHIBITS | 14 |
| STRAIN-COMPOUND:PRODUCES | 13 |
| STRAIN-DISEASE:INHIBITS | 31 |
| STRAIN-ISOLATE:INHABITS | 4 |
| STRAIN-ORGANISM:INHABITS | 13 |
| STRAIN-ORGANISM:INHIBITS | 13 |
| STRAIN-ORGANISM:SYMBIONT_OF | 35 |
| STRAIN-PHENOTYPE:INHIBITS | 11 |
| STRAIN-PHENOTYPE:PRESENTS | 30 |
| STRAIN-PHENOTYPE:PROMOTES | 4 |
| STRAIN-SPECIES:INHIBITS | 13 |
| STRAIN-SPECIES:PROMOTES | 1 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 14 | 2 |
| STRAIN-COMPOUND:DEGRADES | 0 | 0 |
| STRAIN-COMPOUND:PRODUCES | 13 | 6 |
| STRAIN-COMPOUND:RESISTS | 0 | 0 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 0 | 0 |
| STRAIN-DISEASE:INHIBITS | 31 | 12 |
| STRAIN-ISOLATE:INHABITS | 4 | 3 |
| STRAIN-MEDIUM:GROWS_ON | 0 | 0 |
| STRAIN-ORGANISM:INFECTS | 0 | 0 |
| STRAIN-ORGANISM:INHABITS | 13 | 8 |
| STRAIN-ORGANISM:INHIBITS | 13 | 7 |
| STRAIN-ORGANISM:SYMBIONT_OF | 35 | 9 |
| STRAIN-PHENOTYPE:INHIBITS | 11 | 8 |
| STRAIN-PHENOTYPE:PRESENTS | 30 | 17 |
| STRAIN-PHENOTYPE:PROMOTES | 4 | 4 |
| STRAIN-SPECIES:INHIBITS | 13 | 8 |

Configured classifiers with fewer than 50 positive tasks: ['COMPOUND-STRAIN:INHIBITS', 'STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:PRODUCES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-ISOLATE:INHABITS', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS', 'STRAIN-ORGANISM:INHABITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-ORGANISM:SYMBIONT_OF', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-PHENOTYPE:PRESENTS', 'STRAIN-PHENOTYPE:PROMOTES', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: ['STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS']
- Observed typed relations absent from config: 1 patterns / 1 labels (0.5%); complete list is retained in the JSON audit
- Of those, STRAIN-involving omissions: 1 patterns / 1 labels

## Ground-truth actions before retraining

- Keep PMCID, article version, paragraph and sentence identifiers in every new annotation task.
- Independently double-annotate a stratified subset and adjudicate it before reporting inter-annotator agreement.
- Collect or merge evidence for configured relation classifiers with fewer than 50 positive tasks; do not interpret their current holdout scores as stable.
- Adjudicate observed typed relations that are outside the configured model taxonomy instead of silently converting them to negatives.

## Integrity checks

- invalid spans: 0
- span text mismatches: 0
- multi label spans: 0
- duplicate spans: 0
- overlapping span pairs: 25
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 25
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 0
- multi label relations: 0
- duplicate relations: 0
- unknown result types: {}
