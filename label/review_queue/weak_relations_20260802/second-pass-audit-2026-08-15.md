# Ground-truth annotation audit

Latest export: `label/review_queue/weak_relations_20260802/reviewed-export-2026-08-15-second-pass.json` (`a547a3289ee5…`)

## Summary

- Tasks: 68; raw active annotation records: 68; canonical annotations: 68
- Canonical annotators: {'1': 68}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 438; relation labels: 161
- Exact duplicate texts: 0; tasks with multiple annotations: 0
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'review_tier', 'second_pass_reviewed', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 54 |
| DISEASE | 31 |
| ISOLATE | 3 |
| MEDIUM | 3 |
| ORGANISM | 88 |
| PHENOTYPE | 69 |
| SPECIES | 45 |
| STRAIN | 145 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| 1 | 68 | 0 | 6.44 | 2.37 |

### Entity labels per 100 tasks

| Label | Annotator 1 |
|---|---:|
| COMPOUND | 79.4 |
| DISEASE | 45.6 |
| ISOLATE | 4.4 |
| MEDIUM | 4.4 |
| ORGANISM | 129.4 |
| PHENOTYPE | 101.5 |
| SPECIES | 66.2 |
| STRAIN | 213.2 |

### Relation labels per 100 tasks

| Label | Annotator 1 |
|---|---:|
| DEGRADES | 1.5 |
| INHABITS | 22.1 |
| INHIBITS | 108.8 |
| PRESENTS | 36.8 |
| PRODUCES | 25.0 |
| PROMOTES | 5.9 |
| SYMBIONT_OF | 36.8 |

## Typed relation counts

| Relation | Count |
|---|---:|
| COMPOUND-STRAIN:INHIBITS | 14 |
| STRAIN-COMPOUND:DEGRADES | 1 |
| STRAIN-COMPOUND:PRODUCES | 17 |
| STRAIN-DISEASE:INHIBITS | 27 |
| STRAIN-ISOLATE:INHABITS | 2 |
| STRAIN-ORGANISM:INHABITS | 13 |
| STRAIN-ORGANISM:INHIBITS | 10 |
| STRAIN-ORGANISM:SYMBIONT_OF | 25 |
| STRAIN-PHENOTYPE:INHIBITS | 10 |
| STRAIN-PHENOTYPE:PRESENTS | 25 |
| STRAIN-PHENOTYPE:PROMOTES | 4 |
| STRAIN-SPECIES:INHIBITS | 13 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 14 | 2 |
| STRAIN-COMPOUND:DEGRADES | 1 | 1 |
| STRAIN-COMPOUND:PRODUCES | 17 | 8 |
| STRAIN-COMPOUND:RESISTS | 0 | 0 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 0 | 0 |
| STRAIN-DISEASE:INHIBITS | 27 | 11 |
| STRAIN-ISOLATE:INHABITS | 2 | 2 |
| STRAIN-MEDIUM:GROWS_ON | 0 | 0 |
| STRAIN-ORGANISM:INFECTS | 0 | 0 |
| STRAIN-ORGANISM:INHABITS | 13 | 8 |
| STRAIN-ORGANISM:INHIBITS | 10 | 5 |
| STRAIN-ORGANISM:SYMBIONT_OF | 25 | 7 |
| STRAIN-PHENOTYPE:INHIBITS | 10 | 7 |
| STRAIN-PHENOTYPE:PRESENTS | 25 | 14 |
| STRAIN-PHENOTYPE:PROMOTES | 4 | 4 |
| STRAIN-SPECIES:INHIBITS | 13 | 8 |

Configured classifiers with fewer than 50 positive tasks: ['COMPOUND-STRAIN:INHIBITS', 'STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:PRODUCES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-ISOLATE:INHABITS', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS', 'STRAIN-ORGANISM:INHABITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-ORGANISM:SYMBIONT_OF', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-PHENOTYPE:PRESENTS', 'STRAIN-PHENOTYPE:PROMOTES', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: ['STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS']
- Observed typed relations absent from config: 0 patterns / 0 labels (0.0%); complete list is retained in the JSON audit
- Of those, STRAIN-involving omissions: 0 patterns / 0 labels

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
- overlapping span pairs: 26
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 26
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 0
- multi label relations: 0
- duplicate relations: 0
- unknown result types: {}
