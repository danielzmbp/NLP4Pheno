# Ground-truth annotation audit

Latest export: `label/review_queue/network_disagreements_20260901/reviewed-export-2026-09-02-second-pass.json` (`7e1a84d503c1…`)

## Summary

- Tasks: 40; raw active annotation records: 40; canonical annotations: 40
- Canonical annotators: {'1': 40}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 237; relation labels: 128
- Exact duplicate texts: 0; tasks with multiple annotations: 0
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'review_direction', 'second_pass_reviewed', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 15 |
| DISEASE | 11 |
| ISOLATE | 6 |
| MEDIUM | 4 |
| ORGANISM | 76 |
| PHENOTYPE | 13 |
| SPECIES | 17 |
| STRAIN | 95 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| 1 | 40 | 0 | 5.92 | 3.20 |

### Entity labels per 100 tasks

| Label | Annotator 1 |
|---|---:|
| COMPOUND | 37.5 |
| DISEASE | 27.5 |
| ISOLATE | 15.0 |
| MEDIUM | 10.0 |
| ORGANISM | 190.0 |
| PHENOTYPE | 32.5 |
| SPECIES | 42.5 |
| STRAIN | 237.5 |

### Relation labels per 100 tasks

| Label | Annotator 1 |
|---|---:|
| ASSOCIATED_WITH | 7.5 |
| GROWS_ON | 35.0 |
| INFECTS | 67.5 |
| INHABITS | 182.5 |
| PRESENTS | 25.0 |
| SYMBIONT_OF | 2.5 |

## Typed relation counts

| Relation | Count |
|---|---:|
| STRAIN-DISEASE:ASSOCIATED_WITH | 3 |
| STRAIN-ISOLATE:INHABITS | 13 |
| STRAIN-MEDIUM:GROWS_ON | 14 |
| STRAIN-ORGANISM:INFECTS | 27 |
| STRAIN-ORGANISM:INHABITS | 60 |
| STRAIN-ORGANISM:SYMBIONT_OF | 1 |
| STRAIN-PHENOTYPE:PRESENTS | 10 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 0 | 0 |
| STRAIN-COMPOUND:DEGRADES | 0 | 0 |
| STRAIN-COMPOUND:PRODUCES | 0 | 0 |
| STRAIN-COMPOUND:RESISTS | 0 | 0 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 3 | 1 |
| STRAIN-DISEASE:INHIBITS | 0 | 0 |
| STRAIN-ISOLATE:INHABITS | 13 | 6 |
| STRAIN-MEDIUM:GROWS_ON | 14 | 2 |
| STRAIN-ORGANISM:INFECTS | 27 | 17 |
| STRAIN-ORGANISM:INHABITS | 60 | 21 |
| STRAIN-ORGANISM:INHIBITS | 0 | 0 |
| STRAIN-ORGANISM:SYMBIONT_OF | 1 | 1 |
| STRAIN-PHENOTYPE:INHIBITS | 0 | 0 |
| STRAIN-PHENOTYPE:PRESENTS | 10 | 6 |
| STRAIN-PHENOTYPE:PROMOTES | 0 | 0 |
| STRAIN-SPECIES:INHIBITS | 0 | 0 |

Configured classifiers with fewer than 50 positive tasks: ['COMPOUND-STRAIN:INHIBITS', 'STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:PRODUCES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-ISOLATE:INHABITS', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS', 'STRAIN-ORGANISM:INHABITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-ORGANISM:SYMBIONT_OF', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-PHENOTYPE:PRESENTS', 'STRAIN-PHENOTYPE:PROMOTES', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: ['COMPOUND-STRAIN:INHIBITS', 'STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:PRODUCES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-PHENOTYPE:PROMOTES', 'STRAIN-SPECIES:INHIBITS']
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
- overlapping span pairs: 5
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 5
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 0
- multi label relations: 1
- duplicate relations: 0
- unknown result types: {}
