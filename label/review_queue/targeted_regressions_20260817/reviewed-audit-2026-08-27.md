# Ground-truth annotation audit

Latest export: `label/review_queue/targeted_regressions_20260817/reviewed-export-2026-08-27.json` (`7f7ad34fb425…`)

## Summary

- Tasks: 55; raw active annotation records: 55; canonical annotations: 55
- Canonical annotators: {"{'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''}": 55}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 381; relation labels: 263
- Exact duplicate texts: 0; tasks with multiple annotations: 0
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'review_tier', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 95 |
| DISEASE | 27 |
| ISOLATE | 8 |
| MEDIUM | 1 |
| ORGANISM | 41 |
| PHENOTYPE | 53 |
| SPECIES | 13 |
| STRAIN | 143 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} | 55 | 0 | 6.93 | 4.78 |

### Entity labels per 100 tasks

| Label | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|
| COMPOUND | 172.7 |
| DISEASE | 49.1 |
| ISOLATE | 14.5 |
| MEDIUM | 1.8 |
| ORGANISM | 74.5 |
| PHENOTYPE | 96.4 |
| SPECIES | 23.6 |
| STRAIN | 260.0 |

### Relation labels per 100 tasks

| Label | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|
| ASSOCIATED_WITH | 87.3 |
| INFECTS | 1.8 |
| INHABITS | 83.6 |
| INHIBITS | 32.7 |
| PRESENTS | 89.1 |
| PRODUCES | 65.5 |
| PROMOTES | 9.1 |
| RESISTS | 56.4 |
| SYMBIONT_OF | 52.7 |

## Typed relation counts

| Relation | Count |
|---|---:|
| COMPOUND-STRAIN:INHIBITS | 9 |
| STRAIN-COMPOUND:PRODUCES | 36 |
| STRAIN-COMPOUND:RESISTS | 31 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 48 |
| STRAIN-DISEASE:PROMOTES | 2 |
| STRAIN-ISOLATE:INHABITS | 18 |
| STRAIN-ORGANISM:INFECTS | 1 |
| STRAIN-ORGANISM:INHABITS | 28 |
| STRAIN-ORGANISM:SYMBIONT_OF | 29 |
| STRAIN-PHENOTYPE:INHIBITS | 9 |
| STRAIN-PHENOTYPE:PRESENTS | 49 |
| STRAIN-PHENOTYPE:PROMOTES | 3 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 9 | 5 |
| STRAIN-COMPOUND:DEGRADES | 0 | 0 |
| STRAIN-COMPOUND:PRODUCES | 36 | 15 |
| STRAIN-COMPOUND:RESISTS | 31 | 10 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 48 | 11 |
| STRAIN-DISEASE:INHIBITS | 0 | 0 |
| STRAIN-ISOLATE:INHABITS | 18 | 7 |
| STRAIN-MEDIUM:GROWS_ON | 0 | 0 |
| STRAIN-ORGANISM:INFECTS | 1 | 1 |
| STRAIN-ORGANISM:INHABITS | 28 | 10 |
| STRAIN-ORGANISM:INHIBITS | 0 | 0 |
| STRAIN-ORGANISM:SYMBIONT_OF | 29 | 10 |
| STRAIN-PHENOTYPE:INHIBITS | 9 | 7 |
| STRAIN-PHENOTYPE:PRESENTS | 49 | 21 |
| STRAIN-PHENOTYPE:PROMOTES | 3 | 2 |
| STRAIN-SPECIES:INHIBITS | 0 | 0 |

Configured classifiers with fewer than 50 positive tasks: ['COMPOUND-STRAIN:INHIBITS', 'STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:PRODUCES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-ISOLATE:INHABITS', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS', 'STRAIN-ORGANISM:INHABITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-ORGANISM:SYMBIONT_OF', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-PHENOTYPE:PRESENTS', 'STRAIN-PHENOTYPE:PROMOTES', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: ['STRAIN-COMPOUND:DEGRADES', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-SPECIES:INHIBITS']
- Observed typed relations absent from config: 1 patterns / 2 labels (0.8%); complete list is retained in the JSON audit
- Of those, STRAIN-involving omissions: 1 patterns / 2 labels

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
- overlapping span pairs: 21
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 21
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 2
- multi label relations: 7
- duplicate relations: 0
- unknown result types: {}
