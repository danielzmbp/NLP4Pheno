# Ground-truth annotation audit

Latest export: `label/review_queue/prediction_batch1/reviewed-export-2026-07-26-second-pass.json` (`c1712933081b…`)

## Summary

- Tasks: 55; raw active annotation records: 55; canonical annotations: 55
- Canonical annotators: {"{'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''}": 55}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 353; relation labels: 175
- Exact duplicate texts: 0; tasks with multiple annotations: 0
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'curation_notes', 'curation_status', 'human_review_rank', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'curation_status', 'curator', 'review_tier', 'second_pass_reviewed', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 80 |
| DISEASE | 17 |
| ISOLATE | 6 |
| MEDIUM | 7 |
| ORGANISM | 45 |
| PHENOTYPE | 54 |
| SPECIES | 30 |
| STRAIN | 114 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} | 55 | 0 | 6.42 | 3.18 |

### Entity labels per 100 tasks

| Label | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|
| COMPOUND | 145.5 |
| DISEASE | 30.9 |
| ISOLATE | 10.9 |
| MEDIUM | 12.7 |
| ORGANISM | 81.8 |
| PHENOTYPE | 98.2 |
| SPECIES | 54.5 |
| STRAIN | 207.3 |

### Relation labels per 100 tasks

| Label | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|
| ASSOCIATED_WITH | 54.5 |
| DEGRADES | 23.6 |
| GROWS_ON | 16.4 |
| INFECTS | 27.3 |
| INHABITS | 27.3 |
| INHIBITS | 45.5 |
| PRESENTS | 65.5 |
| PRODUCES | 18.2 |
| PROMOTES | 9.1 |
| RESISTS | 20.0 |
| SYMBIONT_OF | 10.9 |

## Typed relation counts

| Relation | Count |
|---|---:|
| COMPOUND-STRAIN:INHIBITS | 10 |
| STRAIN-COMPOUND:DEGRADES | 13 |
| STRAIN-COMPOUND:PRODUCES | 10 |
| STRAIN-COMPOUND:RESISTS | 11 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 30 |
| STRAIN-DISEASE:INHIBITS | 7 |
| STRAIN-ISOLATE:INHABITS | 7 |
| STRAIN-MEDIUM:GROWS_ON | 9 |
| STRAIN-ORGANISM:INFECTS | 15 |
| STRAIN-ORGANISM:INHABITS | 8 |
| STRAIN-ORGANISM:INHIBITS | 2 |
| STRAIN-ORGANISM:SYMBIONT_OF | 6 |
| STRAIN-PHENOTYPE:INHIBITS | 4 |
| STRAIN-PHENOTYPE:PRESENTS | 36 |
| STRAIN-PHENOTYPE:PROMOTES | 5 |
| STRAIN-SPECIES:INHIBITS | 2 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 10 | 5 |
| STRAIN-COMPOUND:DEGRADES | 13 | 4 |
| STRAIN-COMPOUND:PRODUCES | 10 | 7 |
| STRAIN-COMPOUND:RESISTS | 11 | 4 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 30 | 5 |
| STRAIN-DISEASE:INHIBITS | 7 | 5 |
| STRAIN-ISOLATE:INHABITS | 7 | 5 |
| STRAIN-MEDIUM:GROWS_ON | 9 | 4 |
| STRAIN-ORGANISM:INFECTS | 15 | 5 |
| STRAIN-ORGANISM:INHABITS | 8 | 2 |
| STRAIN-ORGANISM:INHIBITS | 2 | 1 |
| STRAIN-ORGANISM:SYMBIONT_OF | 6 | 2 |
| STRAIN-PHENOTYPE:INHIBITS | 4 | 3 |
| STRAIN-PHENOTYPE:PRESENTS | 36 | 22 |
| STRAIN-PHENOTYPE:PROMOTES | 5 | 3 |
| STRAIN-SPECIES:INHIBITS | 2 | 1 |

Configured classifiers with fewer than 50 positive tasks: ['COMPOUND-STRAIN:INHIBITS', 'STRAIN-COMPOUND:DEGRADES', 'STRAIN-COMPOUND:PRODUCES', 'STRAIN-COMPOUND:RESISTS', 'STRAIN-DISEASE:ASSOCIATED_WITH', 'STRAIN-DISEASE:INHIBITS', 'STRAIN-ISOLATE:INHABITS', 'STRAIN-MEDIUM:GROWS_ON', 'STRAIN-ORGANISM:INFECTS', 'STRAIN-ORGANISM:INHABITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-ORGANISM:SYMBIONT_OF', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-PHENOTYPE:PRESENTS', 'STRAIN-PHENOTYPE:PROMOTES', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: none
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
- overlapping span pairs: 9
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 9
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 0
- multi label relations: 0
- duplicate relations: 0
- unknown result types: {}
