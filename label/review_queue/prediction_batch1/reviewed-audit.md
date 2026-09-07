# Ground-truth annotation audit

Latest export: `label/review_queue/prediction_batch1/reviewed-export-2026-07-26.json` (`5aa2376d1f79…`)

## Summary

- Tasks: 55; raw active annotation records: 55; canonical annotations: 55
- Canonical annotators: {"{'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''}": 55}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 357; relation labels: 179
- Exact duplicate texts: 0; tasks with multiple annotations: 0
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'curation_notes', 'curation_status', 'human_review_rank', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'curation_status', 'curator', 'review_tier', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 80 |
| DISEASE | 17 |
| ISOLATE | 6 |
| MEDIUM | 9 |
| ORGANISM | 49 |
| PHENOTYPE | 57 |
| SPECIES | 28 |
| STRAIN | 111 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} | 55 | 0 | 6.49 | 3.25 |

### Entity labels per 100 tasks

| Label | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|
| COMPOUND | 145.5 |
| DISEASE | 30.9 |
| ISOLATE | 10.9 |
| MEDIUM | 16.4 |
| ORGANISM | 89.1 |
| PHENOTYPE | 103.6 |
| SPECIES | 50.9 |
| STRAIN | 201.8 |

### Relation labels per 100 tasks

| Label | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|
| ASSOCIATED_WITH | 56.4 |
| DEGRADES | 21.8 |
| GROWS_ON | 21.8 |
| INFECTS | 20.0 |
| INHABITS | 30.9 |
| INHIBITS | 32.7 |
| PRESENTS | 70.9 |
| PRODUCES | 27.3 |
| PROMOTES | 10.9 |
| RESISTS | 20.0 |
| SYMBIONT_OF | 12.7 |

## Typed relation counts

| Relation | Count |
|---|---:|
| COMPOUND-STRAIN:INHIBITS | 6 |
| STRAIN-COMPOUND:DEGRADES | 12 |
| STRAIN-COMPOUND:PRODUCES | 15 |
| STRAIN-COMPOUND:RESISTS | 11 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 31 |
| STRAIN-DISEASE:INHIBITS | 4 |
| STRAIN-ISOLATE:INHABITS | 7 |
| STRAIN-MEDIUM:GROWS_ON | 12 |
| STRAIN-ORGANISM:INFECTS | 11 |
| STRAIN-ORGANISM:INHABITS | 10 |
| STRAIN-ORGANISM:INHIBITS | 3 |
| STRAIN-ORGANISM:SYMBIONT_OF | 7 |
| STRAIN-PHENOTYPE:INHIBITS | 2 |
| STRAIN-PHENOTYPE:PRESENTS | 39 |
| STRAIN-PHENOTYPE:PROMOTES | 6 |
| STRAIN-SPECIES:INHIBITS | 3 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 6 | 4 |
| STRAIN-COMPOUND:DEGRADES | 12 | 4 |
| STRAIN-COMPOUND:PRODUCES | 15 | 8 |
| STRAIN-COMPOUND:RESISTS | 11 | 4 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 31 | 6 |
| STRAIN-DISEASE:INHIBITS | 4 | 4 |
| STRAIN-ISOLATE:INHABITS | 7 | 5 |
| STRAIN-MEDIUM:GROWS_ON | 12 | 5 |
| STRAIN-ORGANISM:INFECTS | 11 | 4 |
| STRAIN-ORGANISM:INHABITS | 10 | 4 |
| STRAIN-ORGANISM:INHIBITS | 3 | 2 |
| STRAIN-ORGANISM:SYMBIONT_OF | 7 | 3 |
| STRAIN-PHENOTYPE:INHIBITS | 2 | 1 |
| STRAIN-PHENOTYPE:PRESENTS | 39 | 24 |
| STRAIN-PHENOTYPE:PROMOTES | 6 | 4 |
| STRAIN-SPECIES:INHIBITS | 3 | 1 |

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
- duplicate spans: 1
- overlapping span pairs: 11
- same entity type overlap pairs: 1
- cross entity type overlap pairs: 10
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 1
- multi label relations: 0
- duplicate relations: 0
- unknown result types: {}
