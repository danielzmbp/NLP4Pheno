# Ground-truth annotation audit

Latest export: `label/project-10-reviewed-2026-09-02-network-disagreement-pass.json` (`a4ae63517a74…`)

## Summary

- Tasks: 4,224; raw active annotation records: 4,225; canonical annotations: 4,224
- Canonical annotators: {'1': 986, '2': 2893, 'Codex': 27, 'reviewer@localhost': 208, "{'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''}": 110}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 18,045; relation labels: 7,514
- Exact duplicate texts: 0; tasks with multiple annotations: 1
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'curation_notes', 'curation_status', 'human_review_rank', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'curation_status', 'curator', 'review_append_source', 'review_direction', 'review_tier', 'second_pass_reviewed', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 3,649 |
| DISEASE | 640 |
| ISOLATE | 710 |
| MEDIUM | 598 |
| ORGANISM | 2,723 |
| PHENOTYPE | 2,953 |
| SPECIES | 1,242 |
| STRAIN | 5,530 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| 1 | 986 | 49 | 4.39 | 1.90 |
| 2 | 2,893 | 175 | 4.06 | 1.64 |
| Codex | 27 | 0 | 2.59 | 0.78 |
| reviewer@localhost | 208 | 1 | 5.62 | 2.16 |
| {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} | 110 | 0 | 6.62 | 3.94 |

### Entity labels per 100 tasks

| Label | Annotator 1 | Annotator 2 | Annotator Codex | Annotator reviewer@localhost | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|---:|---:|---:|---:|
| COMPOUND | 76.6 | 80.0 | 29.6 | 192.3 | 157.3 |
| DISEASE | 20.4 | 12.4 | 29.6 | 14.4 | 38.2 |
| ISOLATE | 11.9 | 18.3 | 11.1 | 23.1 | 12.7 |
| MEDIUM | 11.6 | 15.1 | 0.0 | 18.8 | 7.3 |
| ORGANISM | 73.0 | 59.9 | 22.2 | 87.0 | 76.4 |
| PHENOTYPE | 90.2 | 63.1 | 25.9 | 59.6 | 97.3 |
| SPECIES | 32.6 | 28.4 | 25.9 | 23.1 | 39.1 |
| STRAIN | 122.9 | 129.0 | 114.8 | 143.8 | 233.6 |

### Relation labels per 100 tasks

| Label | Annotator 1 | Annotator 2 | Annotator Codex | Annotator reviewer@localhost | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|---:|---:|---:|---:|
| ASSOCIATED_WITH | 3.3 | 3.3 | 7.4 | 5.8 | 70.0 |
| DEGRADES | 5.6 | 4.0 | 11.1 | 14.4 | 11.8 |
| GROWS_ON | 16.2 | 22.1 | 0.0 | 38.0 | 8.2 |
| INFECTS | 6.1 | 5.6 | 14.8 | 3.4 | 14.5 |
| INHABITS | 30.5 | 42.0 | 11.1 | 26.0 | 55.5 |
| INHIBITS | 33.8 | 6.6 | 7.4 | 5.3 | 38.2 |
| PRESENTS | 57.0 | 43.1 | 11.1 | 30.3 | 77.3 |
| PRODUCES | 22.0 | 21.3 | 3.7 | 79.8 | 40.0 |
| PROMOTES | 6.1 | 5.0 | 7.4 | 4.8 | 9.1 |
| RESISTS | 4.5 | 9.8 | 3.7 | 8.2 | 38.2 |
| SYMBIONT_OF | 4.8 | 1.1 | 0.0 | 0.0 | 30.9 |

## Typed relation counts

| Relation | Count |
|---|---:|
| COMPOUND-COMPOUND:DEGRADES | 1 |
| COMPOUND-COMPOUND:INHIBITS | 1 |
| COMPOUND-COMPOUND:PRODUCES | 1 |
| COMPOUND-COMPOUND:PROMOTES | 1 |
| COMPOUND-DISEASE:ASSOCIATED_WITH | 5 |
| COMPOUND-ORGANISM:INHIBITS | 19 |
| COMPOUND-PHENOTYPE:ASSOCIATED_WITH | 1 |
| COMPOUND-PHENOTYPE:INHIBITS | 11 |
| COMPOUND-PHENOTYPE:PRESENTS | 48 |
| COMPOUND-PHENOTYPE:PRODUCES | 2 |
| COMPOUND-PHENOTYPE:PROMOTES | 19 |
| COMPOUND-SPECIES:INHIBITS | 3 |
| COMPOUND-STRAIN:INHIBITS | 232 |
| COMPOUND-STRAIN:PROMOTES | 1 |
| DISEASE-DISEASE:ASSOCIATED_WITH | 1 |
| DISEASE-ORGANISM:INHIBITS | 1 |
| ISOLATE-DISEASE:ASSOCIATED_WITH | 1 |
| MEDIUM-PHENOTYPE:INHIBITS | 1 |
| MEDIUM-PHENOTYPE:PROMOTES | 2 |
| ORGANISM-COMPOUND:PRODUCES | 35 |
| ORGANISM-COMPOUND:RESISTS | 2 |
| ORGANISM-DISEASE:ASSOCIATED_WITH | 7 |
| ORGANISM-DISEASE:PROMOTES | 1 |
| ORGANISM-DISEASE:RESISTS | 1 |
| ORGANISM-ISOLATE:INHABITS | 13 |
| ORGANISM-MEDIUM:GROWS_ON | 70 |
| ORGANISM-ORGANISM:INFECTS | 10 |
| ORGANISM-ORGANISM:INHABITS | 17 |
| ORGANISM-ORGANISM:INHIBITS | 9 |
| ORGANISM-PHENOTYPE:ASSOCIATED_WITH | 1 |
| ORGANISM-PHENOTYPE:PRESENTS | 64 |
| ORGANISM-PHENOTYPE:PROMOTES | 1 |
| ORGANISM-PHENOTYPE:RESISTS | 1 |
| ORGANISM-SPECIES:RESISTS | 1 |
| ORGANISM-STRAIN:RESISTS | 7 |
| PHENOTYPE-ISOLATE:INHABITS | 1 |
| PHENOTYPE-PHENOTYPE:INHIBITS | 3 |
| PHENOTYPE-PHENOTYPE:PRESENTS | 7 |
| SPECIES-COMPOUND:DEGRADES | 8 |
| SPECIES-COMPOUND:PRODUCES | 42 |
| SPECIES-COMPOUND:RESISTS | 9 |
| SPECIES-DISEASE:ASSOCIATED_WITH | 16 |
| SPECIES-DISEASE:PRESENTS | 2 |
| SPECIES-ISOLATE:INFECTS | 1 |
| SPECIES-ISOLATE:INHABITS | 23 |
| SPECIES-MEDIUM:GROWS_ON | 17 |
| SPECIES-ORGANISM:INFECTS | 6 |
| SPECIES-ORGANISM:INHABITS | 24 |
| SPECIES-ORGANISM:INHIBITS | 9 |
| SPECIES-ORGANISM:PROMOTES | 1 |
| SPECIES-ORGANISM:SYMBIONT_OF | 1 |
| SPECIES-PHENOTYPE:ASSOCIATED_WITH | 1 |
| SPECIES-PHENOTYPE:INHIBITS | 2 |
| SPECIES-PHENOTYPE:PRESENTS | 127 |
| SPECIES-PHENOTYPE:PROMOTES | 3 |
| SPECIES-SPECIES:INHIBITS | 1 |
| STRAIN-COMPOUND:DEGRADES | 207 |
| STRAIN-COMPOUND:INHIBITS | 4 |
| STRAIN-COMPOUND:PRODUCES | 963 |
| STRAIN-COMPOUND:PROMOTES | 9 |
| STRAIN-COMPOUND:RESISTS | 359 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 174 |
| STRAIN-DISEASE:INHIBITS | 71 |
| STRAIN-ISOLATE:INHABITS | 986 |
| STRAIN-MEDIUM:GROWS_ON | 799 |
| STRAIN-ORGANISM:INFECTS | 232 |
| STRAIN-ORGANISM:INHABITS | 571 |
| STRAIN-ORGANISM:INHIBITS | 67 |
| STRAIN-ORGANISM:SYMBIONT_OF | 113 |
| STRAIN-PHENOTYPE:ASSOCIATED_WITH | 13 |
| STRAIN-PHENOTYPE:INHIBITS | 69 |
| STRAIN-PHENOTYPE:PRESENTS | 1,711 |
| STRAIN-PHENOTYPE:PROMOTES | 189 |
| STRAIN-PHENOTYPE:RESISTS | 4 |
| STRAIN-SPECIES:INHIBITS | 66 |
| STRAIN-STRAIN:INHIBITS | 9 |
| STRAIN-STRAIN:RESISTS | 3 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 232 | 82 |
| STRAIN-COMPOUND:DEGRADES | 207 | 119 |
| STRAIN-COMPOUND:PRODUCES | 963 | 367 |
| STRAIN-COMPOUND:RESISTS | 359 | 123 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 174 | 70 |
| STRAIN-DISEASE:INHIBITS | 71 | 42 |
| STRAIN-ISOLATE:INHABITS | 986 | 508 |
| STRAIN-MEDIUM:GROWS_ON | 799 | 301 |
| STRAIN-ORGANISM:INFECTS | 232 | 123 |
| STRAIN-ORGANISM:INHABITS | 571 | 273 |
| STRAIN-ORGANISM:INHIBITS | 67 | 36 |
| STRAIN-ORGANISM:SYMBIONT_OF | 113 | 54 |
| STRAIN-PHENOTYPE:INHIBITS | 69 | 49 |
| STRAIN-PHENOTYPE:PRESENTS | 1,711 | 768 |
| STRAIN-PHENOTYPE:PROMOTES | 189 | 121 |
| STRAIN-SPECIES:INHIBITS | 66 | 40 |

Configured classifiers with fewer than 50 positive tasks: ['STRAIN-DISEASE:INHIBITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: none
- Observed typed relations absent from config: 61 patterns / 705 labels (9.4%); complete list is retained in the JSON audit
- Of those, STRAIN-involving omissions: 8 patterns / 50 labels

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
- overlapping span pairs: 1031
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 1031
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 0
- multi label relations: 12
- duplicate relations: 0
- unknown result types: {}
