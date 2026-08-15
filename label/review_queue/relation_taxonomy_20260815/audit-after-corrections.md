# Ground-truth annotation audit

Latest export: `label/project-10-reviewed-2026-08-15-taxonomy-pass.json` (`33c35334994b…`)

## Summary

- Tasks: 4,129; raw active annotation records: 4,130; canonical annotations: 4,129
- Canonical annotators: {'1': 946, '2': 2893, 'Codex': 27, 'reviewer@localhost': 208, "{'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''}": 55}; selection policy: `ground_truth_else_latest_active`
- Entity spans: 17,433; relation labels: 7,128
- Exact duplicate texts: 0; tasks with multiple annotations: 1
- Tasks annotated independently by multiple annotators: 0
- Document provenance available: True (data fields: ['article_version', 'candidate_id', 'curation_notes', 'curation_status', 'human_review_rank', 'ontology', 'ontology_id', 'ontology_label', 'ontology_status', 'paragraph', 'pmcid', 'review_rank', 'review_summary', 'review_tier', 'sampled_relation', 'sentence_range', 'straininfo_si_id', 'straininfo_taxon', 'text']; metadata fields: ['candidate_id', 'curation_status', 'curator', 'review_append_source', 'review_tier', 'second_pass_reviewed', 'seeded_entities', 'seeded_relations', 'source'])

## Entity counts

| Entity | Count |
|---|---:|
| COMPOUND | 3,541 |
| DISEASE | 604 |
| ISOLATE | 696 |
| MEDIUM | 593 |
| ORGANISM | 2,608 |
| PHENOTYPE | 2,887 |
| SPECIES | 1,212 |
| STRAIN | 5,292 |

## Annotator profiles

The annotators have no shared independently annotated tasks, so these rates combine annotation behavior with differences in the assigned source material; they are not an agreement score.

| Annotator | Tasks | Empty | Entities/task | Relations/task |
|---|---:|---:|---:|---:|
| 1 | 946 | 49 | 4.33 | 1.84 |
| 2 | 2,893 | 175 | 4.06 | 1.64 |
| Codex | 27 | 0 | 2.59 | 0.78 |
| reviewer@localhost | 208 | 1 | 5.62 | 2.16 |
| {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} | 55 | 0 | 6.42 | 3.18 |

### Entity labels per 100 tasks

| Label | Annotator 1 | Annotator 2 | Annotator Codex | Annotator reviewer@localhost | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|---:|---:|---:|---:|
| COMPOUND | 78.2 | 80.0 | 29.6 | 192.3 | 145.5 |
| DISEASE | 20.1 | 12.4 | 29.6 | 14.4 | 30.9 |
| ISOLATE | 11.7 | 18.3 | 11.1 | 23.1 | 10.9 |
| MEDIUM | 11.6 | 15.1 | 0.0 | 18.8 | 12.7 |
| ORGANISM | 68.1 | 59.9 | 22.2 | 87.0 | 81.8 |
| PHENOTYPE | 92.6 | 63.1 | 25.9 | 59.6 | 98.2 |
| SPECIES | 32.1 | 28.4 | 25.9 | 23.1 | 54.5 |
| STRAIN | 118.1 | 129.0 | 114.8 | 143.8 | 207.3 |

### Relation labels per 100 tasks

| Label | Annotator 1 | Annotator 2 | Annotator Codex | Annotator reviewer@localhost | Annotator {'id': 1, 'email': 'reviewer@localhost', 'first_name': '', 'last_name': ''} |
|---|---:|---:|---:|---:|---:|
| ASSOCIATED_WITH | 3.2 | 3.3 | 7.4 | 5.8 | 54.5 |
| DEGRADES | 5.8 | 4.0 | 11.1 | 14.4 | 23.6 |
| GROWS_ON | 15.4 | 22.1 | 0.0 | 38.0 | 16.4 |
| INFECTS | 3.5 | 5.6 | 14.8 | 3.4 | 27.3 |
| INHABITS | 24.1 | 42.0 | 11.1 | 26.0 | 27.3 |
| INHIBITS | 35.2 | 6.6 | 7.4 | 5.3 | 45.5 |
| PRESENTS | 58.4 | 43.1 | 11.1 | 30.3 | 65.5 |
| PRODUCES | 22.9 | 21.3 | 3.7 | 79.8 | 18.2 |
| PROMOTES | 6.3 | 5.0 | 7.4 | 4.8 | 9.1 |
| RESISTS | 4.7 | 9.8 | 3.7 | 8.2 | 20.0 |
| SYMBIONT_OF | 4.9 | 1.1 | 0.0 | 0.0 | 10.9 |

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
| COMPOUND-STRAIN:INHIBITS | 223 |
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
| STRAIN-COMPOUND:PRODUCES | 929 |
| STRAIN-COMPOUND:PROMOTES | 9 |
| STRAIN-COMPOUND:RESISTS | 328 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 124 |
| STRAIN-DISEASE:INHIBITS | 71 |
| STRAIN-ISOLATE:INHABITS | 955 |
| STRAIN-MEDIUM:GROWS_ON | 785 |
| STRAIN-ORGANISM:INFECTS | 204 |
| STRAIN-ORGANISM:INHABITS | 483 |
| STRAIN-ORGANISM:INHIBITS | 67 |
| STRAIN-ORGANISM:SYMBIONT_OF | 84 |
| STRAIN-PHENOTYPE:ASSOCIATED_WITH | 13 |
| STRAIN-PHENOTYPE:INHIBITS | 61 |
| STRAIN-PHENOTYPE:PRESENTS | 1,652 |
| STRAIN-PHENOTYPE:PROMOTES | 184 |
| STRAIN-PHENOTYPE:RESISTS | 4 |
| STRAIN-SPECIES:INHIBITS | 66 |
| STRAIN-STRAIN:INHIBITS | 9 |
| STRAIN-STRAIN:RESISTS | 3 |

## Configured relation training support

| Classifier | Positive labels | Positive tasks |
|---|---:|---:|
| COMPOUND-STRAIN:INHIBITS | 223 | 77 |
| STRAIN-COMPOUND:DEGRADES | 207 | 119 |
| STRAIN-COMPOUND:PRODUCES | 929 | 353 |
| STRAIN-COMPOUND:RESISTS | 328 | 113 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 124 | 59 |
| STRAIN-DISEASE:INHIBITS | 71 | 42 |
| STRAIN-ISOLATE:INHABITS | 955 | 495 |
| STRAIN-MEDIUM:GROWS_ON | 785 | 299 |
| STRAIN-ORGANISM:INFECTS | 204 | 105 |
| STRAIN-ORGANISM:INHABITS | 483 | 242 |
| STRAIN-ORGANISM:INHIBITS | 67 | 36 |
| STRAIN-ORGANISM:SYMBIONT_OF | 84 | 42 |
| STRAIN-PHENOTYPE:INHIBITS | 61 | 43 |
| STRAIN-PHENOTYPE:PRESENTS | 1,652 | 742 |
| STRAIN-PHENOTYPE:PROMOTES | 184 | 118 |
| STRAIN-SPECIES:INHIBITS | 66 | 40 |

Configured classifiers with fewer than 50 positive tasks: ['STRAIN-DISEASE:INHIBITS', 'STRAIN-ORGANISM:INHIBITS', 'STRAIN-ORGANISM:SYMBIONT_OF', 'STRAIN-PHENOTYPE:INHIBITS', 'STRAIN-SPECIES:INHIBITS']

## Configuration drift

- Configured entities without examples: none
- Observed entities absent from config: none
- Configured relations without positive examples: none
- Observed typed relations absent from config: 61 patterns / 705 labels (9.9%); complete list is retained in the JSON audit
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
- overlapping span pairs: 1009
- same entity type overlap pairs: 0
- cross entity type overlap pairs: 1009
- missing relation endpoints: 0
- self relations: 0
- unlabeled relations: 0
- multi label relations: 6
- duplicate relations: 0
- unknown result types: {}

## Export comparison

- Older: `label/project-10-at-2025-08-21-21-08-cb43bf25.json`
- Newer: `label/project-10-reviewed-2026-08-15-taxonomy-pass.json`
- Added tasks: 150
- Removed tasks: 0
- Tasks with changed annotations: 230
- Changed-task categories: {'tasks_with_labels_changes': 218, 'tasks_with_relation_changes': 31}
- Signature changes: {'labels_added': 246, 'labels_removed': 8, 'relation_added': 40, 'relation_removed': 23}
