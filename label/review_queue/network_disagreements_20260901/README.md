# Network disagreement review queue — 2026-09-01

This 40-task, pre-annotated queue samples network edges that occur in only one
of the preceding and current full-PMC prediction runs:

- 10 old-only `STRAIN-ORGANISM:INFECTS` examples;
- 10 new-only `STRAIN-ORGANISM:INFECTS` examples;
- 10 old-only `STRAIN-ORGANISM:INHABITS` examples;
- 10 new-only `STRAIN-ORGANISM:INHABITS` examples.

Examples are ranked by relation confidence followed by the weaker endpoint
confidence. Sentences already present in the 4,184-task reviewed corpus are
excluded. An old-only suggestion is still shown as a prediction so that the
reviewer can decide whether the relation was genuinely lost; a new-only
suggestion is shown so that the reviewer can decide whether it was genuinely
gained.

`queue.json` contains suggestions, not accepted annotations. Check both entity
spans and the proposed relation before submitting. In particular, use
`INFECTS` for an explicit pathogenic/infectious interaction and `INHABITS` for
host association, residence, colonization, or the broader symbiotic category.

`issues.tsv` preserves scores and provenance, `summary.json` records the
reproducible selection counts, and `label-studio-project.json` records the
local Label Studio project created for review.

## Completed review — 2026-09-02

All 40 tasks were submitted in Label Studio project 8. The raw export is
`reviewed-export-2026-09-02.json`. A reproducible semantic second pass corrected
22 tasks: 43 entity spans or labels were edited, one redundant entity was
removed, ten relations were added, and twelve were removed. Corrections include
negation, phenotype-versus-disease typing, taxonomic-family false positives,
`INFECTS` versus `INHABITS` distinctions, missing explicit pathogen links, and
boundary punctuation.

The corrected export has no invalid spans, mismatched text, duplicate spans or
relations, missing endpoints, unknown result types, or relation signatures
outside `config.yaml`. It was appended to the 4,184-task parent corpus as
`label/project-10-reviewed-2026-09-02-network-disagreement-pass.json`, producing
4,224 gold tasks. The correction, append, and audit JSON files in this directory
record every change and source hash.
