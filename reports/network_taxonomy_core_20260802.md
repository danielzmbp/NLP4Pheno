# NCBI Taxonomy and conservative-network benchmark — 2026-08-02

This benchmark reuses the corrected 4,061-task full-PMC model run and changes
only deterministic CPU post-processing. No model was retrained.

## Versioned results

- Corrected pre-taxonomy baseline:
  `results_20260730_qc_71a4732/preds2509`
- NCBI Taxonomy coverage baseline:
  `results_20260802_taxonomy_c88a57f/preds2509`
- Strict taxonomy-consistency and core-network result:
  `results_20260802_taxonomy_core_df6c495/preds2509`
- NCBI snapshot: official taxdump downloaded 2026-08-02, SHA-256
  `2336a67b78d2b656f254909fb21ecf9dbc03fdd3d529cb6db8a2575b503e92aa`

## Ontology-network comparison

| Measure | Corrected baseline | Strict taxonomy result | Change |
|---|---:|---:|---:|
| Evidence rows | 525,142 | 524,728 | -414 |
| Unique edges | 278,861 | 278,455 | -406 |
| Ontology-matched unique edges | 51,360 | 58,724 | +7,364 (+14.3%) |
| Matched-edge coverage | 18.418% | 21.089% | +2.671 pp |
| Matched-row coverage | 28.484% | 30.716% | +2.232 pp |
| Unique ontology concepts | 5,623 | 7,768 | +2,145 |

All 51,360 ontology-matched baseline edges remain represented. Apparent
old-only/new-only text edges mostly reflect replacement of organism strings by
NCBI taxon IDs and synonym collapse. The strict pass adds 7,364 matched edges
after the consistency filter.

NCBI grounding resolves 14,104 unique `ORGANISM` strings (242,684 prediction
rows; 31.6% row coverage) and 157 unique `SPECIES` strings (644 rows; 37.7%).
Ambiguous names such as `mouse`, `mice`, `pigs`, and `fish` remain unresolved.
The taxonomy root is excluded from automatic grounding.

## Same-taxon consistency guard

The same NCBI snapshot resolves 10,669 of 10,800 unique StrainInfo taxon
strings, covering 88.5% of their prediction rows. Exact agreement between the
strain taxon ID and an `ORGANISM`/`SPECIES` endpoint removes 244 impossible
self-relations:

| Relation | Removed evidence rows |
|---|---:|
| `STRAIN-ORGANISM:INHABITS` | 236 |
| `STRAIN-ORGANISM:INFECTS` | 3 |
| `STRAIN-ORGANISM:SYMBIONT_OF` | 5 |

The audit includes renamed synonyms such as *Nakaseomyces glabratus* / *Candida
glabrata*, *Pichia kudriavzevii* / *Candida krusei*, and *Giardia intestinalis*
/ *Giardia lamblia*. The non-`INHABITS` sentences were manually inspected and
are clear cases where the model attached infection/symbiosis language to the
strain's own taxon. `INHIBITS` is deliberately not guarded because conspecific
inhibition can be biologically meaningful.

Detailed, reproducible records are in
`taxonomy_consistency_audit_20260802.json` and
`taxonomy_consistency_removed_rows_20260802.tsv` in this directory.

## Complete and conservative network views

The text-node evidence summary contains 280,798 unique edges. The conservative
text core contains 76,331 (27.2%):

- 39,850 edges have evidence in at least two distinct PMC articles;
- 36,481 single-article edges contain at least one sentence with RE >= 0.90,
  entity NER >= 0.90, and strain match >= 0.95;
- 204,467 lower-support edges remain in the complete network but not the core.

After ontology synonym collapse and the taxon guard, the concept-aware network
has 278,455 edges and its core has 75,630. Of those core edges, 24,997 have a
unique ontology concept, representing 3,995 concepts. These core files are
additional views; the pipeline retains the full edge and sentence-evidence
tables.

## Remaining annotation weakness

Five configured relation classifiers still have fewer than 50 positive source
tasks in the current 4,061-task gold set. A 68-task, provenance-preserving,
pre-annotated queue now targets those classes under
`label/review_queue/weak_relations_20260802/`. High-confidence grounded and
unresolved cases are presented before relation-boundary and entity-tension
cases. Human decisions are not merged into the gold set automatically.
