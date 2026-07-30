# Full-PMC network quality comparison — 2026-07-28

## Scope

This report compares the successful pre-review full-PMC run in `results/`
(completed 2026-07-25) with the reviewed-annotation run in
`results_20260727_gold4061_4a6ea98/` (completed 2026-07-28).

The intervening timestamped result directories contain failed
startup/provenance attempts and are not valid baselines.

## Annotation changes

The old training export contained 3,979 tasks. The new export contains 4,061:

- 82 new fully annotated tasks
- 124 existing tasks augmented during review
- 557 additional entity spans
- 211 additional relations
- No accepted entity spans or relations from common tasks were removed

The added entity spans comprise 188 COMPOUND, 145 STRAIN, 70 PHENOTYPE,
51 ORGANISM, 37 SPECIES, 33 ISOLATE, 25 DISEASE, and 8 MEDIUM spans.

The old split had no task-to-PMC provenance. The new split links 1,710 tasks
to PMCIDs and keeps tasks from the same paper in one split. The new evaluation
therefore prevents document leakage and is methodologically stronger.

## Model metrics

These values are directional rather than a controlled A/B test because the
test sets changed when document-level grouping was introduced.

| Model | Old macro test F1 | New macro test F1 | Change |
|---|---:|---:|---:|
| NER (8 labels) | 0.629 | 0.663 | +0.034 |
| RE (16 classifiers) | 0.612 | 0.654 | +0.042 |

NER test F1 by entity:

| Entity | Old | New | Change |
|---|---:|---:|---:|
| STRAIN | 0.828 | 0.856 | +0.028 |
| SPECIES | 0.763 | 0.716 | -0.047 |
| ISOLATE | 0.381 | 0.406 | +0.025 |
| COMPOUND | 0.739 | 0.775 | +0.036 |
| MEDIUM | 0.618 | 0.713 | +0.095 |
| ORGANISM | 0.685 | 0.665 | -0.020 |
| PHENOTYPE | 0.462 | 0.441 | -0.021 |
| DISEASE | 0.559 | 0.732 | +0.174 |

The new RE models recover two relation families absent from the old predicted
network: STRAIN–ORGANISM:INHABITS and STRAIN–PHENOTYPE:INHIBITS. The former
has new test F1 0.566. The latter has F1 0.889 but only five positive test
examples, so that estimate is highly uncertain.

## Network comparison

| Measure | Old | New | Change |
|---|---:|---:|---:|
| TSV evidence rows | 799,028 | 847,280 | +6.0% |
| Unique biological evidence records | 798,966 | 839,530 | +5.1% |
| Article-supported edges | 694,344 | 730,629 | +5.2% |
| Unique string-level edges | 429,105 | 462,058 | +7.7% |
| Unique matched strains | 27,711 | 30,797 | +11.1% |
| Unique non-strain entities | 134,237 | 148,477 | +10.6% |
| Supporting PMCIDs | 129,056 | 136,700 | +5.9% |

All rows have valid PMCID and provenance fields, all relation signatures are
valid, and neither run contains self-loop rows or exact duplicate rows.

String-level edge churn is substantial:

- 247,495 unique edges overlap
- 214,563 are new-only
- 181,610 are old-only
- New-network overlap is 53.6%; Jaccard similarity is 38.5%

Some churn is merely a surface-form change. Ontology normalization raises
new-network overlap for matched concept edges to 66.8% and Jaccard similarity
to 49.5%.

## Ontology grounding

| Measure | Old | New | Change |
|---|---:|---:|---:|
| Matched row coverage | 28.66% | 25.87% | -2.79 pp |
| Matched unique edges | 84,177 | 82,718 | -1.7% |
| Unique ontology concepts | 7,408 | 7,501 | +1.3% |

The overall coverage decline is driven partly by the large increase in
ORGANISM relations: ORGANISM and SPECIES currently have no ontology grounding,
and unsupported evidence rows rose from 27,702 to 80,738.

## Main quality findings

### 1. Strain alias matching is the largest immediate problem

The highest-volume strain node, `SI-ID362271`, occurs in 44,228 evidence rows
(5.2% of the new network). It is labelled *Paraliobacillus ryukyuensis*, but
the matcher assigns E. coli O157/H7 mentions to it through the short contained
catalogue alias `O157`. This is a false match: O157 is being used as a
serogroup in those mentions, not as that catalogue strain designation.

The same audit found other short-designation failures, including `SA187`
being assigned to a *Trichophyton mentagrophytes* catalogue record even when
the paper refers to the plant-associated SA187 strain.

The current taxonomy guard misses lowercase normalized strain mentions because
its scientific-name regular expressions require capitalization.

### 2. Entity-type ambiguity is now visible

The old network had 62 edge/evidence records duplicated across NER types. The
new network has 7,750:

- 6,026 INHABITS records
- 1,724 INHIBITS records

These are not exact duplicate TSV rows. They are the same biological
source-target-relation-evidence assignment emitted with competing entity
types, especially ISOLATE versus ORGANISM.

### 3. Some new high-confidence edges are structurally implausible

Examples include Candida strain IDs linked by INHABITS or INHIBITS to
`Candida albicans` as an ORGANISM. Relation confidence alone is near 1.0, but
the endpoints show that the sentence-level candidate construction or
post-processing needs a same-taxon guard.

### 4. Rare classes remain unstable

The weakest new test F1 values are:

- STRAIN–ORGANISM:INFECTS: 0.349
- STRAIN–DISEASE:ASSOCIATED_WITH: 0.389
- STRAIN–PHENOTYPE:PROMOTES: 0.480
- STRAIN–DISEASE:INHIBITS: 0.500
- STRAIN–SPECIES:INHIBITS: 0.500
- STRAIN–ORGANISM:INHABITS: 0.566

PHENOTYPE and ISOLATE NER also remain weak at 0.441 and 0.406.

## Recommended next iteration

1. Fix StrainInfo matching before using this as the final network:
   make taxonomy extraction case-insensitive; reject four-character contained
   aliases unless the mention supplies compatible taxonomy; quarantine short
   non-collection designations without taxonomic support; and add regression
   tests for O157 and SA187-like cases.
2. Rerun only strain matching and its downstream grouping, grounding, and
   network rules. NER and RE do not need retraining for this correction.
3. Add a same-taxon/self-reference guard for ORGANISM and SPECIES relations.
4. Reconcile duplicate ISOLATE/ORGANISM and ORGANISM/SPECIES predictions,
   retaining an explicit ambiguity flag when the text does not decide.
5. Ground ORGANISM and SPECIES with NCBI Taxonomy, and extend curated
   abbreviation/synonym maps for the highest-frequency unresolved MEDIUM,
   PHENOTYPE, and ISOLATE terms.
6. Build the next annotation queue from weak-class uncertainty, model
   disagreement, cross-type conflicts, and high-impact strain hubs—not only
   high-confidence predictions.
7. Publish both a complete evidence network and a conservative core network
   using per-relation calibrated thresholds, strong strain matches, endpoint
   consistency, ontology status, and number of independent supporting PMCIDs.
8. For a causal annotation benchmark, retrain old and new annotations using
   the same frozen document-grouped splits. The current old and new metric
   files do not use the same test documents.

## Reproducible artifacts

On NBI:

- `analysis/network_comparison_20260728.json`
- `analysis/network_change_samples_20260728.tsv`

In the repository:

- `scripts/compare_network_runs.py`
- `scripts/sample_network_changes.py`
