# Corrected-annotation model update — 2026-08-17

## Scope

This report compares the reviewed 4,061-task models in
`run_20260727_gold4061_4a6ea98` with the corrected 4,129-task models in
`run_20260815_gold4129_taxpass_3d696cf`.

The NER test files are byte-identical across all eight entity classes. Fourteen
of the sixteen relation test files are also unchanged. The two relation files
affected by the taxonomy correction pass are treated separately below so that
changes in gold labels are not mistaken for model improvements.

## NER results

Strict entity-span F1 on the frozen test data:

| Entity | 4,061-task model | 4,129-task model | Change |
|---|---:|---:|---:|
| STRAIN | 0.8559 | 0.8623 | +0.0064 |
| SPECIES | 0.7163 | 0.7840 | +0.0677 |
| ISOLATE | 0.4058 | 0.5000 | +0.0942 |
| COMPOUND | 0.7749 | 0.7165 | -0.0584 |
| MEDIUM | 0.7129 | 0.6990 | -0.0139 |
| ORGANISM | 0.6654 | 0.6372 | -0.0282 |
| PHENOTYPE | 0.4410 | 0.4459 | +0.0049 |
| DISEASE | 0.7321 | 0.7179 | -0.0142 |
| **Macro mean** | **0.6630** | **0.6704** | **+0.0073** |

The retrain improves the macro score, with the strongest gains in SPECIES and
ISOLATE. COMPOUND is the main NER regression and should remain a priority for
disagreement and boundary review.

## Relation results

F1 on the frozen test data. For `STRAIN-ORGANISM:INHABITS`, the saved baseline
logits were scored against the corrected labels because candidate inputs were
identical. For `STRAIN-SPECIES:INHIBITS`, 260 of 261 inputs were identical;
the remaining corrected example changed its marked endpoints. The baseline F1
is therefore reported as the exact lower and upper outcomes for that one
unknown prediction. The new-model comparison is decisive across the full
interval.

| Typed relation | Corrected old baseline | 4,129-task model | Change |
|---|---:|---:|---:|
| STRAIN-ISOLATE:INHABITS | 0.9155 | 0.9203 | +0.0048 |
| STRAIN-MEDIUM:GROWS_ON | 0.8691 | 0.8691 | 0.0000 |
| STRAIN-PHENOTYPE:PRESENTS | 0.7730 | 0.7970 | +0.0240 |
| STRAIN-ORGANISM:INHABITS | 0.5844 | 0.6667 | +0.0823 |
| STRAIN-COMPOUND:RESISTS | 0.8727 | 0.8421 | -0.0306 |
| STRAIN-PHENOTYPE:PROMOTES | 0.4800 | 0.6154 | +0.1354 |
| STRAIN-ORGANISM:INFECTS | 0.3492 | 0.5625 | +0.2133 |
| COMPOUND-STRAIN:INHIBITS | 0.7123 | 0.7632 | +0.0509 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 0.3889 | 0.3182 | -0.0707 |
| STRAIN-COMPOUND:DEGRADES | 0.6296 | 0.6552 | +0.0256 |
| STRAIN-ORGANISM:SYMBIONT_OF | 0.6000 | 0.4762 | -0.1238 |
| STRAIN-SPECIES:INHIBITS | 0.4615–0.5714 | 0.9474 | +0.3760–0.4859 |
| STRAIN-ORGANISM:INHIBITS | 0.6286 | 0.7619 | +0.1333 |
| STRAIN-PHENOTYPE:INHIBITS | 0.8889 | 0.8000 | -0.0889 |
| STRAIN-COMPOUND:PRODUCES | 0.7882 | 0.7670 | -0.0212 |
| STRAIN-DISEASE:INHIBITS | 0.5000 | 0.6000 | +0.1000 |
| **Macro mean** | **0.6526–0.6595** | **0.7101** | **+0.0506–0.0575** |

The corrected retrain is clearly better overall. Some of the largest gains are
in formerly weak classes, but rare-class scores remain uncertain because a
handful of positive examples can move F1 sharply.

## Targeted review batch

The next batch contains 55 previously unannotated full-PMC sentences and is
loaded in local Label Studio project 7. It targets the five relation classes
that regressed despite the overall gain:

| Typed relation | Tasks |
|---|---:|
| STRAIN-COMPOUND:PRODUCES | 12 |
| STRAIN-COMPOUND:RESISTS | 12 |
| STRAIN-DISEASE:ASSOCIATED_WITH | 12 |
| STRAIN-ORGANISM:SYMBIONT_OF | 12 |
| STRAIN-PHENOTYPE:INHIBITS | 7 |

The queue is pre-annotated but not pre-accepted. It starts with 12
high-confidence, ontology-grounded examples, followed by 15 high-confidence
unresolved examples, 13 relation-boundary examples, and 15 entity-tension
examples. Every entity span and relation still needs manual confirmation.

Artifacts are retained under
`label/review_queue/targeted_regressions_20260817/`.

## Decision and next steps

1. Use the 4,129-task models as the current working generation. They improve
   both NER and RE macro F1, and both corrected-holdout relation comparisons
   favour the new models.
2. Review project 7, export the completed tasks, validate them with
   `scripts/append_prediction_reviews.py`, and create a new versioned gold
   export. Do not treat the queue's suggestions as gold before submission.
3. Retrain once more with the expanded gold set while retaining the frozen
   dev/test task IDs. This is preferable to choosing old versus new models by
   their test scores, which would leak the benchmark into model selection.
4. Use old/new model disagreements as candidate generators for future review,
   especially for COMPOUND NER and the five targeted relation classes.
5. After the final retrain, run full-PMC inference once, apply strain/taxonomy
   guards and ontology grounding, and compare both the complete and
   conservative-core networks against the preceding network.

The final full-PMC inference is intentionally deferred until the 55-task batch
has been integrated; otherwise the most expensive pipeline stage would need to
be repeated immediately.
