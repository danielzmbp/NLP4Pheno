# Ontology grounding pilot

Generated: 2026-07-23T10:41:28.019441+00:00

Only unique exact label/synonym matches and conservative normalized matches are accepted. No fuzzy or embedding match is counted as grounded.

This is a coverage benchmark against annotated mention strings, not a concept-ID accuracy gold standard. Ambiguous candidates require review.

Annotation source: `label/project-10-reviewed-2026-07-22.json` (SHA-256 `9420f8747571…`).

## Coverage

| Entity | Mentions | Matched | Ambiguous | Unmatched | Coverage | Unique surfaces |
|---|---:|---:|---:|---:|---:|---:|
| COMPOUND | 3,299 | 1,572 | 177 | 1,550 | 47.6% | 1,824 |
| DISEASE | 552 | 209 | 10 | 333 | 37.9% | 385 |
| ISOLATE | 660 | 118 | 10 | 532 | 17.9% | 498 |
| MEDIUM | 582 | 154 | 23 | 405 | 26.5% | 312 |
| PHENOTYPE | 2,744 | 270 | 12 | 2,462 | 9.8% | 1,932 |

## Ontology snapshots

- **CHEBI**: 218,253 terms; 616,466 labels/synonyms; version `253`; SHA-256 `81132a8c34a1…`
- **ENVO**: 3,872 terms; 7,250 labels/synonyms; version `releases/2026-06-26`; SHA-256 `7f5a6580d1b5…`
- **FOODON**: 28,372 terms; 36,253 labels/synonyms; version `not declared`; SHA-256 `1e11fc50283c…`
- **MCO**: 839 terms; 839 labels/synonyms; version `releases/2019-05-15`; SHA-256 `4ac8e44997b6…`
- **MEDIADIVE**: 3,338 terms; 5,613 labels/synonyms; version `not declared`; SHA-256 `d53f6d4319f1…`
- **MONDO**: 32,095 terms; 113,882 labels/synonyms; version `releases/2026-07-06`; SHA-256 `75c51066741e…`
- **OBA**: 25,344 terms; 51,529 labels/synonyms; version `releases/2026-07-14`; SHA-256 `8bfbab7fdf11…`
- **OMP**: 2,006 terms; 2,511 labels/synonyms; version `releases/2024-03-25`; SHA-256 `d31f1a993c39…`
- **PATO**: 1,887 terms; 2,803 labels/synonyms; version `releases/2025-05-14/pato.obo`; SHA-256 `9b65efdf7d8d…`
- **UBERON**: 14,977 terms; 52,808 labels/synonyms; version `releases/2026-06-19`; SHA-256 `7f06d8e84420…`

## COMPOUND: accepted-match breakdown

- Ontologies: CHEBI 1,572
- Methods: abbreviation_exact 29, abbreviation_normalized 1, direct_exact 1,298, direct_formula 210, direct_normalized 34

Frequent ambiguous mentions:

| Surface | Count |
|---|---:|
| nitrogen | 17 |
| copper | 12 |
| sulfur | 9 |
| cysteine | 9 |
| gold | 9 |
| phosphate | 8 |
| erythromycin | 8 |
| alanine | 6 |
| methionine | 6 |
| thiosulfate | 6 |
| glycine | 6 |
| hydrogen | 6 |
| oxygen | 6 |
| proline | 5 |
| fructose | 4 |
| arsenic | 4 |
| polyamine | 4 |
| indole | 3 |
| Asp | 3 |
| Glu | 3 |

## COMPOUND: frequent unmatched mentions

| Surface | Count |
|---|---:|
| GABA | 25 |
| RLs | 19 |
| IAA | 17 |
| EPS | 17 |
| LPS | 12 |
| PHA | 10 |
| β-lactams | 10 |
| Ni | 9 |
| ROS | 9 |
| HM | 9 |
| ITA | 8 |
| DMSO | 8 |
| MDA | 8 |
| IPTG | 8 |
| tryptone | 6 |
| fluoroquinolones | 6 |
| siderophores | 6 |
| exopolysaccharides | 6 |
| Cd | 6 |
| Pb | 6 |

## DISEASE: accepted-match breakdown

- Ontologies: MONDO 209
- Methods: abbreviation_exact 1, direct_exact 208

Frequent ambiguous mentions:

| Surface | Count |
|---|---:|
| FoP | 2 |
| AD | 2 |
| PD | 2 |
| ARDS | 2 |
| hepatoma | 1 |
| RCC | 1 |

## DISEASE: frequent unmatched mentions

| Surface | Count |
|---|---:|
| CPP | 6 |
| bacteremia | 4 |
| outbreaks | 4 |
| outbreak | 4 |
| rice blast | 3 |
| pathogen infection | 3 |
| BPS | 3 |
| tumours | 3 |
| ash dieback | 3 |
| pandemic | 3 |
| obese | 3 |
| lung infection | 2 |
| vascular congestion | 2 |
| colorectal cancer tumor | 2 |
| blast | 2 |
| M. oryzae infection | 2 |
| granuloma | 2 |
| sepsis | 2 |
| PE | 2 |
| CRPC | 2 |

## ISOLATE: accepted-match breakdown

- Ontologies: ENVO 68, FOODON 23, UBERON 27
- Methods: direct_exact 118

Frequent ambiguous mentions:

| Surface | Count |
|---|---:|
| milk | 8 |
| cheese | 1 |
| lettuce | 1 |

## ISOLATE: frequent unmatched mentions

| Surface | Count |
|---|---:|
| human stool | 7 |
| ice sample | 7 |
| marine | 5 |
| kefir grains | 4 |
| mahewu | 4 |
| fecal samples | 4 |
| CF patients | 3 |
| stevia seeds | 3 |
| sputum sample | 3 |
| urine sample | 3 |
| sediment sample | 3 |
| nitrocellulose-contaminated wastewater | 3 |
| saline soil | 3 |
| human feces | 3 |
| soft cheese | 3 |
| clover silage | 3 |
| olives | 3 |
| soil and rhizosphere samples | 3 |
| traditional sourdoughs | 2 |
| cucumber fermentation brine | 2 |

## MEDIUM: accepted-match breakdown

- Ontologies: MCO 22, MEDIADIVE 132
- Methods: abbreviation_exact 24, abbreviation_normalized 2, direct_exact 121, direct_normalized 7

Frequent ambiguous mentions:

| Surface | Count |
|---|---:|
| TSB | 10 |
| ONR7a | 3 |
| nutrient agar | 3 |
| ISP 3 | 1 |
| NA | 1 |
| TYG | 1 |
| NMS | 1 |
| CM | 1 |
| Thermus 162 medium | 1 |
| TH162 | 1 |

## MEDIUM: frequent unmatched mentions

| Surface | Count |
|---|---:|
| M9 | 12 |
| DMEM | 11 |
| CDM | 9 |
| SCFM2 | 7 |
| MRS broth | 6 |
| LB agar | 6 |
| TSA | 6 |
| tryptic soy broth | 6 |
| brain heart infusion | 6 |
| minimal medium | 5 |
| PBS | 5 |
| BG agar | 4 |
| CM9 | 4 |
| MRS agar | 4 |
| V4 | 3 |
| coconut beverage | 3 |
| butter milk | 3 |
| LB agar plates | 3 |
| PDA | 3 |
| RPMI 1640 | 3 |

## PHENOTYPE: accepted-match breakdown

- Ontologies: OBA 1, OMP 185, PATO 84
- Methods: direct_exact 110, direct_normalized 160

Frequent ambiguous mentions:

| Surface | Count |
|---|---:|
| virulence | 6 |
| filamentous | 5 |
| Filamentous | 1 |

## PHENOTYPE: frequent unmatched mentions

| Surface | Count |
|---|---:|
| biofilm | 43 |
| biofilm formation | 28 |
| spores | 18 |
| biofilms | 18 |
| plant growth | 17 |
| non-pathogenic | 15 |
| probiotic | 14 |
| thermophilic | 14 |
| non-motile | 13 |
| motile | 13 |
| endophytic | 12 |
| fermentation | 12 |
| MRSA | 12 |
| spore | 11 |
| pathogenic | 11 |
| antimicrobial | 10 |
| non-spore-forming | 9 |
| antifungal activity | 9 |
| MDR | 8 |
| antimicrobial activity | 7 |
