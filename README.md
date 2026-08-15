# NLP4Pheno: Bacterial Phenotype Prediction Pipeline

[![DOI](https://zenodo.org/badge/736778244.svg)](https://doi.org/10.5281/zenodo.17473326)

This repository contains the code to reproduce the analyses from the paper: [Integrating natural language processing and genome analysis enables accurate bacterial phenotype prediction](https://doi.org/10.1101/2024.12.07.627346).

The pipeline integrates Named Entity Recognition (NER), Relation Extraction (RE), and XGBoost-based phenotype prediction using Snakemake workflows. 

## Requirements

### System Requirements
- Python ≥3.8
- [Snakemake](https://snakemake.readthedocs.io/) ≥6.0
- [Mamba](https://mamba.readthedocs.io/) or Conda ≥4.9
- CUDA-capable GPU (recommended for training)
- At least ~50GB free disk space for the PMC build; model inference and genome
  analysis require additional working space
- 16GB+ RAM recommended

### External Dependencies
- NCBI API key (optional, speeds up genome downloads)
- [InterProScan](https://interproscan-docs.readthedocs.io/) ≥5.0 (for protein annotation)
  - Download and extract to your system
  - Set `interproscan_path` in `config.yaml`

## Configuration

Before running any pipeline, adjust `config.yaml` to match your setup.

## Configuration Parameters

| Parameter | Description | Default/Example |
|-----------|-------------|----------------|
| `dataset` | Corpus identifier (determines output directories) | `1108` |
| `cuda_devices` | GPU devices for training | `[0]` |
| `input_file` | Path to manually annotated training data | `label/project-10-reviewed-2026-07-23.json` |
| `ner_epochs` | Training epochs for NER models | `15` |
| `rel_epochs` | Training epochs for RE models | `25` |
| `ner_test` | Test split ratio for NER | `0.2` |
| `rel_test` | Test split ratio for RE | `0.3` |
| `cutoff_prediction` | Confidence threshold for predictions | `0.50` |
| `core_min_pmcs` | Independent PMC articles needed for multi-article core support | `2` |
| `core_min_relation_score` | RE threshold for single-article core evidence | `0.90` |
| `core_min_entity_score` | NER threshold for single-article core evidence | `0.90` |
| `core_min_strain_score` | Strain-match threshold for single-article core evidence | `0.95` |
| `model` | Model size (base/large) | `"large"` |
| `seed` | Random seed for reproducibility | `97` |
| `output_path` | Base output directory | `/pfs/work9/workspace/scratch/tu_kmpaj01-link` |
| `pmc_parquet_file` | PMC corpus data file | `snakemake_PMC/output/data/pmc_filtered.parquet` |
| `straininfo_designations_file` | Cleaned compact + detailed StrainInfo alias union | `resources/straininfo/designations_union.parquet` |

### Entity Types
```yaml
ner_labels: [STRAIN, SPECIES, ISOLATE, COMPOUND, MEDIUM, ORGANISM, PHENOTYPE, DISEASE]
```

### Relationship Types
```yaml
rel_labels:
  - STRAIN-ISOLATE:INHABITS
  - STRAIN-MEDIUM:GROWS_ON
  - STRAIN-PHENOTYPE:PRESENTS
  - STRAIN-ORGANISM:INHABITS
  - STRAIN-COMPOUND:RESISTS
  # ... (16 total relationship types)
```

## Installation

### 1. Environment Setup

Create the main environments:
```bash
# Create primary environment
mamba env create -f envs/nlp4pheno.yml
# Create PMC corpus processing environment
mamba env create -f envs/pmc.yml
```

**Note:** Additional environments (`pytorch.yml`, `xgb.yml`) are created automatically by Snakemake when needed.

### 2. Activate Environment

Activate the main environment for running Snakemake:
```bash
conda activate nlp4pheno
```

### 3. Verify Installation

Test your setup:
```bash
# Check Snakemake
snakemake --version

# Check available environments
conda env list

# Test configuration parsing
snakemake -n -s ner.smk
```
## Create PubMed Corpus (PMC)

- The PMC builder uses the current ESearch and versioned PMC Open Data on AWS
  services. Configure the release snapshot in `snakemake_PMC/config.yaml`, set a
  contact email for NCBI, and validate the workflow with its offline fixture:

```bash
export NCBI_EMAIL='name@example.org'
python -m unittest discover -s snakemake_PMC/tests -p 'test_*.py'
snakemake -s snakemake_PMC/Snakefile \
  --configfile snakemake_PMC/tests/config.fixture.yaml --cores 2
```

- Build the complete snapshot on SLURM:

```bash
snakemake -s snakemake_PMC/Snakefile \
  --use-conda --executor slurm --jobs 40 --rerun-incomplete
```

The final `snakemake_PMC/output/data/pmc_filtered.parquet` already has the
`pmcid`, `paragraph`, `sentence_range`, and `text` provenance required by the
inference workflow. See `snakemake_PMC/README.md` for schemas and release
artifacts.

## Model Training

### NBI Slurm cluster

For the NBI cluster, where compute and A100 nodes have no internet access, use
the staged launcher in `hpc/`. It prepares the shared environment, pretrained
model, metric modules, and StrainInfo snapshot on `nbi-download`, then runs NER
training, relation training, and full-corpus inference through Snakemake on
`nbi-medium` and `ei-gpu`:

```bash
export NLP4PHENO_CORPUS=/shared/path/pmc_filtered.parquet
export NLP4PHENO_OUTPUT_PATH=/shared/path/nlp4pheno-results
bash hpc/submit_model_pipeline.sh
```

See `hpc/README.md` for resumable stage selection, cache paths, and monitoring.

## Data Preparation

### Annotation Data
The current manually reviewed dataset is provided in
`label/project-10-reviewed-2026-08-15.json` (Label Studio JSON format). It
contains 4,129 tasks, including the audited 68-task weak-relation review batch.
The 2025 export and intermediate 2026 review batches are retained unchanged as
auditable sources.
When a task has multiple active annotations, preprocessing selects a marked
ground-truth record or otherwise the most recently updated record.

`config.yaml` also pins `label/frozen_splits_gold4061_4a6ea98.json`. These are
the exact dev/test task IDs used by the 2026-07-27 gold-4061 model. Later tasks
are added to training only; a task sharing a PMC article with frozen dev/test
data is excluded. This keeps model comparisons fair and prevents article-level
leakage.

Audit the raw exports and their agreement with `config.yaml` before training:

```bash
python scripts/audit_annotations.py \
  label/project-10-at-2025-08-21-21-08-cb43bf25.json \
  label/project-10-reviewed-2026-08-15.json \
  --config config.yaml \
  --json-output label/annotation_audit_2026-08-15.json \
  --markdown-output label/annotation_audit_2026-08-15.md
```

**Format Requirements:**
- Label Studio JSON export format
- Must contain examples for every entity and typed relation configured in
  `config.yaml`
- Annotations should include entity spans and relationship labels
- Keep a PMCID or another document identifier in future exports so evaluation
  can be split by source article, not only by annotation task

The historical export has no document identifier. After building the PMC
corpus, recover only verifiable literal matches and retain ambiguous sources:

```bash
python scripts/link_annotations_to_pmc.py \
  label/project-10-reviewed-2026-08-15.json \
  snakemake_PMC/output/data/pmc_filtered.parquet \
  --matches-output label/annotation_pmc_matches.parquet \
  --summary-output label/annotation_pmc_summary.json
```

Only tasks with status `unique_pmcid` are safe to use for a source-article
grouped split. The script deliberately reports rather than guesses ambiguous
and unmatched tasks. Set `annotation_pmc_matches_file` in `config.yaml` to the
generated Parquet before rebuilding the NER and relation splits.

### Corpus Files
Corpus files are automatically generated from the PMC parquet data during the NER prediction step. The `ner_pred.smk` pipeline will:
- Read from `pmc_parquet_file` specified in config.yaml
- Generate numbered text files (e.g., `0.txt`, `1.txt`) in `corpus{dataset}/` directory
- Process files in chunks for efficient prediction

### Named Entity Recognition (NER)
Build and inspect the CPU-only NER datasets before scheduling GPU training:

```bash
snakemake --cores 1 -s ner.smk NER/dataset_summary.json
```

The workflow converts character offsets directly to token-level B/I/O data;
it no longer depends on an undeclared Label Studio converter executable. When
`annotation_pmc_matches_file` is set, tasks with a unique recovered PMCID are
kept together in one split for both NER and relation training.
Source text, including word-internal hyphens, is preserved unchanged through
training and inference so entity offsets and PMC provenance remain exact.

Train NER models for each entity type (STRAIN, SPECIES, PHENOTYPE, etc.):
```bash
snakemake --cores 20 --use-conda -s ner.smk
```

**Requirements:** 8GB GPU memory recommended

**Output:**
- `NER/`: Data splits for training and testing
- `NER_output/`: Trained models, metrics, and Nervaluate evaluation results

### Relation Extraction (RE)
Train models to predict relationships between entities:
```bash
snakemake --cores 20 --use-conda -s rel.smk
```

Relation examples are marked using their annotated character offsets and are
split by recovered PMCID when it is unique, otherwise by source task. This
prevents linked sentences from the same paper from crossing train, development,
and test sets. Multi-label entity pairs remain one example and count as positive
in every applicable binary classifier. The CPU-only preparation target is:

```bash
snakemake --cores 1 -s rel.smk REL/split_summary.json
```

**Requirements:** 8GB GPU memory recommended

**Output:**
- `REL/`: Data for training and testing
- `REL_output/`: Trained models and metrics

### Strain registry

The retired StrainSelect download is not a reproducible source for a new build.
Snapshot the maintained StrainInfo alias catalog instead; its version and every
page hash are recorded alongside the Parquet files:

```bash
python scripts/fetch_straininfo_catalog.py \
  --expected-version 2025.10 \
  --strains-output resources/straininfo/strains.parquet \
  --designations-output resources/straininfo/designations.parquet \
  --summary-output resources/straininfo/summary.json
```

Exact normalized designations can be resolved locally to persistent SI-IDs;
ambiguous aliases remain explicit. Detailed StrainInfo API records should be
requested only for resolved SI-IDs that need genome assemblies.

Build the cleaned union after obtaining a detailed StrainInfo synonym CSV. The
compact snapshot remains unchanged and supplies authoritative aliases plus any
SI-IDs missed by the detailed download:

```bash
python scripts/build_straininfo_union.py \
  --compact resources/straininfo/designations.parquet \
  --detailed /path/to/straininfo_synonyms.csv \
  --output resources/straininfo/designations_union.parquet \
  --summary-output resources/straininfo/union_summary.json
```

The union records whether each alias came from the compact catalog, a detailed
deposit designation, or a detailed cross-reference. Matching prefers compact
assertions when a detailed cross-reference creates a collision. Weak aliases
(purely numeric or shorter than four normalized characters) are not resolved
without supporting taxonomy.

Audit matching against the annotated STRAIN mentions before using the catalog
in prediction:

```bash
python scripts/audit_straininfo_matches.py \
  label/project-10-reviewed-2026-07-23.json \
  resources/straininfo/designations_union.parquet \
  --output label/straininfo_match_audit.json
```

The audit accepts strong exact aliases and complete token-bounded identifiers
inside a longer mention. It rejects weak uncontextualized aliases and taxonomy
contradictions and has no fuzzy fallback; unmatched and ambiguous mentions
remain unresolved rather than receiving a plausible-looking but unsupported
genome identifier.

### Ontology grounding

Entity strings can be grounded after NER instead of being treated as unrelated
spellings. The pilot terminology index currently covers:

| Entity | Sources |
|---|---|
| `PHENOTYPE` | OMP, OBA, PATO |
| `COMPOUND` | ChEBI |
| `DISEASE` | Mondo |
| `ISOLATE` | ENVO, UBERON, FoodOn |
| `MEDIUM` | MCO, MediaDive |
| `SPECIES`, `ORGANISM` | NCBI Taxonomy |

Build a local, versioned index and benchmark it against the reviewed
annotations:

```bash
python scripts/build_ontology_index.py
python scripts/ground_ontology_annotations.py \
  label/project-10-reviewed-2026-07-23.json
```

The downloaded releases and generated Parquet indexes are written below
`resources/ontologies/runtime/` and are deliberately ignored by Git. Its
`manifest.json` records each resolved release, byte size, and SHA-256. The
tracked report is `label/ontology_grounding_pilot.md`.

Automatic grounding is deliberately conservative: an exact preferred label,
exact ontology synonym, source-derived MediaDive alias, locally defined
abbreviation, or case-preserving ChEBI formula must identify one concept.
Multiple candidates remain `ambiguous`; fuzzy and embedding similarities are
not auto-accepted. `STRAIN` continues to use the StrainInfo resolver, while
`SPECIES` and `ORGANISM` use unique exact names and conservative synonyms from
NCBI Taxonomy. The pilot
measures resolvable coverage, not concept-level accuracy, so newly introduced
sources should still be sampled before their matches are used in the final
network.

After `group_entities`, the relation workflow reconciles competing entity
types for the same strain/entity/relation edge using mean joint NER/RE
confidence. It also removes same-taxon `INHABITS`, `INFECTS`, and
`SYMBIONT_OF` candidates, while retaining potentially meaningful conspecific
`INHIBITS` predictions. `REL_output/reconciliation_summary.json` records the
effect of this step. Ontology grounding then retains each reconciled prediction
and its original `word_qc_group`, while adding the ontology ID, canonical
label, match method, rule-based evidence strength, and ambiguity status.
StrainInfo taxon names are independently grounded to the same NCBI snapshot;
exact taxon-ID agreement removes additional impossible same-taxon rows that
spelling comparison missed. It produces:

- `REL_output/preds_straininfo_grounded.pqt`: predictions with ontology and
  strain-taxonomy columns after the taxon-ID consistency guard.
- `REL_output/ontology_groundings.parquet`: one auditable mapping per entity
  type and grouped string, including all candidates for ambiguous mappings.
- `REL_output/strain_taxonomy_groundings.parquet`: the independent NCBI
  mapping of StrainInfo taxon strings used by the same-taxon guard.
- `REL_output/ontology_grounding_summary.json`: coverage by entity type,
  ontology, and matching rule, plus frequent unresolved strings.
- `network_ontology.tsv` and `network_ontology_pmc.tsv`: concept-aware network
  and evidence table. Unique matches use the ontology ID as the node, so
  synonymous surfaces merge; ambiguous, unmatched, and unsupported values keep
  their original grouped text as the node.

`network.tsv` remains the text-node baseline, and `network_pmc.tsv` adds its
sentence-level evidence and confidence columns. `STRAIN` is still identified
by StrainInfo.

Both network variants also produce an edge-level evidence summary and a
conservative core (`network_evidence_summary.tsv`, `network_core.tsv`,
`network_ontology_evidence_summary.tsv`, and `network_ontology_core.tsv`). The
core keeps edges supported by at least `core_min_pmcs` distinct PMC articles,
or by one sentence whose RE, NER, and strain scores all pass the configured
strict thresholds. The complete evidence tables are always retained.

StrainInfo matching rejects serogroup/serotype labels such as `O157` as strain
identifiers. Taxonomy checks accept normalized lowercase scientific names,
and short catalogue aliases require authoritative provenance or compatible
taxonomy. This avoids turning common serotype or short laboratory labels into
unrelated SI-IDs.

For uniquely resolved SI-IDs, the detailed API can provide genome accessions.
The resolver retains the response hash for each genome and selects one assembly
per SI-ID by assembly level, then recency, avoiding multiple near-identical
assemblies from the same strain being treated as independent observations.

## Prediction

### NER prediction
Apply trained NER models to the PMC corpus:

```bash
snakemake --cores 20 --use-conda -s ner_pred.smk
```

**Process:** Runs STRAIN model on entire corpus, then applies other NER models on strain-containing sentences. Keeps sentences with both STRAIN and phenotype entities.

**Requirements:** GPU recommended

**Output:** Saved to directory specified in `config.yaml`

### RE prediction
Apply trained RE models to extract relationships:

```bash
snakemake --cores 20 --use-conda -s rel_pred.smk
```

**Process:** Analyzes sentences containing STRAIN and phenotype entities,
matches strain mentions conservatively to the pinned StrainInfo alias catalog,
resolves one preferred genome per unique SI-ID, and creates a provenance-linked
network. There is no fuzzy strain fallback.

**Requirements:** GPU recommended

## Genome Analysis

### Download and Annotate Assemblies
1. Optionally create a `.ncbi_api_key` file in the root directory with your NCBI API key (recommended to speed up downloads)
2. Adjust InterProScan installation path in the pipeline
3. Run genome download and annotation:
```bash
snakemake --cores 20 --use-conda -s ip.smk
```

The workflow validates the generated `strain/assembly` manifest through a
Snakemake checkpoint. A clean run cannot silently expand to zero genomes; an
empty or malformed manifest stops the workflow.
   
Output will be saved to `assemblies_{dataset}/` directory.

### Phenotype Prediction
Run XGBoost models for phenotype prediction based on protein domains:
```bash
snakemake --cores 40 --use-conda -s xgboost.smk
```

**Output:**
- `xgboost/annotations{dataset}/binary/binary.pkl`: Main results file
- `xgboost/seqfiles{dataset}/`: Sequences for evolution analysis
- `xgboost/features{dataset}/`: Feature matrices and importance scores
- Analysis notebook: `analyze_xgboost_tidy.ipynb`

## Output Files Overview

### NER Training
```
NER/                          # Training data splits
├── {ENTITY}/
│   ├── train.json           # Training set
│   ├── dev.json             # Validation set
│   └── test.json            # Test set

NER_output/                   # Model outputs
├── {ENTITY}/
│   ├── model.safetensors    # Trained model
│   ├── all_results.json     # Training metrics
│   ├── overall_results.json # Nervaluate results
│   └── test_predictions.txt # Test predictions
└── aggregated_eval.tsv      # Combined metrics
```

### RE Training
```
REL/                          # Training data
├── {RELATION}/
│   ├── train.json
│   ├── dev.json
│   └── test.json

REL_output/                   # Model outputs
├── {RELATION}/
│   ├── model.safetensors
│   └── all_results.json
└── all_metrics.tsv
```

### Prediction Outputs
```
corpus{dataset}/              # Input corpus files
preds{dataset}/               # NER/RE predictions
├── NER_output/
└── REL_output/

assemblies_{dataset}/         # Genome data
├── {strain}/
│   ├── genomic.fna          # Genome sequence
│   └── protein.faa          # Protein sequences

xgboost/                      # ML outputs
├── annotations{dataset}/
├── features{dataset}/
└── seqfiles{dataset}/
```


### Evolution Analysis
Analyze selective pressure on important protein domains:
```bash
snakemake --cores 20 --use-conda -s evolution.smk
```
Analysis notebook: `analyze_evolution.ipynb`

## Pipeline Workflow

```
PMC Corpus Creation
│
├── Manual Annotations (Label Studio)
│   │
│   ├── 1. NER Training (ner.smk) ────────────┐
│   │                                         │
│   └── 2. RE Training (rel.smk)              │
│       │                                     │
│       └── 3. NER Prediction (ner_pred.smk) ─┴─┐
│           │                                   │
│           └── 4. RE Prediction (rel_pred.smk) │
│               │                               │
│               ├── 5. Genome Download & Annotation (ip.smk)
│               │   │
│               └── 6. XGBoost Phenotype Prediction (xgboost.smk)
│                   │
│                   └── 7. Evolution Analysis (evolution.smk)
```

**Dependencies:**
- Steps 3-4: Require trained models from steps 1-2
- Step 5: Can run independently after step 4
- Step 6: Requires outputs from steps 4-5
- Step 7: Requires outputs from step 6

## Citation

If you use this pipeline, please cite:

```bibtex
@article{nlp4pheno2024,
  title={Integrating natural language processing and genome analysis enables accurate bacterial phenotype prediction},
  author={[Authors]},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.12.07.627346},
  url={https://doi.org/10.1101/2024.12.07.627346}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
