# NLP4Pheno: Bacterial Phenotype Prediction Pipeline

This repository contains the code to reproduce the analyses from the paper: [Integrating natural language processing and genome analysis enables accurate bacterial phenotype prediction](https://doi.org/10.1101/2024.12.07.627346).

The pipeline integrates Named Entity Recognition (NER), Relation Extraction (RE), and XGBoost-based phenotype prediction using Snakemake workflows. 

## Requirements

### System Requirements
- Python ≥3.8
- [Snakemake](https://snakemake.readthedocs.io/) ≥6.0
- [Mamba](https://mamba.readthedocs.io/) or Conda ≥4.9
- CUDA-capable GPU (recommended for training)
- ~50GB free disk space for full pipeline
- 16GB+ RAM recommended

### External Dependencies
- NCBI API key (optional, speeds up genome downloads)
- [InterProScan](https://interproscan-docs.readthedocs.io/) ≥5.0 (for protein annotation)
  - Download and extract to your system
  - Update the path in `ip.smk` at line containing `interproscan.sh`

## Configuration

Before running any pipeline, adjust `config.yaml` to match your setup.

## Configuration Parameters

| Parameter | Description | Default/Example |
|-----------|-------------|----------------|
| `dataset` | Corpus identifier (determines output directories) | `1108` |
| `cuda_devices` | GPU devices for training | `[0]` |
| `input_file` | Path to manually annotated training data | `label/project-10-at-2025-08-21-21-08-cb43bf25.json` |
| `ner_epochs` | Training epochs for NER models | `15` |
| `rel_epochs` | Training epochs for RE models | `25` |
| `ner_test` | Test split ratio for NER | `0.2` |
| `rel_test` | Test split ratio for RE | `0.3` |
| `cutoff_prediction` | Confidence threshold for predictions | `0.50` |
| `model` | Model size (base/large) | `"large"` |
| `seed` | Random seed for reproducibility | `97` |
| `output_path` | Base output directory | `/pfs/work9/workspace/scratch/tu_kmpaj01-link` |
| `pmc_parquet_file` | PMC corpus data file | `snakemake_PMC/output/data/pmc_filtered.parquet` |

### Entity Types
```yaml
ner_labels: [STRAIN, SPECIES, ISOLATE, COMPOUND, MEDIUM, ORGANISM, PHENOTYPE, EFFECT, DISEASE]
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

- Use the code in `snakemake_PMC/` to download files (requires pmc environment):
```bash
conda activate pmc
snakemake --cores 20 --use-conda --executor slurm -s snakemake_PMC/Snakefile
```
- Prepare corpus files using the `scripts/make_test_corpus.py` script. Files are saved to `corpus{dataset}/` directory (where `{dataset}` is defined in `config.yaml`).

## Model Training

## Data Preparation

### Annotation Data
The manually annotated dataset is provided in `label/project-10-at-2025-08-21-21-08-cb43bf25.json` (Label Studio JSON format).

**Format Requirements:**
- Label Studio JSON export format
- Must contain annotations for all 9 entity types: `STRAIN`, `SPECIES`, `ISOLATE`, `COMPOUND`, `MEDIUM`, `ORGANISM`, `PHENOTYPE`, `EFFECT`, `DISEASE`
- Annotations should include entity spans and relationship labels

### Corpus Files
Corpus files are automatically generated from the PMC parquet data during the NER prediction step. The `ner_pred.smk` pipeline will:
- Read from `pmc_parquet_file` specified in config.yaml
- Generate numbered text files (e.g., `0.txt`, `1.txt`) in `corpus{dataset}/` directory
- Process files in chunks for efficient prediction

### Named Entity Recognition (NER)
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
rm -rf REL*
snakemake --cores 20 --use-conda -s rel.smk
```

**Requirements:** 8GB GPU memory recommended

**Output:**
- `REL/`: Data for training and testing
- `REL_output/`: Trained models and metrics

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

**Process:** Analyzes sentences containing STRAIN and phenotype entities to extract relationships.

**Requirements:** GPU recommended

## Genome Analysis

### Download and Annotate Assemblies
1. Optionally create a `.ncbi_api_key` file in the root directory with your NCBI API key (recommended to speed up downloads)
2. Adjust InterProScan installation path in the pipeline
3. Run genome download and annotation:
```bash
snakemake --cores 20 --use-conda -s ip.smk
```
   
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
