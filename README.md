# NLP4Pheno: Bacterial Phenotype Prediction Pipeline

This repository contains the code to reproduce the analyses from the paper: [Integrating natural language processing and genome analysis enables accurate bacterial phenotype prediction](https://doi.org/10.1101/2024.12.07.627346).

The pipeline integrates Named Entity Recognition (NER), Relation Extraction (RE), and XGBoost-based phenotype prediction using Snakemake workflows. 

## Prerequisites

- [Snakemake](https://snakemake.readthedocs.io/) (≥6.0)
- [Mamba](https://mamba.readthedocs.io/) or Conda
- NCBI API key (for genome downloads)
- InterProScan installation (for protein annotation)

## Configuration

Before running any pipeline, adjust `config.yaml` to match your setup. Key parameters:
- `dataset`: Corpus identifier (determines output directories)
- `cuda_devices`: GPU devices for training
- `input_file`: Path to manually annotated training data

## Environment Setup

Create the required Python environments:
```
mamba env create -f envs/nlp4pheno.yml
mamba env create -f envs/pytorch.yml
mamba env create -f envs/xgb.yml
```
## Create PubMed Corpus (PMC)

- Use the code in `snakemake_PMC/` to download files.
```
snakemake --cores 20 --use-conda -s snakemake_PMC/Snakefile
```
- Prepare corpus files using the `scripts/make_test_corpus.py` script. Files are saved to `corpus{dataset}/` directory (where `{dataset}` is defined in `config.yaml`).

## Model Training

The manually annotated dataset is provided in `label/project-5-at-2025-04-03-15-30-d43aa787.json` (Label Studio JSON format).

### Named Entity Recognition (NER)
Train NER models for each entity type (STRAIN, SPECIES, PHENOTYPE, etc.): 
```
snakemake --cores 20 --use-conda -s ner.smk
```

The output for this will be saved to the base directory. It includes:
- `NER/`: directory with the data splits for the training and testing.
- `NER_output/`: directory with the trained model and metrics.

To obtain the partial metrics with Nervaluate, run the following script:

```
python scripts/run_nervaluate.py
```
They will be generated to the same output folder.

### Relation Extraction (RE)
Train models to predict relationships between entities:
```bash
rm -rf REL*
snakemake --cores 20 --use-conda -s rel.smk
```
The output for this will be saved to the base directory. It includes:
- `REL/`: directory with the data for the training and testing.
- `REL_output/`: directory with the trained model and metrics.

## Prediction

### NER prediction
To predict the NER annotations in the PMC corpus as produced earlier, run the following command:

```
snakemake --cores 20 --use-conda -s ner_pred.smk
```

This will first run the STRAIN model on all the PMC corpus, then apply the other NER models on sentences with strains. All of the sentences or paragraphs that include both a STRAIN entity and a phenotype entity will be kept for further analysis. The output will be saved in the directory specified in the `config.yaml` file.

### RE prediction
To predict the RE annotations in the sentences or paragraphs predicted to contain at least a STRAIN and another phenotype entity, run the following command:

```
snakemake --cores 20 --use-conda -s rel_pred.smk
```

## Genome Analysis

### Download and Annotate Assemblies
1. Create a `.ncbi_api_key` file in the root directory with your NCBI API key
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
- Analysis notebook: `analyze_xgboost_tidy.ipynb`


### Evolution Analysis
Analyze selective pressure on important protein domains:
```bash
snakemake --cores 20 --use-conda -s evolution.smk
```
Analysis notebook: `analyze_evolution.ipynb`
