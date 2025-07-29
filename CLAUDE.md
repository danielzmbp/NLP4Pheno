# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains **NLP4Pheno**, a machine learning pipeline for bacterial phenotype prediction using natural language processing on the PubMed corpus combined with genome analysis. The pipeline integrates Named Entity Recognition (NER), Relation Extraction (RE), and XGBoost-based phenotype prediction.

## Repository Structure

The codebase is organized around several Snakemake workflows that orchestrate the entire pipeline:

- **Core Snakemake workflows**: `ner.smk`, `rel.smk`, `ner_pred.smk`, `rel_pred.smk`, `xgboost.smk`, `evolution.smk`, `ip.smk`
- **Configuration**: `config.yaml` contains all pipeline parameters, labels, and paths
- **Python scripts**: Located in `scripts/` directory for data processing, model training, and prediction
- **Notebooks**: Analysis and visualization notebooks in `notebooks/`
- **Environment definitions**: Conda environments defined in `envs/*.yml`

## Core Components

### 1. Named Entity Recognition (NER)
- Trains separate models for each entity type: STRAIN, SPECIES, ISOLATE, COMPOUND, MEDIUM, ORGANISM, PHENOTYPE, EFFECT, DISEASE
- Uses transformer-based models (default: LinkBERT) for biomedical text understanding
- Training script: `scripts/run_ner.py`
- Pipeline: `ner.smk` (training), `ner_pred.smk` (prediction on corpus)

### 2. Relation Extraction (RE)
- Predicts relationships between entities (e.g., STRAIN-PHENOTYPE:PRESENTS, STRAIN-COMPOUND:RESISTS)
- 17 different relation types defined in `config.yaml`
- Training script: `scripts/run_seqcls.py`
- Pipeline: `rel.smk` (training), `rel_pred.smk` (prediction)

### 3. Genome Analysis & Feature Engineering
- Downloads and annotates bacterial genomes using InterProScan
- Creates protein domain features for XGBoost models
- Pipeline: `ip.smk`

### 4. Phenotype Prediction
- XGBoost models predict phenotypes based on protein domain features
- Generates feature importance scores
- Pipeline: `xgboost.smk`
- Scripts: `scripts/xgboost_binary_snakemake.py`, `scripts/xgboost_binary_snakemake_cpu.py`

### 5. Evolution Analysis
- Analyzes selective pressure on important protein domains
- Performs multiple sequence alignments and phylogenetic analysis
- Pipeline: `evolution.smk`

## Key Commands

### Environment Setup
```bash
# Create conda environments
mamba env create -f envs/base.yml
mamba env create -f envs/torch.yml
mamba env create -f envs/xgb.yml
mamba env create -f envs/l.yml
```

### Data Preparation
```bash
# Download PubMed corpus
snakemake --cores 20 --use-conda -s snakemake_PMC/Snakefile

# Prepare corpus files
python scripts/make_test_corpus.py
```

### Model Training
```bash
# Train NER models
snakemake --cores 20 --use-conda -s ner.smk

# Train relation extraction models
snakemake --cores 20 --use-conda -s rel.smk

# Get NER evaluation metrics
python scripts/run_nervaluate.py
```

### Prediction on Corpus
```bash
# Predict NER annotations on corpus
snakemake --cores 20 --use-conda -s ner_pred.smk

# Predict relations
snakemake --cores 20 --use-conda -s rel_pred.smk
```

### Genome Analysis & Phenotype Prediction
```bash
# Download and annotate genomes
snakemake --cores 20 --use-conda -s ip.smk

# Run XGBoost models
snakemake --cores 40 --use-conda -s xgboost.smk

# Evolution analysis
snakemake --cores 20 --use-conda -s evolution.smk
```

## Configuration

The `config.yaml` file controls all aspects of the pipeline:
- **Entity and relation labels**: Lists of NER entities and RE relations to predict
- **Model parameters**: Epochs, test split ratios, CUDA devices
- **Dataset identifier**: Used for corpus selection and output paths
- **File paths**: Input annotations file and output directory

Key parameters:
- `dataset`: Corpus identifier (e.g., "3103")
- `model`: Model size ("large" for LinkBERT-large)
- `cuda_devices`: GPU devices to use
- `ner_epochs`, `rel_epochs`: Training epochs
- `input_file`: Path to manually annotated training data (Label Studio format)

## Input Data

- **Manual annotations**: `label/project-5-at-2025-04-03-15-30-d43aa787.json` (Label Studio JSON format)
- **PubMed corpus**: Downloaded and processed text files in `corpus{dataset}/` directories
- **Genome data**: Downloaded from NCBI based on strain-genome mappings

## Output Structure

- **NER/**: Training data splits and model outputs
- **NER_output/**: Trained NER models and evaluation metrics
- **REL/**: Relation extraction training data
- **REL_output/**: Trained RE models and metrics
- **xgboost/annotations{dataset}/**: XGBoost results and feature importance
- **xgboost/seqfiles{dataset}/**: Sequences for evolution analysis

## Development Notes

- The pipeline uses Slurm for cluster computing (see `resources` sections in Snakemake rules)
- GPU training requires CUDA-enabled environments
- Some scripts have both GPU and CPU versions (e.g., XGBoost)
- Notebooks in `notebooks/` provide analysis and visualization of results
- The pipeline processes biomedical literature, so be mindful of large file sizes and processing times

## Important Scripts

- `scripts/run_ner.py`: Hugging Face Transformers-based NER training
- `scripts/run_seqcls.py`: Sequence classification for relation extraction
- `scripts/ner_prediction.py`: Apply trained NER models to new text
- `scripts/rel_prediction.py`: Apply trained RE models
- `scripts/xgboost_binary_snakemake.py`: XGBoost training and evaluation
- `scripts/create_evolution_dataset_snakemake.py`: Prepare data for evolution analysis

This pipeline integrates multiple machine learning approaches to extract phenotypic information from scientific literature and connect it to genomic features for bacterial phenotype prediction.