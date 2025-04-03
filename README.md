# LinkBERT model for phenotyope prediction
This repository contains the code to reproduce the analyses from the paper: [Integrating natural language processing and genome analysis enables accurate bacterial phenotype prediction](https://doi.org/10.1101/2024.12.07.627346). 

The repository consists of a series of Snakemake pipelines, scripts and notebooks to download and process data, train models, and analyze results.

## Install python environments
```
mamba env create -n envs/*.yaml
```
## Create PubMed Corpus (PMC)

- Use the code in `snakemake_PMC/` to download files.
```
snakemake --cores 20 --use-conda -s snakemake_PMC/Snakefile
```
- Prepare files with `scripts/make_test_corpus.py` script and save to the `corpus/` directory.

## Training

### NER finetuning

- Adjust GPU cores to use for training in `ner.smk` file.
- Adjust `config.yaml` for labels to train on and the number of epochs

```
rm -rf NER*; snakemake --cores 20 --use-conda -s ner.smk
```

### RE finetuning

- Adjust `config.yaml` for labels to train on and the number of epochs
- Adjust GPU cores to use for training in `rel.smk` file.

```
rm -rf REL*; snakemake --cores 20 --use-conda -s rel.smk
```

## Prediction

### NER prediction

```
snakemake --cores 20 --use-conda -s ner_pred.smk
```

### REL prediction

```
snakemake --cores 20 --use-conda -s rel_pred.smk
```

## Download assemblies and annotate

- Run `ip.smk`to download and annotate assemblies using Pfam with InterProScan.
  - You might need to adjust the path to your IP installation.
- Run with `scripts/ip_slurm.sh` to run using slurm.

### XGBoost importances

- Run `xgboost.smk` Snakemake pipeline, config in `config.yaml` file.

```
snakemake --cores 40 --use-conda -s xgboost.smk
```

- Then analyze with notebooks:
  - `analyze_xgboost_tidy.ipynb` for binary classification with either gain as metric, respectively.


#### Pipeline for evolution analysis

- Create evolution dataset using `scripts/create_evolution_dataset.py`.
  - This will collate sequences  with the highest importance annotations grouped by phenotype relation. 
- Run `evolution.smk` to make alignments and calculate selective pressures.
```
snakemake --cores 20 --use-conda -s evolution.smk
```
- Analyze with `analyze_evolution.ipynb`.
