# LinkBERT model for phenotyope prediction
This repository contains the code to reproduce the analyses from the paper: [Integrating natural language processing and genome analysis enables accurate bacterial phenotype prediction](https://doi.org/10.1101/2024.12.07.627346). 

The repository consists of a series of Snakemake pipelines, scripts and notebooks to download and process data, train models for Named Entity Recognition (NER) and Relation Extration (RE), and analyze results. To reproduce, first adjust `config.yaml` to match your particular setup.

## Install python environments
To reproduce this analysis, first create the necessary Python environments:
```
mamba env create -n envs/*.yaml
```
## Create PubMed Corpus (PMC)

- Use the code in `snakemake_PMC/` to download files.
```
snakemake --cores 20 --use-conda -s snakemake_PMC/Snakefile
```
- Prepare files with `scripts/make_test_corpus.py` script and save to the `corpus{dataset}/` directory, where dataset is defined in the `config.yaml`.

## Training
The manually annotated dataset is provided in `label/project-5-at-2025-04-03-15-30-d43aa787.json` in label-studio json format.

### NER finetuning
First, we need to finetune the base model on the NER task for each of the entities. 
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

### RE finetuning

```
rm -rf REL*; snakemake --cores 20 --use-conda -s rel.smk
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

This wil first run the STRAIN model on all the PMC corpus, then apply the other NER models on sentences with strains. All of the sentences or paragraphs that include both a STRAIN entity and a phenotype entity will be kept for further analysis. The output will be saved in the directory specified in the `config.yaml` file.

### RE prediction
To predict the RE annotations in the sentences or paragraphs predicted to contain at least a STRAIN and another phenotype entity, run the following command:

```
snakemake --cores 20 --use-conda -s rel_pred.smk
```

## Download assemblies and annotate

- Run `ip.smk`to download and annotate all representative assemblies from strains that have at least one relation using Pfam with InterProScan.
  - You will need to adjust the path to your IP installation.
- Run with `scripts/ip_slurm.sh` to run using slurm.
- The output will be in `assemblies_{dataset}/` directory.

### XGBoost importances

- Run `xgboost.smk` Snakemake pipeline, config in `config.yaml` file.

```
snakemake --cores 40 --use-conda -s xgboost.smk
```

- This will first group all strains with the same phenotype relation, create the features and then run XGBoost models for each.

- The results will be saved to the output directory as `xgboost/annotations{dataset}`. The main output containing all the results is a pickle file found in `xgboost/annotations{dataset}/binary/binary.pkl`.

- This will also create `xgboost/seqfiles{dataset}` which contains the files for the evolution analysis.
  - This consists of the collated sequences with the highest importance annotations grouped by phenotype relation. 

- Then analyze with notebooks:
  - `analyze_xgboost_tidy.ipynb`.


#### Pipeline for evolution analysis

- Run `evolution.smk` to make alignments and calculate selective pressures.

```
snakemake --cores 20 --use-conda -s evolution.smk
```
- Analyze with `analyze_evolution.ipynb`.
