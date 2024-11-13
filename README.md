# LinkBERT model for phenotyope prediction
## Install python environments
```
mamba env create -n envs/*.yaml
```
## Create PMC corpus

- Use `snakemake_PMC/` to download files.
- Prepare files with `scripts/make_test_corpus.py` script and save to `corpus/` directory.

## Training

### NER training

- Adjust GPU cores to use for training in `ner.smk` file.
- Adjust `config.yaml` for labels to train on and the number of epochs

```
rm -rf NER*; snakemake --cores 20 --use-conda -s ner.smk
```

### REL training

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

- Run `ip.smk`to download and annotate assemblies.
- Run with `scripts/ip_slurm.sh` to run using slurm.

### XGBoost importances

- Run xgboost.smk snakemake pipeline, config in `config.yaml` file.

```
snakemake --cores 40 --use-conda -s xgboost.smk
```

- Then analyze with notebooks:
  - `analyze_xgboost_binary.ipynb` and `analyze_xgboost_binary_gain.ipynb` for binary classification with either weight or gain as metric, respectively.


#### Pipeline for evolution analysis

- Create evolution dataset using `scripts/create_evolution_dataset.py`.
- Run `evolution.smk` to make alignments and calculate selective pressures.
- Analyze with `analyze_evolution.ipynb`.
