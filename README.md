# LinkBERT model for phenotyope prediction

## Create PMC corpus

- Use snakemake_PMC repository to download files.
- Prepare files with `scripts/make_test_corpus.py` script and save to `corpus/` directory.

## Training

### NER training

- Adjust GPU cores to use for training in `ner.smk` file.
- Adjust `config.yaml` for labels to train on and the number of epochs
- Run

```
rm -rf NER*; snakemake --cores 20 --use-conda -s ner.smk
```

### REL training

- Adjust `config.yaml` for labels to train on and the number of epochs
- Adjust GPU cores to use for training in `rel.smk` file.
- Run

```
rm -rf REL*; snakemake --cores 20 --use-conda -s rel.smk
```

## Prediction

### NER prediction

- Run

```
snakemake --cores 20 --use-conda -s ner_pred.smk
```

### REL prediction

- Run

```
snakemake --cores 20 --use-conda -s rel_pred.smk
```

## Run all

```
rm -rf NER*; snakemake --cores 20 --use-conda -s ner.smk; rm -rf REL*; snakemake --cores 20 --use-conda -s rel.smk; snakemake --cores 20 --use-conda -s ner_pred.smk; snakemake --cores 20 --use-conda -s rel_pred.smk; python scripts/download_assemblies.py
```

### Pipeline for evolution analysis

- Run `scripts/replace_gff.sh` to make gff compatible with proteinortho
- Run `notebooks/sort_assemblies_into_categories.ipynb` to prepare files for proteinortho
- Run `scripts/run_proteinortho.sh` to run proteinortho (still need to prepare script to run per category)
