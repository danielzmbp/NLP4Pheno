# LinkBERT model for phenotyope prediction

## Create PMC corpus

- Use snakemake_PMC repository to download files.
- Prepare files with `scripts/make_test_corpus.py` script and save to `corpus/` directory.

## Training

### NER training

- Adjust GPU cores to use for training in `ner.smk` file.
- Adjust `ner.config` for labels to train on and the number of epochs
- Run

```
rm -rf NER*; snakemake --cores 20 --use-conda -s ner.smk
```

### REL training

- Adjust `rel.config` for labels to train on and the number of epochs
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
