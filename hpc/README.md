# NBI offline GPU workflow

The NBI compute nodes do not have internet access. The launcher therefore
submits two dependent Slurm jobs:

1. `prepare_offline.sbatch` runs on `nbi-download`. It creates one shared Conda
   environment, downloads BioLinkBERT-large, caches the Hugging Face evaluation
   modules used during training, and snapshots the pinned StrainInfo catalog.
2. `run_model_pipeline.sbatch` runs as an offline controller and executes, in
   order, `ner.smk`, `rel.smk`, `ner_pred.smk`, and `rel_pred.smk`. Snakemake
   sends GPU rules to `ei-gpu`, CPU rules to `nbi-medium`, and the one online
   StrainInfo assembly-resolution rule back to `nbi-download`.

All paths must be on storage shared by the login, download, CPU, and GPU nodes.
Do not use node-local `/tmp` for the environment, model cache, corpus, or output.
The checkout itself should be on high-capacity shared/project storage because
the trained `NER_output/` and `REL_output/` models and the `corpus2509/` text
shards are written beside the workflow files.

## Before submitting

Place the `dev` checkout and reconstructed corpus on shared storage. The ten
binary upload pieces can be joined with:

```bash
cat pmc_filtered.parquet.part-* > pmc_filtered.parquet
shasum -a 256 pmc_filtered.parquet
```

The expected SHA-256 is:

```text
24deb4209a72c26d8e178c67a82cd8b8ac70853a7e67f88645f63866923e7f45
```

Also upload `straininfo_synonyms.csv` to
`resources/straininfo/straininfo_synonyms.csv`, or point the launcher to it:

```bash
export NLP4PHENO_STRAININFO_DETAILED_CSV=/shared/path/straininfo_synonyms.csv
```

The download-queue preparation job combines it with the pinned compact
StrainInfo snapshot. GPU and ordinary compute nodes only read the resulting
local Parquet catalog.

## Submit all four stages

From the repository root on an NBI login node:

```bash
export NLP4PHENO_CORPUS=/shared/path/pmc_filtered.parquet
export NLP4PHENO_OUTPUT_PATH=/shared/path/nlp4pheno-results
bash hpc/submit_model_pipeline.sh
```

The launcher prints both Slurm job IDs. The controller starts only if the
internet-preparation job succeeds.

Useful optional settings:

```bash
export NLP4PHENO_JOBS=48
export NLP4PHENO_STAGES="ner rel ner_pred rel_pred"
export NLP4PHENO_ENV_PREFIX=/shared/path/nlp4pheno-conda
export NLP4PHENO_HF_HOME=/shared/path/huggingface-cache
export NLP4PHENO_DRY_RUN=1
```

`NLP4PHENO_STAGES` makes a retry or partial run straightforward. For example,
after training has completed:

```bash
export NLP4PHENO_STAGES="ner_pred rel_pred"
bash hpc/submit_model_pipeline.sh
```

Snakemake uses `--rerun-incomplete`, so rerunning the full list also resumes
from valid outputs rather than starting from zero.

## Monitor

```bash
squeue -u "$USER"
tail -f hpc/logs/nlp4pheno-prepare-*.out
tail -f hpc/logs/nlp4pheno-models-*.out
```

Failed rule logs are printed by Snakemake and retained under `.snakemake/` and
the rule-specific output directories.
