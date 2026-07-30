# NBI offline GPU workflow

The NBI compute nodes do not have internet access. The launcher therefore
submits two dependent Slurm jobs:

1. `prepare_offline.sbatch` runs on `nbi-download`. It creates one shared Conda
   environment, downloads BioLinkBERT-large, caches the Hugging Face evaluation
   modules used during training, snapshots the pinned StrainInfo catalog, and
   builds the versioned ontology alias index used by postprocessing.
2. `link_annotations.sbatch` scans the exact PMC corpus on `nbi-medium` and
   recovers unambiguous article provenance for annotation texts. Training
   splits use those PMCIDs as groups to prevent source-article leakage. The
   matcher explicitly iterates over bounded Arrow record batches and reduces
   matches per batch, so the full text column is never materialized in memory.
3. `run_model_pipeline.sbatch` runs as an offline controller and executes, in
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
export NLP4PHENO_REFRESH_ONTOLOGIES=1
export NLP4PHENO_LINK_ANNOTATIONS=0
```

`NLP4PHENO_REFRESH_ONTOLOGIES=1` intentionally downloads fresh ontology
releases; omit it to reuse the local snapshots and their manifest.

`NLP4PHENO_LINK_ANNOTATIONS=0` skips provenance regeneration and should only be
used when the configured match Parquet already corresponds to the exact
annotation export and PMC corpus used by the run.

`NLP4PHENO_STAGES` makes a retry or partial run straightforward. For example,
after training has completed:

```bash
export NLP4PHENO_STAGES="ner_pred rel_pred"
bash hpc/submit_model_pipeline.sh
```

Snakemake uses `--rerun-incomplete`, so rerunning the full list also resumes
from valid outputs rather than starting from zero.

To force a named rule and its downstream dependencies without deleting
outputs, select the relevant stage and pass a space-separated rule list:

```bash
export NLP4PHENO_STAGES="rel_pred"
export NLP4PHENO_FORCE_RULES="match_straininfo"
bash hpc/submit_model_pipeline.sh
```

Full-corpus predictions use the `ner_prediction_runtime` limit in
`hpc/config.nbi.yaml` (minutes) and `ner_prediction_mem_mb` host-memory limit.
The default NBI configuration reserves seven hours and 32 GB per non-`STRAIN`
label; the current 3.8-million-sentence corpus takes about five to six hours
per label on one A100 and needs extra host memory to assemble the final
Parquet table.

The final STRAIN-to-entity candidate join uses Polars' streaming Parquet
engine. Its `ner_merge_runtime` and `ner_merge_mem_mb` limits are also
configured in `hpc/config.nbi.yaml`; the NBI profile retains 96 GB as headroom
for join state while avoiding full Pandas materialization.

Relation preparation also streams its formatted Parquet output. Each relation
model scans that table in bounded row batches and writes positive predictions
incrementally, controlled by `rel_row_batch_size` and
`rel_inference_batch_size`. This avoids holding the full relation-candidate
table or all model outputs in host memory.

Near-duplicate entity grouping keeps the 95% token-sort similarity rule but
computes it in bounded matrices instead of allocating a dense all-pairs
matrix. `rel_group_matrix_mb` caps each temporary similarity block;
`rel_group_workers`, `rel_group_mem_mb`, and `rel_group_runtime` control its
Slurm allocation. The grouped relation table is written with Polars streaming.
The following reconciliation rule chooses one entity type for conflicting
matched edges from mean joint NER/RE confidence and removes impossible
same-taxon `INHABITS`, `INFECTS`, and `SYMBIONT_OF` candidates. Its audit is
written to `REL_output/reconciliation_summary.json`.
Network construction and PMC evidence joins also stream their outputs, using
the `rel_network_*` and `rel_link_*` limits in the NBI configuration. The
StrainInfo assembly resolver requests one CPU on `nbi-download`; its configured
worker count is an I/O thread pool rather than a CPU allocation.

After grouping, `ground_ontology` resolves each unique entity string against
the versioned offline alias index prepared under
`resources/ontologies/runtime/`. It requires no compute-node internet access.
Only unique conservative matches are accepted; ambiguous and unsupported
strings remain text nodes. The `rel_ground_runtime` and `rel_ground_mem_mb`
settings control this CPU job. The workflow retains the original text network
and additionally writes `network_ontology.tsv`,
`network_ontology_pmc.tsv`, an auditable mapping Parquet, and a JSON coverage
summary. The runtime requirement is `aliases.parquet` plus `manifest.json`;
`terms.parquet` is a build intermediate and is not required when reusing a
prepared index.

## Monitor

```bash
squeue -u "$USER"
tail -f hpc/logs/nlp4pheno-prepare-*.out
tail -f hpc/logs/nlp4pheno-models-*.out
```

Failed rule logs are printed by Snakemake and retained under `.snakemake/` and
the rule-specific output directories.
