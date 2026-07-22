#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

export NLP4PHENO_PROJECT="${NLP4PHENO_PROJECT:-${repo_root}}"
export NLP4PHENO_CORPUS="${NLP4PHENO_CORPUS:-${NLP4PHENO_PROJECT}/snakemake_PMC/output/data/pmc_filtered.parquet}"
export NLP4PHENO_OUTPUT_PATH="${NLP4PHENO_OUTPUT_PATH:-${NLP4PHENO_PROJECT}/hpc_output}"
export NLP4PHENO_ENV_PREFIX="${NLP4PHENO_ENV_PREFIX:-${NLP4PHENO_PROJECT}/.hpc/conda/nlp4pheno}"
export NLP4PHENO_HF_HOME="${NLP4PHENO_HF_HOME:-${NLP4PHENO_PROJECT}/.hpc/huggingface}"
export NLP4PHENO_MODEL_DIR="${NLP4PHENO_MODEL_DIR:-${NLP4PHENO_PROJECT}/resources/models/BioLinkBERT-large}"
export NLP4PHENO_MODEL_ID="${NLP4PHENO_MODEL_ID:-michiyasunaga/BioLinkBERT-large}"
export NLP4PHENO_STRAININFO_VERSION="${NLP4PHENO_STRAININFO_VERSION:-2025.10}"
export NLP4PHENO_STRAININFO_DETAILED_CSV="${NLP4PHENO_STRAININFO_DETAILED_CSV:-${NLP4PHENO_PROJECT}/resources/straininfo/straininfo_synonyms.csv}"
export NLP4PHENO_JOBS="${NLP4PHENO_JOBS:-48}"
export NLP4PHENO_STAGES="${NLP4PHENO_STAGES:-ner rel ner_pred rel_pred}"

controller_partition="${NLP4PHENO_CONTROLLER_PARTITION:-nbi-long}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this launcher on an NBI Slurm login node" >&2
    exit 1
fi
if [[ ! -d "${NLP4PHENO_PROJECT}/.git" ]]; then
    echo "NLP4PHENO_PROJECT is not a Git checkout: ${NLP4PHENO_PROJECT}" >&2
    exit 1
fi
if [[ ! -s "${NLP4PHENO_CORPUS}" ]]; then
    echo "PMC corpus is missing or empty: ${NLP4PHENO_CORPUS}" >&2
    echo "Finish merging the ten upload parts, then rerun this launcher." >&2
    exit 1
fi

mkdir -p \
    "${NLP4PHENO_PROJECT}/hpc/logs" \
    "${NLP4PHENO_OUTPUT_PATH}" \
    "$(dirname -- "${NLP4PHENO_ENV_PREFIX}")" \
    "${NLP4PHENO_HF_HOME}" \
    "${NLP4PHENO_MODEL_DIR}"

cd "${NLP4PHENO_PROJECT}"

prep_submission="$(sbatch \
    --parsable \
    --export=ALL \
    hpc/prepare_offline.sbatch)"
prep_job="${prep_submission%%;*}"

pipeline_submission="$(sbatch \
    --parsable \
    --dependency="afterok:${prep_job}" \
    --partition="${controller_partition}" \
    --export=ALL \
    hpc/run_model_pipeline.sbatch)"
pipeline_job="${pipeline_submission%%;*}"

printf 'Preparation job: %s (nbi-download)\n' "${prep_job}"
printf 'Pipeline controller: %s (%s, afterok:%s)\n' \
    "${pipeline_job}" "${controller_partition}" "${prep_job}"
printf 'Monitor: squeue -j %s,%s\n' "${prep_job}" "${pipeline_job}"
printf 'Logs: %s/hpc/logs\n' "${NLP4PHENO_PROJECT}"
