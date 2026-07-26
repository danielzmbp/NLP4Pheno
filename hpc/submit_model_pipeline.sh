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
export NLP4PHENO_LINK_ANNOTATIONS="${NLP4PHENO_LINK_ANNOTATIONS:-1}"

controller_partition="${NLP4PHENO_CONTROLLER_PARTITION:-nbi-long}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this launcher on an NBI Slurm login node" >&2
    exit 1
fi
if [[ ! -s "${NLP4PHENO_PROJECT}/config.yaml" \
      || ! -s "${NLP4PHENO_PROJECT}/ner.smk" \
      || ! -s "${NLP4PHENO_PROJECT}/rel.smk" ]]; then
    echo "NLP4PHENO_PROJECT does not contain the required workflow: ${NLP4PHENO_PROJECT}" >&2
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

controller_dependency="${prep_job}"
link_job=""
if [[ "${NLP4PHENO_LINK_ANNOTATIONS}" == "1" ]]; then
    link_submission="$(sbatch \
        --parsable \
        --dependency="afterok:${prep_job}" \
        --export=ALL \
        hpc/link_annotations.sbatch)"
    link_job="${link_submission%%;*}"
    controller_dependency="${link_job}"
fi

pipeline_submission="$(sbatch \
    --parsable \
    --dependency="afterok:${controller_dependency}" \
    --partition="${controller_partition}" \
    --export=ALL \
    hpc/run_model_pipeline.sbatch)"
pipeline_job="${pipeline_submission%%;*}"

printf 'Preparation job: %s (nbi-download)\n' "${prep_job}"
if [[ -n "${link_job}" ]]; then
    printf 'PMC provenance job: %s (afterok:%s)\n' "${link_job}" "${prep_job}"
fi
printf 'Pipeline controller: %s (%s, afterok:%s)\n' \
    "${pipeline_job}" "${controller_partition}" "${controller_dependency}"
job_ids="${prep_job}"
if [[ -n "${link_job}" ]]; then
    job_ids="${job_ids},${link_job}"
fi
job_ids="${job_ids},${pipeline_job}"
printf 'Monitor: squeue -j %s\n' "${job_ids}"
printf 'Logs: %s/hpc/logs\n' "${NLP4PHENO_PROJECT}"
