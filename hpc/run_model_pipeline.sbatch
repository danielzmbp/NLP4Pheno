#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-models
#SBATCH --partition=nbi-long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the shared repository path}"
: "${NLP4PHENO_CORPUS:?Set NLP4PHENO_CORPUS to pmc_filtered.parquet}"
: "${NLP4PHENO_OUTPUT_PATH:?Set NLP4PHENO_OUTPUT_PATH to a shared output directory}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"
: "${NLP4PHENO_HF_HOME:?Set NLP4PHENO_HF_HOME to a shared cache directory}"
: "${NLP4PHENO_MODEL_DIR:?Set NLP4PHENO_MODEL_DIR to the local model directory}"

NLP4PHENO_JOBS="${NLP4PHENO_JOBS:-48}"
NLP4PHENO_STAGES="${NLP4PHENO_STAGES:-ner rel ner_pred rel_pred}"

cd "${NLP4PHENO_PROJECT}"

for required in \
    "${NLP4PHENO_CORPUS}" \
    "${NLP4PHENO_MODEL_DIR}/config.json" \
    resources/straininfo/designations.parquet; do
    if [[ ! -s "${required}" ]]; then
        echo "Required offline input is missing or empty: ${required}" >&2
        exit 1
    fi
done

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${NLP4PHENO_ENV_PREFIX}"

export HF_HOME="${NLP4PHENO_HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=true

mkdir -p "${NLP4PHENO_OUTPUT_PATH}"

common_args=(
    --configfile hpc/config.nbi.yaml
    --executor slurm
    --slurm-no-account
    --jobs "${NLP4PHENO_JOBS}"
    --default-resources
    "slurm_partition=nbi-medium"
    "runtime=60"
    "mem_mb=8000"
    --latency-wait 300
    --rerun-incomplete
    --printshellcmds
    --show-failed-logs
    --config
    "output_path=${NLP4PHENO_OUTPUT_PATH}"
    "pmc_parquet_file=${NLP4PHENO_CORPUS}"
    "pretrained_model_path=${NLP4PHENO_MODEL_DIR}"
    "straininfo_strains_file=${NLP4PHENO_PROJECT}/resources/straininfo/strains.parquet"
    "straininfo_designations_file=${NLP4PHENO_PROJECT}/resources/straininfo/designations.parquet"
)

if [[ "${NLP4PHENO_DRY_RUN:-0}" == "1" ]]; then
    common_args+=(--dry-run)
fi

run_stage() {
    local stage="$1"
    local snakefile
    case "${stage}" in
        ner) snakefile="ner.smk" ;;
        rel) snakefile="rel.smk" ;;
        ner_pred) snakefile="ner_pred.smk" ;;
        rel_pred) snakefile="rel_pred.smk" ;;
        *)
            echo "Unknown NLP4PHENO_STAGES entry: ${stage}" >&2
            return 2
            ;;
    esac
    echo "===== Starting ${stage} with ${snakefile} at $(date --iso-8601=seconds) ====="
    snakemake --snakefile "${snakefile}" "${common_args[@]}"
    echo "===== Completed ${stage} at $(date --iso-8601=seconds) ====="
}

for stage in ${NLP4PHENO_STAGES}; do
    run_stage "${stage}"
done

echo "All requested NLP4Pheno model stages completed"
