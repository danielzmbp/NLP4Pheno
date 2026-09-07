#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-rel-rescore
#SBATCH --partition=ei-gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the code snapshot}"
: "${NLP4PHENO_BASELINE_RUN:?Set NLP4PHENO_BASELINE_RUN to the old model run}"
: "${NLP4PHENO_CORRECTED_RUN:?Set NLP4PHENO_CORRECTED_RUN to the corrected-data run}"
: "${NLP4PHENO_RESCORE_OUTPUT:?Set NLP4PHENO_RESCORE_OUTPUT to a new output directory}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"
: "${NLP4PHENO_HF_HOME:?Set NLP4PHENO_HF_HOME to the shared Hugging Face cache}"

relations="${NLP4PHENO_RESCORE_RELATIONS:-STRAIN-ORGANISM:INHABITS STRAIN-SPECIES:INHIBITS}"

cd "${NLP4PHENO_PROJECT}"
mkdir -p hpc/logs "${NLP4PHENO_RESCORE_OUTPUT}"

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
export USE_CODALAB=1

python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

for relation in ${relations}; do
    model_dir="${NLP4PHENO_BASELINE_RUN}/REL_output/${relation}"
    data_dir="${NLP4PHENO_CORRECTED_RUN}/REL/${relation}"
    output_dir="${NLP4PHENO_RESCORE_OUTPUT}/${relation}"
    for required in \
        "${model_dir}/model.safetensors" \
        "${data_dir}/dev.json" \
        "${data_dir}/test.json"; do
        if [[ ! -s "${required}" ]]; then
            echo "Required re-scoring input is missing: ${required}" >&2
            exit 1
        fi
    done
    mkdir -p "${output_dir}"
    echo "===== Re-scoring ${relation} at $(date --iso-8601=seconds) ====="
    python3 -u scripts/run_seqcls.py \
        --model_name_or_path "${model_dir}" \
        --validation_file "${data_dir}/dev.json" \
        --test_file "${data_dir}/test.json" \
        --do_eval \
        --do_predict \
        --report_to none \
        --metric_name PRF1 \
        --per_device_eval_batch_size 64 \
        --fp16 \
        --max_seq_length 512 \
        --output_dir "${output_dir}" \
        --overwrite_output_dir \
        2>&1 | tee "${output_dir}/log.txt"
done

echo "Corrected-holdout re-scoring completed"
