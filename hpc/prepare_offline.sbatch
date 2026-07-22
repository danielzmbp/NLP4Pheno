#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-prepare
#SBATCH --partition=nbi-download
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the shared repository path}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"
: "${NLP4PHENO_HF_HOME:?Set NLP4PHENO_HF_HOME to a shared cache directory}"
: "${NLP4PHENO_MODEL_DIR:?Set NLP4PHENO_MODEL_DIR to the local model directory}"

NLP4PHENO_MODEL_ID="${NLP4PHENO_MODEL_ID:-michiyasunaga/BioLinkBERT-large}"
NLP4PHENO_STRAININFO_VERSION="${NLP4PHENO_STRAININFO_VERSION:-2025.10}"

cd "${NLP4PHENO_PROJECT}"

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"

if [[ ! -x "${NLP4PHENO_ENV_PREFIX}/bin/python" ]]; then
    echo "Creating the shared HPC environment at ${NLP4PHENO_ENV_PREFIX}"
    mamba env create \
        --yes \
        --prefix "${NLP4PHENO_ENV_PREFIX}" \
        --file hpc/environment.yml
elif [[ "${NLP4PHENO_SKIP_ENV_UPDATE:-0}" != "1" ]]; then
    echo "Updating the shared HPC environment"
    mamba env update \
        --yes \
        --prefix "${NLP4PHENO_ENV_PREFIX}" \
        --file hpc/environment.yml \
        --prune
fi

conda activate "${NLP4PHENO_ENV_PREFIX}"

export HF_HOME="${NLP4PHENO_HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p \
    "${HF_HOME}" \
    "${NLP4PHENO_MODEL_DIR}" \
    resources/straininfo

export NLP4PHENO_MODEL_ID NLP4PHENO_MODEL_DIR
python - <<'PY'
import os
from pathlib import Path

import evaluate
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

model_id = os.environ["NLP4PHENO_MODEL_ID"]
model_dir = Path(os.environ["NLP4PHENO_MODEL_DIR"])
print(f"Caching {model_id} in {model_dir}")
snapshot_download(repo_id=model_id, local_dir=model_dir)

# Both training programs call evaluate.load at runtime. Cache the metric
# modules now because the A100 nodes cannot contact the Hub.
evaluate.load("seqeval")
evaluate.load("accuracy")

AutoConfig.from_pretrained(model_dir, local_files_only=True)
AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
if not weights:
    raise RuntimeError(f"No model weights were downloaded to {model_dir}")
print(f"Cached model files: {len(list(model_dir.iterdir()))}")
PY

if [[ ! -s resources/straininfo/strains.parquet \
      || ! -s resources/straininfo/designations.parquet \
      || "${NLP4PHENO_REFRESH_STRAININFO:-0}" == "1" ]]; then
    python scripts/fetch_straininfo_catalog.py \
        --expected-version "${NLP4PHENO_STRAININFO_VERSION}" \
        --strains-output resources/straininfo/strains.parquet \
        --designations-output resources/straininfo/designations.parquet \
        --summary-output resources/straininfo/runtime_summary.json
fi

# Prove that the assets needed by offline GPU jobs resolve without network.
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python - <<'PY'
import os

import evaluate
from transformers import AutoConfig, AutoTokenizer

path = os.environ["NLP4PHENO_MODEL_DIR"]
AutoConfig.from_pretrained(path, local_files_only=True)
AutoTokenizer.from_pretrained(path, local_files_only=True)
evaluate.load("seqeval")
evaluate.load("accuracy")
print("Offline model, tokenizer, and metric checks passed")
PY

python - <<'PY'
import pyarrow.parquet as pq

for path in (
    "resources/straininfo/strains.parquet",
    "resources/straininfo/designations.parquet",
):
    metadata = pq.read_metadata(path)
    if metadata.num_rows == 0:
        raise RuntimeError(f"Empty StrainInfo snapshot: {path}")
    print(f"{path}: {metadata.num_rows:,} rows")
PY

echo "Internet-dependent preparation completed successfully"
