#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-network-review
#SBATCH --partition=nbi-medium
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the code snapshot}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"
: "${NLP4PHENO_OLD_PREDICTIONS:?Set the preceding linked prediction Parquet}"
: "${NLP4PHENO_NEW_PREDICTIONS:?Set the current linked prediction Parquet}"
: "${NLP4PHENO_REVIEW_ANNOTATIONS:?Set the current reviewed annotation JSON}"
: "${NLP4PHENO_REVIEW_OUTPUT:?Set a new queue output directory}"

review_relations="${NLP4PHENO_REVIEW_RELATIONS:-STRAIN-ORGANISM:INFECTS STRAIN-ORGANISM:INHABITS}"
per_direction="${NLP4PHENO_REVIEW_PER_DIRECTION:-10}"

cd "${NLP4PHENO_PROJECT}"
mkdir -p hpc/logs "${NLP4PHENO_REVIEW_OUTPUT}"

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${NLP4PHENO_ENV_PREFIX}"

args=(
    "${NLP4PHENO_OLD_PREDICTIONS}"
    "${NLP4PHENO_NEW_PREDICTIONS}"
    "${NLP4PHENO_REVIEW_ANNOTATIONS}"
    --queue-output "${NLP4PHENO_REVIEW_OUTPUT}/queue.json"
    --issues-output "${NLP4PHENO_REVIEW_OUTPUT}/issues.tsv"
    --summary-output "${NLP4PHENO_REVIEW_OUTPUT}/summary.json"
    --per-relation-direction "${per_direction}"
)
for relation in ${review_relations}; do
    args+=(--include-relation "${relation}")
done

python -u scripts/build_network_disagreement_queue.py "${args[@]}"
echo "Network disagreement review queue completed"
