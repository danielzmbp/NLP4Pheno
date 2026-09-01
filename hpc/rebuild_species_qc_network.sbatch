#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-species-network
#SBATCH --partition=nbi-medium
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the code snapshot}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"
: "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR:?Set the validated species-QC directory}"
: "${NLP4PHENO_PMC_PARQUET:?Set the filtered PMC Parquet}"

cd "${NLP4PHENO_PROJECT}"
mkdir -p hpc/logs "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}"

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${NLP4PHENO_ENV_PREFIX}"

accepted="${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/preds_straininfo_species_qc.pqt"
linked="${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/preds_straininfo_species_qc_pmc.pqt"
network="${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/network_species_qc.tsv"
network_evidence="${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/network_species_qc_pmc.tsv"

python scripts/create_relation_network.py \
    "${accepted}" \
    --network-output "${network}" \
    --strains-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/strains_species_qc.txt"

python scripts/link_relation_evidence.py predictions \
    "${accepted}" \
    "${NLP4PHENO_PMC_PARQUET}" \
    --output "${linked}"

python scripts/link_relation_evidence.py network \
    "${network}" \
    "${linked}" \
    --output "${network_evidence}"

python scripts/summarize_network_evidence.py \
    "${network_evidence}" \
    --summary-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/network_species_qc_evidence_summary.tsv" \
    --core-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/network_species_qc_core.tsv" \
    --min-pmcs 2 \
    --min-relation-score 0.90 \
    --min-entity-score 0.90 \
    --min-strain-score 0.95

echo "Species-QC network rebuild completed: ${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}"
