#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-species-qc
#SBATCH --partition=nbi-medium
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the code snapshot}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"
: "${NLP4PHENO_SPECIES_QC_INPUT:?Set NLP4PHENO_SPECIES_QC_INPUT to grounded predictions}"
: "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR:?Set NLP4PHENO_SPECIES_QC_OUTPUT_DIR}"

NLP4PHENO_ONTOLOGY_ALIASES="${NLP4PHENO_ONTOLOGY_ALIASES:-${NLP4PHENO_PROJECT}/resources/ontologies/runtime/aliases.parquet}"
NLP4PHENO_ONTOLOGY_MANIFEST="${NLP4PHENO_ONTOLOGY_MANIFEST:-${NLP4PHENO_PROJECT}/resources/ontologies/runtime/manifest.json}"

for required in \
    "${NLP4PHENO_PROJECT}/scripts/qc_species_predictions.py" \
    "${NLP4PHENO_SPECIES_QC_INPUT}" \
    "${NLP4PHENO_ONTOLOGY_ALIASES}" \
    "${NLP4PHENO_ONTOLOGY_MANIFEST}"; do
    if [[ ! -s "${required}" ]]; then
        echo "Required species-QC input is missing or empty: ${required}" >&2
        exit 1
    fi
done

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${NLP4PHENO_ENV_PREFIX}"

mkdir -p "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}"
cd "${NLP4PHENO_PROJECT}"

python scripts/qc_species_predictions.py \
    "${NLP4PHENO_SPECIES_QC_INPUT}" \
    "${NLP4PHENO_ONTOLOGY_ALIASES}" \
    "${NLP4PHENO_ONTOLOGY_MANIFEST}" \
    --accepted-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/preds_straininfo_species_qc.pqt" \
    --quarantine-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/species_qc_quarantine.parquet" \
    --audit-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/species_qc_audit.parquet" \
    --summary-output "${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}/species_qc_summary.json"

echo "Species QC completed: ${NLP4PHENO_SPECIES_QC_OUTPUT_DIR}"
