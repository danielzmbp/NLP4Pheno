#!/usr/bin/env bash
#SBATCH --job-name=nlp4pheno-pmc-links
#SBATCH --partition=nbi-medium
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

: "${NLP4PHENO_PROJECT:?Set NLP4PHENO_PROJECT to the shared workflow directory}"
: "${NLP4PHENO_CORPUS:?Set NLP4PHENO_CORPUS to pmc_filtered.parquet}"
: "${NLP4PHENO_ENV_PREFIX:?Set NLP4PHENO_ENV_PREFIX to the shared Conda prefix}"

cd "${NLP4PHENO_PROJECT}"

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${NLP4PHENO_ENV_PREFIX}"

readarray -t annotation_paths < <(
    python - <<'PY'
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config.yaml").read_text())
for key in ("input_file", "annotation_pmc_matches_file"):
    value = config.get(key)
    if not value:
        raise SystemExit(f"config.yaml must define {key}")
    print(value)
PY
)

annotations="${annotation_paths[0]}"
matches="${annotation_paths[1]}"
summary="${matches%.parquet}.summary.json"

if [[ ! -s "${annotations}" ]]; then
    echo "Annotation input is missing or empty: ${annotations}" >&2
    exit 1
fi
if [[ ! -s "${NLP4PHENO_CORPUS}" ]]; then
    echo "PMC corpus is missing or empty: ${NLP4PHENO_CORPUS}" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${matches}")"
python scripts/link_annotations_to_pmc.py \
    "${annotations}" \
    "${NLP4PHENO_CORPUS}" \
    --matches-output "${matches}" \
    --summary-output "${summary}"

python - "${matches}" "${summary}" <<'PY'
import json
import sys

import pyarrow.parquet as pq

matches, summary = sys.argv[1:]
metadata = pq.read_metadata(matches)
report = json.load(open(summary))
if metadata.num_rows == 0:
    raise SystemExit("PMC provenance matching produced no rows")
if report["tasks"]["unique_pmcid"] == 0:
    raise SystemExit("PMC provenance matching produced no unique PMCIDs")
print(
    f"Validated {metadata.num_rows:,} provenance rows; "
    f"{report['tasks']['unique_pmcid']:,} tasks have a unique PMCID"
)
PY
