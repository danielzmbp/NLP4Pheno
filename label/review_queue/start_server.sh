#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

exec "${repo_root}/.envs/label-studio/bin/label-studio" start \
  nlp4pheno-annotation-review \
  --data-dir "${repo_root}/label/label_studio_data" \
  --port 8080 \
  --internal-host 127.0.0.1 \
  --host http://127.0.0.1:8080 \
  --no-browser \
  --enable-legacy-api-token \
  --log-level INFO
