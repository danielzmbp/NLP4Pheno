# PMC corpus builder

This Snakemake workflow creates the versioned PMC sentence corpus consumed by
`ner_pred.smk`. It uses PMC's current ESearch and Open Data on AWS services; it
does not use the legacy baseline/incremental FTP archives that are being removed
in August 2026.

## Data flow

1. ESearch discovers all PMCIDs matching the configured reusable-content query.
2. The workflow downloads and pins the latest S3 inventory on or before the
   configured snapshot date.
3. The inventory resolves the newest article version for each PMCID.
4. The article manifest is assigned to stable PMCID-modulo chunks.
5. Each metadata ETag and XML MD5 is verified before versioned JATS XML is
   parsed and filtered in parallel.
6. Validated chunks are streamed into the final Parquet files.

The default query includes Open Access articles published since 1950, excludes
retracted articles, and applies a positive microbial scope. The scope combines
exploded Bacteria/Archaea MeSH terms with bacterial, microbial, prokaryotic,
archaeal and 16S rRNA terms in title, abstract and body fields. Major bacterial
clade terms are limited to title/abstract, preventing an incidental body mention
from admitting an otherwise unrelated paper. These explicit terms cover older
or incompletely indexed articles that say, for example, “cyanobacterium” but
never use a word beginning with “bacter”. It intentionally has no journal-name
blacklist: relevant microbial evidence occurs in medical, chemical,
agricultural and materials journals.

## Configuration

Edit `config.yaml` before starting a release. The important release fields are:

- `source.snapshot_date`: last PMC repository date included by ESearch.
- `source.base_query`: reuse, date and retraction eligibility.
- `source.scope_query`: independently benchmarkable scientific scope. The
  workflow records the fully composed query in its search manifest.
- `source.inventory_version`: `latest` selects the newest report on or before the
  snapshot and records its resolved version. Pin that resolved version to recreate
  the identical inventory later.
- `processing.chunk_count`: number of stable download/parse jobs.
- `processing.max_text_chars`: maximum packed context length. A single sentence
  longer than the limit is retained rather than truncated.
- `processing.excluded_section_patterns`: auditable section-title filters.

Relative paths are resolved from `snakemake_PMC`, independent of the launch
directory.

## Run a local fixture first

From the repository root:

```bash
python -m unittest discover -s snakemake_PMC/tests -p 'test_*.py'
snakemake \
  -s snakemake_PMC/Snakefile \
  --configfile snakemake_PMC/tests/config.fixture.yaml \
  --cores 2
```

The fixture is offline: it verifies version selection, JATS parsing, provenance,
section filtering, language filtering, sentence packing and final schema merging.

## Run the bounded real-data benchmark

Generated benchmark data are ignored by Git. The benchmark uses 100 deterministic
PMCIDs with evidence in the released network and 100 deterministic, recent OA
PMCIDs as a background cohort. The first cohort measures continuity with the old
database; it is not a gold-standard biological label because it contains errors
from the old NER/RE models.

Download the released network and current inventory, then construct the cohort:

```bash
mkdir -p snakemake_PMC/benchmark_data
curl -L \
  'https://zenodo.org/records/17474463/files/network_pmc.tsv?download=1' \
  -o snakemake_PMC/benchmark_data/network_pmc.tsv
snakemake -s snakemake_PMC/Snakefile --cores 1 fetch_inventory
python snakemake_PMC/scripts/build_benchmark_sample.py \
  --network snakemake_PMC/benchmark_data/network_pmc.tsv \
  --inventory-completion snakemake_PMC/temp/inventory/inventory_complete.json \
  --output-ids snakemake_PMC/benchmark_data/smoke_pmcids.txt \
  --output-cohorts snakemake_PMC/benchmark_data/smoke_cohorts.tsv \
  --output-summary snakemake_PMC/benchmark_data/sample_summary.json
```

Evaluate the production scope through PMC ESearch and run the 200-article XML
pipeline:

```bash
python snakemake_PMC/scripts/evaluate_scope_query.py \
  --config snakemake_PMC/config.yaml \
  --cohorts snakemake_PMC/benchmark_data/smoke_cohorts.tsv \
  --decisions snakemake_PMC/benchmark_data/scope_decisions.tsv \
  --summary snakemake_PMC/benchmark_data/scope_summary.json
snakemake -s snakemake_PMC/Snakefile \
  --configfile snakemake_PMC/config.smoke.yaml \
  --cores 8 --rerun-incomplete
```

For the nested 1,000-article benchmark, use separate output names and the same
seed and inventory:

```bash
python snakemake_PMC/scripts/build_benchmark_sample.py \
  --network snakemake_PMC/benchmark_data/network_pmc.tsv \
  --inventory-completion snakemake_PMC/temp/inventory/inventory_complete.json \
  --output-ids snakemake_PMC/benchmark_data/smoke_1000_pmcids.txt \
  --output-cohorts snakemake_PMC/benchmark_data/smoke_1000_cohorts.tsv \
  --output-summary snakemake_PMC/benchmark_data/sample_1000_summary.json \
  --cohort-size 500 --overselect-factor 8 --seed 2509
python snakemake_PMC/scripts/evaluate_scope_query.py \
  --config snakemake_PMC/config.yaml \
  --cohorts snakemake_PMC/benchmark_data/smoke_1000_cohorts.tsv \
  --decisions snakemake_PMC/benchmark_data/scope_1000_decisions.tsv \
  --summary snakemake_PMC/benchmark_data/scope_1000_summary.json
snakemake -s snakemake_PMC/Snakefile \
  --configfile snakemake_PMC/config.smoke.1000.yaml \
  --cores 8 --rerun-incomplete
```

The 1,000-article output and work directories are separate from the 200-article
run, so the two summaries can be compared directly.

Inspect `scope_summary.json`, `scope_decisions.tsv`, `processing_summary.json`
and every TSV in `smoke_output/failures/`. Review scope rejects by title and old
network evidence before broadening the query; a higher carry-over percentage can
simply preserve old model false positives.

## Build a real snapshot

NCBI asks E-utilities clients to identify themselves. Set a contact email; an API
key is optional but increases the permitted request rate:

```bash
export NCBI_EMAIL='name@example.org'
export NCBI_API_KEY='optional-key'
```

Dry-run the production graph:

```bash
snakemake -s snakemake_PMC/Snakefile --use-conda --cores 1 --dry-run
```

Run locally only for a small query:

```bash
snakemake -s snakemake_PMC/Snakefile --use-conda --cores 8
```

For the full corpus, use the bwUni SLURM executor and limit concurrent jobs to a
responsible value for both the cluster and the public S3 service:

```bash
snakemake \
  -s snakemake_PMC/Snakefile \
  --use-conda \
  --executor slurm \
  --jobs 40 \
  --rerun-incomplete
```

The discovery step recursively partitions ESearch by PMC repository date so no
query window exceeds the PMC 10,000-result response limit. XML jobs retry
transient S3 failures. Failure TSVs are always preserved; the merge rejects a
snapshot when its aggregate failure fraction is too high, or when a sufficiently
large individual chunk exceeds the same limit. This avoids unstable percentage
decisions on tiny test chunks without weakening the production threshold.

## Outputs

`output/data/pmc_filtered.parquet` is the downstream corpus:

| Column | Meaning |
|---|---|
| `pmcid` | PMC accession |
| `article_version` | Explicit PMC article version, such as `PMC12855588.1` |
| `section` | Normalized JATS section path |
| `paragraph` | Stable 1-based paragraph number within the parsed article |
| `sentence_range` | Original 1-based sentence number or packed range |
| `text` | Sentence or packed adjacent sentences |

`output/data/pmc_articles.parquet` contains one row per successfully parsed article
with PMID, DOI, citation, title, journal, publication year, language, license,
Open Access/retraction flags, inventory ETag, source XML URL, inclusion status
and exclusion reason.

Other reproducibility artifacts:

- `output/metadata/search_manifest.json`: exact query and date partitions.
- `output/metadata/article_manifest.parquet`: selected article versions and ETags.
- `temp/inventory/inventory_complete.json`: resolved inventory report and hashes.
- `output/stats/processing_summary.json`: complete row/article accounting and
  output schemas.
- `output/failures/*.tsv`: article-level download or parse failures.

Literal section names remain available per corpus row. They are intentionally
not duplicated into a huge JSON histogram in the processing statistics.

The workflow preserves repeated text in different articles. It only avoids XML
duplication by selecting one latest version per PMCID; evidence provenance is not
discarded through global text deduplication.

## Refreshing the database

For a new release, change `snapshot_date`, rerun discovery, and use the new S3
inventory. Article versions and metadata ETags make additions and updates
auditable. Keep the prior `output/metadata` and `output/stats` directories with
each released database so the exact source snapshot can be reconstructed.

## Supported source services

- [PMC Open Data on AWS](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)
- [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
