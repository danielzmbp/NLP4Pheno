# Ontology resources

`sources.yaml` defines the external terminology sources used by the grounding
pilot. Run the build from the repository root:

```bash
python scripts/build_ontology_index.py
```

This creates the ignored `runtime/` directory:

- `raw/`: exact downloaded source files;
- `terms.parquet`: one row per supported concept/entity-type pair;
- `aliases.parquet`: labels, synonyms, normalized lookup keys, and eligibility;
- `manifest.json`: resolved versions, hashes, sizes, and row counts.

Existing raw files are reused. Pass `--refresh` only when intentionally taking a
new snapshot. Keep the resulting manifest with any network release so the
grounding can be reproduced.

The automatic tier accepts only unique, source-supported evidence. Broad,
narrow, and related synonyms are excluded by default. ChEBI is the one
exception: case-preserving formula-shaped related aliases are allowed because
formulas such as `H2O2` and `NaCl` are commonly emitted with PMC/XML subscript
artifacts. Bare acronyms are not accepted by that exception.

MediaDive does not expose a synonym list in its media endpoint, so the builder
derives only aliases stated directly by a canonical resource label, such as
`MRS MEDIUM` → `MRS` and `LB (Luria-Bertani) MEDIUM` → `LB` and
`Luria-Bertani`. If an alias names several records it remains ambiguous.

Run the reviewed-annotation pilot with:

```bash
python scripts/ground_ontology_annotations.py \
  label/project-10-reviewed-2026-07-22.json
```

The detailed result is
`runtime/pilot/groundings.parquet`; the JSON summary stays under `runtime/` and
the human-readable benchmark is written to
`label/ontology_grounding_pilot.md`.
