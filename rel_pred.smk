import itertools
import polars as pl
import sys
import os

sys.path.append("scripts")
from relation_prediction_utils import add_formatted_text


configfile: "config.yaml"


HF_HOME = config.get("hf_home")
if HF_HOME:
    os.environ.update(
        {
            "HF_HOME": HF_HOME,
            "HF_DATASETS_CACHE": f"{HF_HOME}/datasets",
            "HF_MODULES_CACHE": f"{HF_HOME}/modules",
            "TRANSFORMERS_CACHE": f"{HF_HOME}/transformers",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )


cutoff = config["cutoff_prediction"]
core_min_pmcs = config.get("core_min_pmcs", 2)
core_min_relation_score = config.get("core_min_relation_score", 0.90)
core_min_entity_score = config.get("core_min_entity_score", 0.90)
core_min_strain_score = config.get("core_min_strain_score", 0.95)
output_path = config["output_path"]
preds = f"{output_path}/preds" + str(config["dataset"])
labels = config["rel_labels"]
cuda = config["cuda_devices"]
pmc_file = config["pmc_parquet_file"]
straininfo_designations = config["straininfo_designations_file"]
straininfo_version = config["straininfo_version"]
ontology_aliases = config.get(
    "ontology_aliases_file",
    "resources/ontologies/runtime/aliases.parquet",
)
ontology_manifest = config.get(
    "ontology_manifest_file",
    "resources/ontologies/runtime/manifest.json",
)
straininfo_assembly_workers = int(config.get("straininfo_assembly_workers", 8))
straininfo_max_failure_fraction = float(
    config.get("straininfo_max_failure_fraction", 0.01)
)
CPU_PARTITION = config.get("slurm_cpu_partition", "cpu")
GPU_PARTITION = config.get(
    "slurm_gpu_partition", "gpu_h100,gpu_h100_il,gpu_a100_il"
)
DOWNLOAD_PARTITION = config.get("slurm_download_partition", "cpu_il,cpu")
GPU_GRES = config.get("slurm_gpu_gres", "gpu:1")
REL_FORMAT_RUNTIME = int(config.get("rel_format_runtime", 120))
REL_FORMAT_MEM_MB = int(config.get("rel_format_mem_mb", 48000))
REL_PREDICTION_RUNTIME = int(config.get("rel_prediction_runtime", 600))
REL_PREDICTION_MEM_MB = int(config.get("rel_prediction_mem_mb", 32000))
REL_ROW_BATCH_SIZE = int(config.get("rel_row_batch_size", 8192))
REL_INFERENCE_BATCH_SIZE = int(config.get("rel_inference_batch_size", 256))
REL_GROUP_RUNTIME = int(config.get("rel_group_runtime", 240))
REL_GROUP_MEM_MB = int(config.get("rel_group_mem_mb", 64000))
REL_GROUP_WORKERS = int(config.get("rel_group_workers", 20))
REL_GROUP_MATRIX_MB = int(config.get("rel_group_matrix_mb", 256))
REL_GROUND_RUNTIME = int(config.get("rel_ground_runtime", 180))
REL_GROUND_MEM_MB = int(config.get("rel_ground_mem_mb", 64000))
REL_SPECIES_QC_RUNTIME = int(config.get("rel_species_qc_runtime", 180))
REL_SPECIES_QC_MEM_MB = int(config.get("rel_species_qc_mem_mb", 64000))
REL_NETWORK_RUNTIME = int(config.get("rel_network_runtime", 60))
REL_NETWORK_MEM_MB = int(config.get("rel_network_mem_mb", 16000))
REL_LINK_RUNTIME = int(config.get("rel_link_runtime", 180))
REL_LINK_MEM_MB = int(config.get("rel_link_mem_mb", 32000))

# Common resource configurations
COMMON_RESOURCES = {
    "slurm_partition": CPU_PARTITION,
    "runtime": 30,
    "mem_mb": 10000,
}

# Optimized processing - no global caches needed with polars

GPU_RESOURCES = {
    "slurm_partition": GPU_PARTITION,
    "gres": GPU_GRES,
    "runtime": REL_PREDICTION_RUNTIME,
    "mem_mb": REL_PREDICTION_MEM_MB,
}


rule all:
    input:
        f"{preds}/REL_output/strains_assemblies.txt",
        f"{preds}/network.tsv",
        f"{preds}/network_pmc.tsv",
        f"{preds}/network_evidence_summary.tsv",
        f"{preds}/network_core.tsv",
        f"{preds}/REL_output/reconciliation_summary.json",
        f"{preds}/REL_output/preds_straininfo_grounded.pqt",
        f"{preds}/REL_output/ontology_groundings.parquet",
        f"{preds}/REL_output/strain_taxonomy_groundings.parquet",
        f"{preds}/REL_output/ontology_grounding_summary.json",
        f"{preds}/REL_output/preds_straininfo_species_qc.pqt",
        f"{preds}/REL_output/species_qc_quarantine.parquet",
        f"{preds}/REL_output/species_qc_audit.parquet",
        f"{preds}/REL_output/species_qc_summary.json",
        f"{preds}/network_ontology.tsv",
        f"{preds}/network_ontology_pmc.tsv",
        f"{preds}/network_ontology_evidence_summary.tsv",
        f"{preds}/network_ontology_core.tsv",
        f"{preds}/network_species_qc.tsv",
        f"{preds}/network_species_qc_pmc.tsv",
        f"{preds}/network_species_qc_evidence_summary.tsv",
        f"{preds}/network_species_qc_core.tsv",


rule format_sentences:
    input:
        f"{preds}/NER_output/preds.parquet",
    output:
        f"{preds}/NER_output/ner_preds.parquet",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_FORMAT_RUNTIME,
        mem_mb=REL_FORMAT_MEM_MB,
    shell:
        """
        python scripts/format_relation_sentences.py \
          {input[0]} \
          --output {output[0]}
        """


rule make_device_file:
    output:
        f"{preds}/REL_output/device_models.txt",
    resources:
        **COMMON_RESOURCES,
    run:
        dev = [str(x) for x in cuda]
        models = [x + " " + y for x, y in zip(itertools.cycle(dev), labels)]
        with open(output[0], "w") as f:
            for i in models:
                f.write(f"{i}\n")


rule run_all_models:
    input:
        f"{preds}/NER_output/ner_preds.parquet",
        f"{preds}/REL_output/device_models.txt",
    output:
        preds + "/REL_output/{l}.parquet",
    conda:
        "envs/pytorch.yml"
    resources:
        **GPU_RESOURCES,
    shell:
        """
        while read -r d m; do
            if [ "$m" = "{wildcards.l}" ]; then
                export CUDA_VISIBLE_DEVICES=$d
                python -c "import torch; print(f'Using GPU: {{torch.cuda.is_available()}}'); print(f'GPU Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}}')"
                python scripts/rel_prediction.py \
                  --model $m \
                  --device 0 \
                  --output {preds}/REL_output/$m.parquet \
                  --input {input[0]} \
                  --row-batch-size {REL_ROW_BATCH_SIZE} \
                  --inference-batch-size {REL_INFERENCE_BATCH_SIZE}
            fi
        done < {input[1]}
        """


rule merge_preds:
    input:
        expand(preds + "/REL_output/{l}.parquet", l=labels),
    output:
        f"{preds}/REL_output/preds.pqt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=90,
        mem_mb=24000,
    run:
        import sys

        sys.path.append("scripts")
        from entity_normalization import (
            normalize_compounds,
            filter_uninterpretable_entities,
            apply_length_filters,
            normalize_strain_entities,
            normalize_entity_column,
        )

        # Use polars for efficient processing - no more caching needed
        # Ensure consistent column ordering before concatenation
        df_list = []
        for file_path in input:
            df_temp = pl.read_parquet(file_path)
            # Select columns in consistent order to avoid column mismatch errors
            df_temp = df_temp.select(sorted(df_temp.columns))
            df_list.append(df_temp)
        df = pl.concat(df_list)

        # Expand re_result JSON column efficiently with polars
        df = df.with_columns(
            [
                pl.col("re_result").struct.field("label").alias("label_rel"),
                pl.col("re_result").struct.field("score").alias("score_rel"),
            ]
        ).drop("re_result")
        # Apply optimized normalizations using the new module
        df = normalize_compounds(df)
        df = apply_length_filters(df)
        df = filter_uninterpretable_entities(df)
        df = normalize_strain_entities(df)
        df = normalize_entity_column(df, "word", "GENERAL")

        # Filter by score threshold and save
        df = df.filter(pl.col("score_rel") > cutoff)
        df.write_parquet(output[0], compression="snappy")


rule match_straininfo:
    input:
        predictions=f"{preds}/REL_output/preds.pqt",
        designations=straininfo_designations,
    output:
        matched=f"{preds}/REL_output/preds_straininfo.pqt",
        summary=f"{preds}/REL_output/straininfo_match_summary.json",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=240,
        mem_mb=32000,
        cpus_per_task=4,
    shell:
        "python scripts/match_straininfo_predictions.py {input.predictions} {input.designations} --output {output.matched} --summary {output.summary}"


rule group_entities:
    input:
        f"{preds}/REL_output/preds_straininfo.pqt",
    output:
        f"{preds}/REL_output/preds_straininfo_grouped.pqt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_GROUP_RUNTIME,
        mem_mb=REL_GROUP_MEM_MB,
        cpus_per_task=REL_GROUP_WORKERS,
    threads: REL_GROUP_WORKERS
    shell:
        """
        python scripts/group_relation_entities.py \
          {input} \
          --output {output} \
          --cutoff 95 \
          --workers {threads} \
          --matrix-mb {REL_GROUP_MATRIX_MB}
        """


rule reconcile_relation_predictions:
    input:
        f"{preds}/REL_output/preds_straininfo_grouped.pqt",
    output:
        reconciled=f"{preds}/REL_output/preds_straininfo_reconciled.pqt",
        summary=f"{preds}/REL_output/reconciliation_summary.json",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_GROUP_RUNTIME,
        mem_mb=REL_GROUP_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/reconcile_relation_predictions.py \
          {input} \
          --output {output.reconciled} \
          --summary {output.summary}
        """


rule ground_ontology:
    input:
        predictions=f"{preds}/REL_output/preds_straininfo_reconciled.pqt",
        aliases=ontology_aliases,
        manifest=ontology_manifest,
    output:
        grounded=f"{preds}/REL_output/preds_straininfo_grounded.pqt",
        mapping=f"{preds}/REL_output/ontology_groundings.parquet",
        strain_mapping=f"{preds}/REL_output/strain_taxonomy_groundings.parquet",
        summary=f"{preds}/REL_output/ontology_grounding_summary.json",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_GROUND_RUNTIME,
        mem_mb=REL_GROUND_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/ground_relation_ontology.py \
          {input.predictions} \
          {input.aliases} \
          {input.manifest} \
          --output {output.grounded} \
          --mapping-output {output.mapping} \
          --strain-taxonomy-mapping-output {output.strain_mapping} \
          --summary {output.summary}
        """


rule qc_species_predictions:
    input:
        predictions=f"{preds}/REL_output/preds_straininfo_grounded.pqt",
        aliases=ontology_aliases,
        manifest=ontology_manifest,
    output:
        accepted=f"{preds}/REL_output/preds_straininfo_species_qc.pqt",
        quarantine=f"{preds}/REL_output/species_qc_quarantine.parquet",
        audit=f"{preds}/REL_output/species_qc_audit.parquet",
        summary=f"{preds}/REL_output/species_qc_summary.json",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_SPECIES_QC_RUNTIME,
        mem_mb=REL_SPECIES_QC_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/qc_species_predictions.py \
          {input.predictions} \
          {input.aliases} \
          {input.manifest} \
          --accepted-output {output.accepted} \
          --quarantine-output {output.quarantine} \
          --audit-output {output.audit} \
          --summary-output {output.summary}
        """


rule resolve_straininfo_assemblies:
    input:
        f"{preds}/REL_output/preds_straininfo_reconciled.pqt",
    output:
        assemblies=f"{preds}/straininfo/assemblies.parquet",
        manifest=f"{preds}/REL_output/strains_assemblies.txt",
        summary=f"{preds}/straininfo/assembly_summary.json",
    resources:
        slurm_partition=DOWNLOAD_PARTITION,
        runtime=240,
        mem_mb=8000,
        cpus_per_task=1,
    shell:
        "python scripts/fetch_straininfo_assemblies.py {input} --expected-version {straininfo_version} --workers {straininfo_assembly_workers} --max-failure-fraction {straininfo_max_failure_fraction} --output {output.assemblies} --manifest-output {output.manifest} --summary {output.summary}"


rule link_pmc:
    input:
        f"{preds}/REL_output/preds_straininfo_reconciled.pqt",
        pmc_file,
    output:
        f"{preds}/REL_output/preds_straininfo_grouped_pmc.pqt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_LINK_RUNTIME,
        mem_mb=REL_LINK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/link_relation_evidence.py predictions \
          {input[0]} \
          {input[1]} \
          --output {output}
        """


rule create_network:
    input:
        f"{preds}/REL_output/preds_straininfo_reconciled.pqt",
    output:
        f"{preds}/network.tsv",
        f"{preds}/strains.txt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_NETWORK_RUNTIME,
        mem_mb=REL_NETWORK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/create_relation_network.py \
          {input} \
          --network-output {output[0]} \
          --strains-output {output[1]}
        """

rule link_pmc_network:
    input:
        f"{preds}/network.tsv",
        f"{preds}/REL_output/preds_straininfo_grouped_pmc.pqt",
    output:
        evidence=f"{preds}/network_pmc.tsv",
        summary=f"{preds}/network_evidence_summary.tsv",
        core=f"{preds}/network_core.tsv",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_LINK_RUNTIME,
        mem_mb=REL_LINK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/link_relation_evidence.py network \
          {input[0]} \
          {input[1]} \
          --output {output.evidence}
        python scripts/summarize_network_evidence.py \
          {output.evidence} \
          --summary-output {output.summary} \
          --core-output {output.core} \
          --min-pmcs {core_min_pmcs} \
          --min-relation-score {core_min_relation_score} \
          --min-entity-score {core_min_entity_score} \
          --min-strain-score {core_min_strain_score}
        """


rule link_ontology_pmc:
    input:
        f"{preds}/REL_output/preds_straininfo_grounded.pqt",
        pmc_file,
    output:
        f"{preds}/REL_output/preds_straininfo_grounded_pmc.pqt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_LINK_RUNTIME,
        mem_mb=REL_LINK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/link_relation_evidence.py predictions \
          {input[0]} \
          {input[1]} \
          --output {output}
        """


rule create_ontology_network:
    input:
        f"{preds}/REL_output/preds_straininfo_grounded.pqt",
    output:
        f"{preds}/network_ontology.tsv",
        f"{preds}/strains_ontology.txt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_NETWORK_RUNTIME,
        mem_mb=REL_NETWORK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/create_relation_network.py \
          {input} \
          --network-output {output[0]} \
          --strains-output {output[1]}
        """


rule link_ontology_pmc_network:
    input:
        f"{preds}/network_ontology.tsv",
        f"{preds}/REL_output/preds_straininfo_grounded_pmc.pqt",
    output:
        evidence=f"{preds}/network_ontology_pmc.tsv",
        summary=f"{preds}/network_ontology_evidence_summary.tsv",
        core=f"{preds}/network_ontology_core.tsv",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_LINK_RUNTIME,
        mem_mb=REL_LINK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/link_relation_evidence.py network \
          {input[0]} \
          {input[1]} \
          --output {output.evidence}
        python scripts/summarize_network_evidence.py \
          {output.evidence} \
          --summary-output {output.summary} \
          --core-output {output.core} \
          --min-pmcs {core_min_pmcs} \
          --min-relation-score {core_min_relation_score} \
          --min-entity-score {core_min_entity_score} \
          --min-strain-score {core_min_strain_score}
        """


rule link_species_qc_pmc:
    input:
        f"{preds}/REL_output/preds_straininfo_species_qc.pqt",
        pmc_file,
    output:
        f"{preds}/REL_output/preds_straininfo_species_qc_pmc.pqt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_LINK_RUNTIME,
        mem_mb=REL_LINK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/link_relation_evidence.py predictions \
          {input[0]} \
          {input[1]} \
          --output {output}
        """


rule create_species_qc_network:
    input:
        f"{preds}/REL_output/preds_straininfo_species_qc.pqt",
    output:
        f"{preds}/network_species_qc.tsv",
        f"{preds}/strains_species_qc.txt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_NETWORK_RUNTIME,
        mem_mb=REL_NETWORK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/create_relation_network.py \
          {input} \
          --network-output {output[0]} \
          --strains-output {output[1]}
        """


rule link_species_qc_pmc_network:
    input:
        f"{preds}/network_species_qc.tsv",
        f"{preds}/REL_output/preds_straininfo_species_qc_pmc.pqt",
    output:
        evidence=f"{preds}/network_species_qc_pmc.tsv",
        summary=f"{preds}/network_species_qc_evidence_summary.tsv",
        core=f"{preds}/network_species_qc_core.tsv",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=REL_LINK_RUNTIME,
        mem_mb=REL_LINK_MEM_MB,
        cpus_per_task=4,
    threads: 4
    shell:
        """
        python scripts/link_relation_evidence.py network \
          {input[0]} \
          {input[1]} \
          --output {output.evidence}
        python scripts/summarize_network_evidence.py \
          {output.evidence} \
          --summary-output {output.summary} \
          --core-output {output.core} \
          --min-pmcs {core_min_pmcs} \
          --min-relation-score {core_min_relation_score} \
          --min-entity-score {core_min_entity_score} \
          --min-strain-score {core_min_strain_score}
        """
