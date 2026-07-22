import pandas as pd
import itertools
from rapidfuzz import process
from rapidfuzz import fuzz
import numpy as np
from collections import defaultdict
import polars as pl
import sys
import os

sys.path.append("scripts")
from relation_prediction_utils import add_formatted_text


configfile: "config.yaml"


cutoff = config["cutoff_prediction"]
output_path = config["output_path"]
preds = f"{output_path}/preds" + str(config["dataset"])
labels = config["rel_labels"]
cuda = config["cuda_devices"]
pmc_file = config["pmc_parquet_file"]
straininfo_designations = config["straininfo_designations_file"]
straininfo_version = config["straininfo_version"]
straininfo_assembly_workers = int(config.get("straininfo_assembly_workers", 8))
straininfo_max_failure_fraction = float(
    config.get("straininfo_max_failure_fraction", 0.01)
)
CPU_PARTITION = config.get("slurm_cpu_partition", "cpu")
GPU_PARTITION = config.get(
    "slurm_gpu_partition", "gpu_h100,gpu_h100_il,gpu_a100_il"
)
DOWNLOAD_PARTITION = config.get("slurm_download_partition", "cpu_il,cpu")
GPU_GRES = config.get("slurm_gpu_gres", "--gres=gpu:1")

# Common resource configurations
COMMON_RESOURCES = {
    "slurm_partition": CPU_PARTITION,
    "runtime": 30,
    "mem_mb": 10000,
}

# Optimized processing - no global caches needed with polars

GPU_RESOURCES = {
    "slurm_partition": GPU_PARTITION,
    "slurm_extra": GPU_GRES,
    "runtime": 600,
    "mem_mb": 24000,
}


rule all:
    input:
        f"{preds}/REL_output/strains_assemblies.txt",
        f"{preds}/network.tsv",
        f"{preds}/network_pmc.tsv",


rule format_sentences:
    input:
        f"{preds}/NER_output/preds.parquet",
    output:
        f"{preds}/NER_output/ner_preds.parquet",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=60,
        mem_mb=24000,
    run:
        df = pl.read_parquet(input[0])
        df = add_formatted_text(df)
        df.write_parquet(output[0], compression="snappy")


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
                python scripts/rel_prediction.py --model $m --device 0 --output {preds}/REL_output/$m.parquet --input {input[0]} 
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
        runtime=80,
        mem_mb=120000,
        tasks=20,
    run:
        df = pd.read_parquet(input[0])
        df = df.drop(columns=["label_rel", "label"])
        df = df.rename(
            columns={
                "score": "ner_score",
            }
        )

        l = []
        for ner in df["ner"].unique():
            df_filter = df[df["ner"] == ner]

            words = df_filter[df_filter["ner"] == ner].word_qc.value_counts()
            query_words = words[words > 1].index
            all_words = words.index
            cutoff = 95

            # Use more workers for faster processing
            result = process.cdist(
                query_words,
                all_words,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=cutoff,
                workers=20,
            )
            indices = np.argwhere(result >= cutoff)
            word_indices = list(zip(query_words[indices[:, 0]], all_words[indices[:, 1]]))
            matchesdf = pd.DataFrame(word_indices)

            scores = result[indices[:, 0], indices[:, 1]]
            matchesdf["score"] = scores
            unique_matches = matchesdf[matchesdf[0] != matchesdf[1]]

            word_counts = df_filter.word_qc.value_counts()
            unique_matches.loc[:, "total_count_0"] = unique_matches[0].map(word_counts)
            unique_matches.loc[:, "total_count_1"] = unique_matches[1].map(word_counts)

            unique_matches.loc[:, "consensus_word"] = unique_matches.apply(
                lambda x: x[0] if x["total_count_0"] > x["total_count_1"] else x[1],
                axis=1,
            )

            # Create a dictionary to group words based on common connections
            grouped_words = defaultdict(list)
            for _, row in unique_matches.iterrows():
                grouped_words[row["consensus_word"]].append(row)

                # Create a dictionary to map each consensus word to all connected words
            consensus_to_words = defaultdict(set)

            # Iterate through each group to check their abundances and select the consensus word
            for group_key, group_values in grouped_words.items():
                # Calculate the total count for each word in the group
                total_counts = {
                    word: sum(
                        unique_matches[unique_matches[0] == word]["total_count_0"]
                    )
                    + sum(unique_matches[unique_matches[1] == word]["total_count_1"])
                    for word in [row[0] for row in group_values]
                    + [row[1] for row in group_values]
                }
                # Select the word with the highest total count as the consensus word
                consensus_word = max(total_counts, key=total_counts.get)

                # Add all words in the group to the set of the consensus word
                for row in group_values:
                    consensus_to_words[consensus_word].update([row[0], row[1]])

            df_filter["word_qc_group"] = df_filter["word_qc"].apply(
                lambda x: next((k for k, v in consensus_to_words.items() if x in v), x)
            )
            l.append(df_filter)

        finaldf = pd.concat(l, ignore_index=True)
        finaldf.to_parquet(output[0], compression="snappy")


rule resolve_straininfo_assemblies:
    input:
        f"{preds}/REL_output/preds_straininfo_grouped.pqt",
    output:
        assemblies=f"{preds}/straininfo/assemblies.parquet",
        manifest=f"{preds}/REL_output/strains_assemblies.txt",
        summary=f"{preds}/straininfo/assembly_summary.json",
    resources:
        slurm_partition=DOWNLOAD_PARTITION,
        runtime=240,
        mem_mb=8000,
        cpus_per_task=straininfo_assembly_workers,
    shell:
        "python scripts/fetch_straininfo_assemblies.py {input} --expected-version {straininfo_version} --workers {straininfo_assembly_workers} --max-failure-fraction {straininfo_max_failure_fraction} --output {output.assemblies} --manifest-output {output.manifest} --summary {output.summary}"


rule link_pmc:
    input:
        f"{preds}/REL_output/preds_straininfo_grouped.pqt",
        pmc_file,
    output:
        f"{preds}/REL_output/preds_straininfo_grouped_pmc.pqt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=80,
        mem_mb=80000,
        tasks=20,
    run:
        df = pl.scan_parquet(input[0])
        pmc = (
            pl.scan_parquet(input[1])
            .select(
                ["text", "pmcid", "article_version", "paragraph", "sentence_range"]
            )
        )

        df_merged = df.join(pmc, on="text", how="left")

        merged_df = df_merged.collect(engine="streaming")

        merged_df.write_parquet(output[0], compression="snappy")


rule create_network:
    input:
        f"{preds}/REL_output/preds_straininfo_grouped.pqt",
    output:
        f"{preds}/network.tsv",
        f"{preds}/strains.txt",
    resources:
        **COMMON_RESOURCES,
    run:
        df = pd.read_parquet(input[0])
        # filter out wrongly assigned strains
        df = df[~df["word_strain_qc"].str.contains("adapted|covid", na=False)]

        matched = df[df["straininfo_si_id"].notna()].copy()
        matched.loc[:, "strain_id"] = matched["straininfo_si_id"].astype(int).map(
            lambda value: f"SI-ID{value}"
        )

        network = (
            matched.loc[:, ["strain_id", "word_qc_group", "rel"]]
            .drop_duplicates(["strain_id", "word_qc_group", "rel"])
        )
        network.loc[:, "source"] = np.where(
            network["rel"].str.startswith("STRAIN"),
            network.strain_id,
            network.word_qc_group,
        )
        network.loc[:, "target"] = np.where(
            network["rel"].str.startswith("STRAIN") == False,
            network.strain_id,
            network.word_qc_group,
        )

        network = network.loc[:, ["source", "target", "rel"]]

        network = pd.concat(
            [
                network,
                network.rel.str.split(":", expand=True)[0]
                .str.split("-", expand=True)
                .rename(columns={0: "source_ner", 1: "target_ner"}),
            ],
            axis=1,
        )

        network["rel"] = network.rel.str.split(":", expand=True)[1]

        network.to_csv(output[0], index=False, sep="\t")

        with open(output[1], "w") as f:
            for s in sorted(set(matched.strain_id.to_list())):
                f.write(f"{s}\n")

rule link_pmc_network:
    input:
        f"{preds}/network.tsv",
        f"{preds}/REL_output/preds_straininfo_grouped_pmc.pqt",
    output:
        f"{preds}/network_pmc.tsv",
    resources:
        **COMMON_RESOURCES,
    run:
        df = pl.read_parquet(input[1])
        network = pl.read_csv(input[0], separator="\t")
        evidence = (
            df.filter(pl.col("straininfo_si_id").is_not_null())
            .with_columns(
                pl.concat_str(
                    pl.lit("SI-ID"),
                    pl.col("straininfo_si_id").cast(pl.Int64).cast(pl.String),
                ).alias("strain_id"),
            )
            .with_columns(
                pl.when(pl.col("rel").str.starts_with("STRAIN"))
                .then(pl.col("strain_id"))
                .otherwise(pl.col("word_qc_group"))
                .alias("source"),
                pl.when(pl.col("rel").str.starts_with("STRAIN"))
                .then(pl.col("word_qc_group"))
                .otherwise(pl.col("strain_id"))
                .alias("target"),
                pl.col("rel").str.split(":").list.get(1).alias("rel_name"),
            )
            .select(
                [
                    "source",
                    "target",
                    "rel_name",
                    "pmcid",
                    "article_version",
                    "paragraph",
                    "sentence_range",
                ]
            )
            .unique()
        )
        network = network.join(
            evidence,
            left_on=["source", "target", "rel"],
            right_on=["source", "target", "rel_name"],
            how="left",
        )
        network.write_csv(output[0], separator="\t")
