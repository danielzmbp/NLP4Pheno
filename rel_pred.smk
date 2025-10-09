import pandas as pd
import itertools
from rapidfuzz import process
from rapidfuzz import fuzz
from scipy import sparse
from tqdm.auto import tqdm
import dask.array as da
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from glob import glob
import polars as pl


configfile: "config.yaml"


cutoff = config["cutoff_prediction"]
output_path = config["output_path"]
preds = f"{output_path}/preds" + str(config["dataset"])
labels = config["rel_labels"]
cuda = config["cuda_devices"]
pmc_file = config["pmc_parquet_file"]

# Common resource configurations
COMMON_RESOURCES = {"slurm_partition": "cpu", "runtime": 30, "mem_mb": 10000}

# Optimized processing - no global caches needed with polars

GPU_RESOURCES = {
    "slurm_partition": "gpu_h100,gpu_h100_il,gpu_a100_il",
    "slurm_extra": "--gres=gpu:1",
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
        slurm_partition="cpu",
        runtime=60,
        mem_mb=24000,
    run:
        df = pl.read_parquet(input[0])

        # Optimized text formatting using polars expressions
        df = df.with_columns(
            [
                pl.when(
                    (pl.col("start_strain") > pl.col("start"))
                    & (pl.col("end_strain") > pl.col("end"))
                )
                .then(
                    pl.col("text").str.slice(0, pl.col("start_strain"))
                    + "@STRAIN$"
                    + pl.col("text").str.slice(pl.col("end_strain"))
                    + pl.col("text").str.slice(0, pl.col("start"))
                    + "@"
                    + pl.col("ner")
                    + "$"
                    + pl.col("text").str.slice(pl.col("end"))
                )
                .when(
                    (pl.col("start_strain") <= pl.col("start"))
                    & (pl.col("end") > pl.col("end_strain"))
                )
                .then(
                    pl.col("text").str.slice(0, pl.col("start"))
                    + "@"
                    + pl.col("ner")
                    + "$"
                    + pl.col("text").str.slice(pl.col("end"))
                    + pl.col("text").str.slice(0, pl.col("start_strain"))
                    + "@STRAIN$"
                    + pl.col("text").str.slice(pl.col("end_strain"))
                )
                .otherwise(None)
                .alias("formatted_text")
            ]
        )

        # Remove rows where formatting failed
        df = df.filter(pl.col("formatted_text").is_not_null())
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
        slurm_partition="cpu",
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


rule download_strainselect:
    output:
        f"{preds}/strainselect/StrainSelect21_edges.tab.txt",
        f"{preds}/strainselect/StrainSelect21_vertices.tab.txt",
    resources:
        **COMMON_RESOURCES,
    shell:
        "wget https://gg-sg-web.s3-us-west-2.amazonaws.com/downloads/strainselect_database/StrainSelect21/StrainSelect21_edges.tab.txt -O {output[0]}; wget https://gg-sg-web.s3-us-west-2.amazonaws.com/downloads/strainselect_database/StrainSelect21/StrainSelect21_vertices.tab.txt -O {output[1]}"


rule split_batches_strainselect:
    input:
        preds_file=f"{preds}/REL_output/preds.pqt",
    output:
        batch_files=expand(
            f"{preds}/REL_output/batched_input/{{batch_id}}.pqt",
            batch_id=range(0, 1000),
        ),
    resources:
        slurm_partition="cpu",
        runtime=50,
        mem_mb=10000,
        tasks=2,
    run:
        import sys

        sys.path.append("scripts")
        from entity_normalization import create_vertex_dot_column, extract_genus_hint

        # Use polars for efficient processing
        df = pl.read_parquet(input[0])

        # Add genus hints from sentence context (CRITICAL for improved matching)
        df = extract_genus_hint(df)

        df = create_vertex_dot_column(df)

        # Filter out short matches that are likely noise
        df = df.filter(~pl.col("vertex_dot").str.contains("^[a-z]\\.[a-z]+$"))

        num_batches = 1000
        batch_size = len(df) // num_batches + 1

        os.makedirs(f"{preds}/REL_output/batched_input/", exist_ok=True)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_df = df.slice(start_idx, end_idx - start_idx)
            batch_df.write_parquet(f"{preds}/REL_output/batched_input/{i}.pqt")


rule match_batch_strainselect:
    input:
        batch_file=f"{preds}/REL_output/batched_input/{{batch_id}}.pqt",
        vertices=f"{preds}/strainselect/StrainSelect21_vertices.tab.txt",
    output:
        batch_output=f"{preds}/batched_output_results/{{batch_id}}.parquet",
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=1000,
        mem_mb=10000,
    threads: 8
    run:
        import sys
        sys.path.append("scripts")
        from improved_strain_matching import match_strains_improved, filter_ambiguous_matches
        from rapidfuzz import process, fuzz

        workers = threads

        batch_id = wildcards.batch_id
        print(f"[BATCH {batch_id}] Starting HYBRID strain matching process...")

        # Load batch data (now includes genus_hint column)
        df = pd.read_parquet(input[0])

        # Load vertices database
        vertices = pd.read_csv(input[1], sep="\t", usecols=[0, 1, 2], low_memory=False)
        vertices["vertex_dot"] = vertices.vertex.str.replace("_", ".").str.lower()

        # Clean strain names
        strains_series = (
            df.vertex_dot.str.replace("\.$", "", regex=True)
            .str.replace("^\.", "", regex=True)
        )

        # Filter out very short strains (likely noise)
        mask = strains_series.str.replace(".", "", regex=False).str.len() > 2
        strains_series = strains_series[mask]

        # Get genus hints aligned with strains
        genus_hints_series = df.loc[strains_series.index, 'genus_hint'] if 'genus_hint' in df.columns else None

        # Get unique strains (but keep genus hints)
        strain_genus_df = pd.DataFrame({
            'strain': strains_series,
            'genus_hint': genus_hints_series if genus_hints_series is not None else None
        }).drop_duplicates(subset=['strain'])

        print(f"[BATCH {batch_id}] Data loaded: {len(df)} rows, {len(strain_genus_df)} unique strains")
        if genus_hints_series is not None:
            n_with_genus = strain_genus_df['genus_hint'].notna().sum()
            print(f"[BATCH {batch_id}] Genus hints: {n_with_genus} / {len(strain_genus_df)} ({100*n_with_genus/len(strain_genus_df):.1f}%)")

        # STEP 1: Try NEW (improved) matching algorithm
        print(f"[BATCH {batch_id}] STEP 1/2: Running improved context-aware matching...")
        new_matches = match_strains_improved(
            strains=strain_genus_df['strain'],
            vertices=vertices,
            genus_hints=strain_genus_df['genus_hint'] if 'genus_hint' in strain_genus_df.columns else None,
            workers=workers,
            verbose=False
        )
        new_matches = filter_ambiguous_matches(new_matches, score_threshold=5.0)

        # STEP 2: Fallback to OLD strategy for unmatched strains
        matched_by_new = set(new_matches['strain'].tolist()) if len(new_matches) > 0 else set()
        unmatched_strains = [s for s in strain_genus_df['strain'] if s not in matched_by_new]

        print(f"[BATCH {batch_id}] NEW matched: {len(matched_by_new)}/{len(strain_genus_df)}")
        print(f"[BATCH {batch_id}] STEP 2/2: Running OLD strategy on {len(unmatched_strains)} unmatched strains...")

        if len(unmatched_strains) > 0:
            # OLD strategy: simple fuzzy matching without genus filtering
            vertices_noass = vertices[vertices.vertex_type == 'gss'].copy()
            vertices_list = vertices_noass.vertex_dot.to_list()

            partial_matches = process.cdist(
                unmatched_strains,
                vertices_list,
                scorer=fuzz.partial_ratio,
                workers=workers,
                score_cutoff=90
            )

            old_matches_list = []
            for i, strain in enumerate(unmatched_strains):
                strain_scores = partial_matches[i]
                if strain_scores.max() > 0:
                    best_idx = strain_scores.argmax()
                    best_match = vertices_list[best_idx]
                    best_score = strain_scores[best_idx]

                    old_matches_list.append({
                        'strain': strain,
                        'strainselect': best_match,
                        'score_weighted': best_score,
                        'score_partial': best_score
                    })

            if len(old_matches_list) > 0:
                old_fallback = pd.DataFrame(old_matches_list)
                old_fallback = old_fallback.merge(
                    vertices_noass[['vertex_dot', 'vertex', 'vertex_type', 'StrainSelectID']],
                    left_on='strainselect',
                    right_on='vertex_dot',
                    how='left'
                )
                # Combine NEW + OLD
                matches = pd.concat([new_matches, old_fallback], ignore_index=True)
                print(f"[BATCH {batch_id}] OLD matched: {len(old_fallback)} additional strains")
            else:
                matches = new_matches
                print(f"[BATCH {batch_id}] OLD matched: 0 additional strains")
        else:
            matches = new_matches

        print(f"[BATCH {batch_id}] HYBRID TOTAL: {len(matches)}/{len(strain_genus_df)} unique strains matched")

        # Merge matches back to original df
        # Clean vertex_dot in df for matching
        df_clean = df.copy()
        df_clean['vertex_dot_clean'] = (
            df_clean.vertex_dot.str.replace("\.$", "", regex=True)
            .str.replace("^\.", "", regex=True)
        )

        # Drop score_partial from matches to avoid duplicate after rename
        if 'score_partial' in matches.columns:
            matches = matches.drop(columns=['score_partial'])

        final = df_clean.merge(
            matches,
            left_on='vertex_dot_clean',
            right_on='strain',
            how='left'
        )

        # Rename columns to match expected output format
        final = final.drop(columns=['vertex_dot_clean', 'vertex_dot_y', 'strainselect', 'strain'], errors='ignore')
        if 'vertex_dot_x' in final.columns:
            final = final.rename(columns={'vertex_dot_x': 'vertex_dot'})

        final = final.rename(
            columns={
                "vertex": "strainselect_vertex",
                "score_weighted": "score_partial"
            },
            errors='ignore'
        )

        # Add score_full as NaN (not used in improved matching)
        if 'score_full' not in final.columns:
            final['score_full'] = np.nan

        # Final statistics
        matched_rows = final['StrainSelectID'].notna().sum()
        print(f"[BATCH {batch_id}] FINAL RESULTS: {matched_rows}/{len(final)} rows have strain matches ({100*matched_rows/len(final):.1f}%)")

        final.to_parquet(output[0])


rule merge_batch_outputs_strainselect:
    input:
        batch_outputs=expand(
            f"{preds}/batched_output_results/{{batch_id}}.parquet",
            batch_id=range(0, 1000),
        ),
    output:
        merged_output=f"{preds}/REL_output/preds_strainselect.pqt",
    resources:
        slurm_partition="cpu",
        runtime=600,
        mem_mb=16000,
    run:
        # Efficient batch processing with polars
        df_list = []
        for file_path in input:
            df_temp = pl.read_parquet(file_path)
            df_list.append(df_temp)
        df = pl.concat(df_list)
        df.write_parquet(output[0], compression="snappy")


rule group_entities:
    input:
        f"{preds}/REL_output/preds_strainselect.pqt",
    output:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
    resources:
        slurm_partition="cpu",
        runtime=80,
        mem_mb=120000,
        tasks=20,
    run:
        df = pd.read_parquet(input[0])
        df = df.drop(columns=["label_rel", "label"])
        df.rename(
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
            word_indices = list(zip(all_words[indices[:, 0]], all_words[indices[:, 1]]))
            matchesdf = pd.DataFrame(word_indices)

            scores = result[indices[:, 0], indices[:, 1]]
            matchesdf["score"] = scores
            unique_matches = matchesdf[matchesdf[0] != matchesdf[1]]

            word_counts = df.word_qc.value_counts()
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


rule write_download_file:
    input:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
        f"{preds}/strainselect/StrainSelect21_vertices.tab.txt",
    output:
        f"{preds}/REL_output/strains_assemblies.txt",
    resources:
        **{**COMMON_RESOURCES, "runtime": 60},
    run:
        import polars as pl
        
        # Read parquet file and filter out nulls
        df = pl.read_parquet(input[0]).filter(pl.col("StrainSelectID").is_not_null())
        
        # Read vertices file and filter for rs_assembly only during reading
        v_rs = (
            pl.read_csv(input[1], separator="\t", n_rows=None)
            .select(["StrainSelectID", "vertex", "vertex_type"])
            .filter(pl.col("vertex_type") == "rs_assembly")
        )
        
        # Perform join and create assemblies column
        merged = df.join(v_rs, on="StrainSelectID", how="inner")
        merged = merged.with_columns(
            (pl.col("StrainSelectID") + "/" + pl.col("vertex").str.replace(" ", "_")).alias("assemblies")
        )
        
        # Get unique assemblies
        assemblies = merged.select("assemblies").unique().to_series().to_list()
        
        with open(output[0], "w") as f:
            for a in assemblies:
                f.write(a + "\n")


rule link_pmc:
    input:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
        pmc_file,
    output:
        f"{preds}/REL_output/preds_strainselect_grouped_pmc.pqt",
    resources:
        slurm_partition="cpu",
        runtime=80,
        mem_mb=80000,
        tasks=20,
    run:
        df = pl.scan_parquet(input[0])
        pmc = (
            pl.scan_parquet(input[1])
            .select(["text", "pmcid", "paragraph", "sentence_range"])
            .with_columns(pl.col("text").str.replace_all(r"(\w)-(\w)", r"$1 $2"))
            .with_columns(pl.col("text").str.replace_all(r"(\w)-(\w)", r"$1 $2"))
        )

        df_merged = df.join(pmc, on="text", how="left")

        merged_df = df_merged.collect(engine="streaming")

        merged_df.write_parquet(output[0], compression="snappy")


rule create_network:
    input:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
    output:
        f"{preds}/network.tsv",
        f"{preds}/strains.txt",
        # f"{preds}/network_assemblies.tsv",
        # f"{preds}/strains_assemblies.txt",
    resources:
        **COMMON_RESOURCES,
    params:
        data=str(config["dataset"]),
    run:
        df = pd.read_parquet(input[0])
        # filter out wrongly assigned strains
        df = df[~df["word_strain_qc"].str.contains("adapted|covid")]

        network = (
            df[df.StrainSelectID.isna() == False]
            .loc[:, ["StrainSelectID", "word_qc_group", "rel"]]
            .drop_duplicates(["StrainSelectID", "word_qc_group", "rel"])
        )
        network.loc[:, "source"] = np.where(
            network["rel"].str.startswith("STRAIN"),
            network.StrainSelectID,
            network.word_qc_group,
        )
        network.loc[:, "target"] = np.where(
            network["rel"].str.startswith("STRAIN") == False,
            network.StrainSelectID,
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
            for s in sorted(
                set(df[df.StrainSelectID.isna() == False].StrainSelectID.to_list())
            ):
                f.write(f"{s}\n")

                # # filtered network

                # folders = glob(f"{output_path}/assemblies_{params.data}/*")

                # strains = [f.split("/")[-1] for f in folders]
                # filtered_df= df[df.StrainSelectID.isin(strains)]

                # network = filtered_df.loc[:,["StrainSelectID","word_qc_group","rel"]].drop_duplicates(["StrainSelectID","word_qc_group","rel"])
                # network.loc[:,"source"] = np.where(network['rel'].str.startswith("STRAIN"), network.StrainSelectID	, network.word_qc_group)
                # network.loc[:,"target"] = np.where(network['rel'].str.startswith("STRAIN")==False, network.StrainSelectID, network.word_qc_group)

                # network = network.loc[:,["source","target","rel"]]

                # network = pd.concat([network,
                # network.rel.str.split(":",expand=True)[0].str.split("-",expand=True).rename(
                # columns={0:"source_ner",1:"target_ner"})]
                #     ,axis=1
                #     )

                # network["rel"] = network.rel.str.split(":",expand=True)[1]
                # network.to_csv(output[2],index=False,sep="\t")

                # with open(output[3], "w") as f:
                #     for s in sorted(set(filtered_df.StrainSelectID.to_list())):
                #         f.write(f"{s}\n")



rule link_pmc_network:
    input:
        f"{preds}/network.tsv",
        f"{preds}/REL_output/preds_strainselect_grouped_pmc.pqt",
    output:
        f"{preds}/network_pmc.tsv",
    resources:
        **COMMON_RESOURCES,
    run:
        df = pl.read_parquet(input[1])
        network = pl.read_csv(input[0], separator="\t")
        network = network.join(
            df.select(
                [
                    "StrainSelectID",
                    "word_qc_group",
                    "pmcid",
                    "paragraph",
                    "sentence_range",
                ]
            ),
            left_on=["source", "target"],
            right_on=["StrainSelectID", "word_qc_group"],
            how="left",
        )
        # Only drop columns if they exist in the result
        cols_to_drop = [col for col in ["StrainSelectID", "word_qc_group"] if col in network.columns]
        if cols_to_drop:
            network = network.drop(cols_to_drop)
        network.write_csv(output[0], separator="\t")
