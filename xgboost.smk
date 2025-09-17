import pandas as pd
from glob import glob
import os
import pickle
import polars as pl
from tqdm import tqdm


configfile: "config.yaml"

# Define constants
DATA = config["dataset"]
path = config["output_path"]
input_df = f"{path}/preds{DATA}/REL_output/preds_strainselect_grouped.pqt"

# Common resource configurations
COMMON_CPU_RESOURCES = {
    "slurm_partition": "cpu,cpu_il",
    "runtime": 60,
    "mem_mb": 10000
}

HEAVY_CPU_RESOURCES = {
    "slurm_partition": "cpu,cpu_il",
    "runtime": 1500,
    "mem_mb": 100000
}

XGBOOST_RESOURCES = {
    "slurm_partition": "cpu,cpu_il",
    "runtime": 900,
    "mem_mb": 50000
}

def get_rels():
    """Get unique relationship types from processed predictions"""
    df = pd.read_parquet(input_df)
    return df["rel"].unique()

def load_and_process_annotation(args):
    """Load and process a single annotation file with optimized memory usage"""
    from pathlib import Path
    
    strain, assembly, base_path, cols_to_drop_list = args
    annotation_file = Path(base_path) / strain / assembly / "annotation.parquet"

    if not annotation_file.exists():
        return None

    try:
        # Use Polars for more efficient loading and processing
        annotation_df = pl.read_parquet(annotation_file)
        
        # Drop unnecessary columns if they exist
        cols_present = [col for col in cols_to_drop_list if col in annotation_df.columns]
        if cols_present:
            annotation_df = annotation_df.drop(cols_present)

        # Drop rows with missing InterPro accessions
        annotation_df = annotation_df.filter(pl.col("InterPro_accession").is_not_null())

        if not annotation_df.is_empty():
            # Add identifiers back for merging
            annotation_df = annotation_df.with_columns([
                pl.lit(strain).alias("strain"),
                pl.lit(assembly).alias("assembly")
            ])

            essential_cols = ['InterPro_accession', 'Protein_ID', 'Length', 'Protein_accession', 'strain', 'assembly']
            cols_to_keep_final = [col for col in essential_cols if col in annotation_df.columns]
            return annotation_df.select(cols_to_keep_final).to_pandas()
        else:
            return None
    except Exception as e:
        print(f"Error processing {annotation_file}: {e}")
        return None

def filter_by_genus_diversity(drel, strainselect_vertices):
    """Filter relationships by genus diversity requirements"""
    ss = strainselect_vertices[strainselect_vertices["vertex_type"] == "gss"].copy()
    ss = ss.copy()
    ss["genus"] = ss['vertex'].str.split('.', n=1, expand=True)[0].astype('category')
    ss = ss[["StrainSelectID", "genus"]]

    m = drel.merge(ss, on="StrainSelectID", how="left")
    m['genus_count_in_group'] = m.groupby("word_qc_group", observed=False)['genus'].transform('nunique')
    m['genus_count'] = m.groupby(["word_qc_group", "genus"], observed=False)['StrainSelectID'].transform('count')
    m['total_strains_in_group'] = m.groupby("word_qc_group", observed=False)['StrainSelectID'].transform('count')
    m['genus_proportion'] = m['genus_count'] / m['total_strains_in_group']

    # Filter groups where nunique > 4 and no single genus makes up > 30% of the group
    m = m[m['genus_count_in_group'] > 4]
    m = m[m['genus_proportion'] <= 0.3]

    valid_groups = m['word_qc_group'].unique()
    return drel[drel["word_qc_group"].isin(valid_groups)]


rule all:
    input:
        f"{path}/xgboost/annotations{DATA}/binary/binary.pkl",
        f"{path}/xgboost/seqfiles_{DATA}/.EVOLUTION_DATASET_COMPLETE"

# NOTE: strains_assemblies_downloaded.txt is now created by ip.smk checkpoint
# This rule has been removed to avoid duplication

        
# Rule for processing relationship files for all the assemblies
# Assuming 'path', 'DATA', 'input_df', 'log' are defined earlier in the Snakefile

rule process_rel:
    input:
        rel_file=input_df,
        downloaded_strains= f"{path}/preds{DATA}/REL_output/strains_assemblies_downloaded.txt",
        strainselect_vertices=f"{path}/preds{DATA}/strainselect/StrainSelect21_vertices.tab.txt",
    output:
        rel_output=path + "/xgboost/annotations{data}/{rel}.parquet",
    params:
        base_annotation_path=f"{path}/assemblies_{DATA}",
        cols_to_drop=[
            "Sequence_MD5_digest", "Score", "Sequence_length",
            "Start_location", "Stop_location", "GO_annotations", "Pathways_annotations"
        ]
    threads: 16
    resources:
        **HEAVY_CPU_RESOURCES,
    run:
        from pathlib import Path
        import concurrent.futures
        import pyarrow as pa

        # Helper function is defined at module level



        # --- 1. Load Initial Data ---
        # Use Polars lazy evaluation for better memory management
        df = pl.read_parquet(input.rel_file).select([
            "StrainSelectID", "word_qc_group", "rel"
        ]).lazy()

        # Load strain/assembly data efficiently
        das = pl.read_csv(input.downloaded_strains, separator="/", 
                         has_header=False, new_columns=["strain", "assembly"])
        downloaded_strains_set = set(das["strain"].unique())

        # Load strain vertices data
        strainselect_vertices = pl.read_csv(
            input.strainselect_vertices,
            separator="\t"
        ).select(["StrainSelectID", "vertex_type", "vertex"])
 

        # --- 2. Initial Filtering of Relation Data ---
        # Apply all filters using lazy evaluation
        drel = (df
                .filter(pl.col("StrainSelectID").is_not_null())
                .unique()
                .filter(
                    (pl.col("rel") == wildcards.rel) & 
                    (pl.col("StrainSelectID").is_in(downloaded_strains_set))
                )
                .collect()  # Materialize only when needed
        )
        
        # Keep only word_qc_groups appearing more than 2 times
        word_counts = drel.group_by("word_qc_group").agg(pl.count().alias("count"))
        valid_groups = word_counts.filter(pl.col("count") > 2)["word_qc_group"]
        drel = drel.filter(pl.col("word_qc_group").is_in(valid_groups))

        # --- 3. Filter by Genus Diversity ---
        drel = filter_by_genus_diversity(drel.to_pandas(), strainselect_vertices.to_pandas())
        del strainselect_vertices 

        # --- 4. Prepare for Annotation Loading ---
        
        # Convert back to Polars for merging
        drel_pl = pl.from_pandas(drel)
        
        # Merge with available assemblies
        drel_expanded = drel_pl.join(das, left_on="StrainSelectID", right_on="strain", how="inner")
        del drel, drel_pl, das

        # Check if join was successful
        if drel_expanded.is_empty():
            print(f"Warning: No matching assemblies found for {wildcards.rel}")
            # Create empty output file with proper schema
            empty_schema = {
                'InterPro_accession': pl.Utf8,
                'Protein_ID': pl.Utf8,
                'Length': pl.Int64,
                'Protein_accession': pl.Utf8,
                'sa_ner': pl.Utf8,
                'word_qc_group': pl.Utf8
            }
            pl.DataFrame(schema=empty_schema).write_parquet(output.rel_output)
            return

        unique_sa_to_load = drel_expanded.select(['StrainSelectID', 'assembly']).unique().rename({'StrainSelectID': 'strain'})

        # --- 5. Load and Process Annotations in Chunked Batches ---
        CHUNK_SIZE = max(1, min(100, len(unique_sa_to_load)))  # Process in chunks to manage memory
        
        all_processed_annotations = []
        unique_sa_pandas = unique_sa_to_load.to_pandas()
        
        for chunk_start in range(0, len(unique_sa_pandas), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(unique_sa_pandas))
            chunk_data = unique_sa_pandas.iloc[chunk_start:chunk_end]
            
            # Prepare arguments for the parallel function
            tasks_args = [
                (row['strain'], row['assembly'], params.base_annotation_path, params.cols_to_drop)
                for _, row in chunk_data.iterrows()
            ]

            # Process chunk in parallel
            chunk_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(threads, len(tasks_args))) as executor:
                future_to_sa = {executor.submit(load_and_process_annotation, arg): arg for arg in tasks_args}
                for future in concurrent.futures.as_completed(future_to_sa):
                    try:
                        result_df = future.result()
                        if result_df is not None and not result_df.empty:
                            chunk_results.append(result_df)
                    except Exception as e:
                        print(f"Error processing annotation: {e}")

            # Combine chunk results
            if chunk_results:
                chunk_combined = pd.concat(chunk_results, ignore_index=True)
                all_processed_annotations.append(chunk_combined)
                del chunk_results, chunk_combined

            print(f"Processed chunk {chunk_start//CHUNK_SIZE + 1}/{(len(unique_sa_pandas) + CHUNK_SIZE - 1)//CHUNK_SIZE}")

        del unique_sa_to_load, unique_sa_pandas

        # --- 6. Combine All Annotations ---
        if all_processed_annotations:
            df_annotations_combined = pd.concat(all_processed_annotations, ignore_index=True)
            del all_processed_annotations
            
            # Check if the combined dataframe is empty
            if df_annotations_combined.empty:
                print("Warning: Combined annotation dataframe is empty")
                return
        else:
            print("Warning: No annotation data was successfully loaded")
            return 

        # --- 7. Final Merging and Processing ---
        # Convert annotations to Polars for merging
        annotations_pl = pl.from_pandas(df_annotations_combined)
        del df_annotations_combined

        final_df_pl = drel_expanded.join(
            annotations_pl,
            left_on=['StrainSelectID', 'assembly'],
            right_on=['strain', 'assembly'],
            how="inner"
        )
        del drel_expanded, annotations_pl

        cols_to_drop = ["rel"]
        if "strain" in final_df_pl.columns:
            cols_to_drop.append("strain")
        
        final_df_pl = (final_df_pl
                      .drop(cols_to_drop)
                      .with_columns(
                          pl.concat_str([
                              pl.col("StrainSelectID").cast(pl.Utf8),
                              pl.lit("!"),
                              pl.col("assembly").cast(pl.Utf8),
                              pl.lit("!"),
                              pl.col("word_qc_group").cast(pl.Utf8)
                          ], separator="").alias("sa_ner")
                      )
                      .drop("StrainSelectID"))

        # --- 8. Final Filtering and Output ---
        # Apply final filter for word_qc_groups with more than 2 entries
        final_df_pl = (final_df_pl
                      .with_columns(pl.col("word_qc_group").count().over("word_qc_group").alias("temp_wqc_count"))
                      .filter(pl.col("temp_wqc_count") > 2)
                      .drop("temp_wqc_count")
        )

        # Define and select output columns
        output_columns_desired = ['InterPro_accession', 'Protein_ID', 'Length', 'Protein_accession', 'sa_ner', 'word_qc_group']
        final_output_columns_pl = [col for col in output_columns_desired if col in final_df_pl.columns]

        # Write output with error handling
        if not final_output_columns_pl:
            print(f"Warning: No required columns found. Expected: {output_columns_desired}")
            return
        elif final_df_pl.is_empty():
            print(f"Warning: Empty DataFrame after filtering. Writing empty file.")
            schema_for_empty_df = {col: final_df_pl.schema[col] for col in final_output_columns_pl}
            pl.DataFrame(schema=schema_for_empty_df).write_parquet(output.rel_output)
        else:
            # Write using Polars
            final_df_pl.select(final_output_columns_pl).write_parquet(output.rel_output)
            print(f"Successfully wrote {len(final_df_pl)} rows to {output.rel_output}")




# Rule for creating pickle files, which includes the X, y, and index for input to XGBoost (features, labels, and index)
rule process_file:
    input:
        parquet_file=path + "/xgboost/annotations{data}/{rel}.parquet",
    output:
        pickle_file=path + "/xgboost/annotations{data}/{rel}.pkl",
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=1000,
        mem_mb=140000,
    run:
        # Read the parquet file
        d = pl.read_parquet(input.parquet_file)

        # Create initial count of InterPro_accession
        t = (
            d.group_by("InterPro_accession")
            .agg(pl.count("InterPro_accession").alias("count"))
            .sort("InterPro_accession")
        )

        # Select only necessary columns and perform operations in one go
        d_tiny = d.select(["Protein_accession", "InterPro_accession", "sa_ner"])
        sa_unique = d_tiny.select("sa_ner").unique()

        # Perform grouping and aggregation for all sa_ner values at once
        result = (
            d_tiny.group_by(["sa_ner", "InterPro_accession"])
            .agg(pl.count("*").alias("count"))
            .group_by(["sa_ner", "InterPro_accession"])
            .agg(pl.sum("count").alias("count"))
            .pivot(values="count", index="InterPro_accession", on="sa_ner")
            .fill_null(0)
        )

        # Join the results with the initial count
        t = t.join(result, on="InterPro_accession", how="left")

        # Convert to pandas and perform final operations
        t = t.to_pandas().set_index("InterPro_accession")
        ind = t.index.to_list()
        tt = t.transpose()

        # Extract the third part of the index after splitting by '!'
        index_parts = tt.reset_index()["index"].str.split("!", expand=True)
        if index_parts.shape[1] < 3:
            raise ValueError("Index format error: Expected at least 3 parts separated by '!'")
        temp = index_parts[2]
        tempdf = pd.concat([tt.reset_index(), temp], axis=1)
        tempdf = tempdf[tempdf[2].duplicated(keep=False)]
        tempdf.set_index("index", inplace=True)
        tempdf.drop(columns=[2], inplace=True)

        X = tempdf.to_numpy()
        y_index_parts = tempdf.reset_index()["index"].str.split("!", expand=True)
        if y_index_parts.shape[1] < 3:
            raise ValueError("Index format error for y extraction: Expected at least 3 parts separated by '!'")
        y = y_index_parts[2].to_numpy()

        # Save the results
        with open(output.pickle_file, "wb") as f:
            pickle.dump([X, y, ind], f)



# Run XGBoost on the binary classification task, outputs are the pickles containing the model and the predictions
rule xgboost_binary_parts:
    input:
        path + "/xgboost/annotations{data}/{rel}.pkl"
    output:
        path + "/xgboost/annotations{data}/{rel}.pickle",
    threads: 32
    resources:
        **XGBOOST_RESOURCES,
    params:
        data=DATA,
        device=config["cuda_devices"],
        path=path
    conda:
        "envs/xgb.yml"
    script:
        "scripts/xgboost_binary_snakemake_cpu.py"


rule xgboost_binary_join:
    input:
        expand(
            path + "/xgboost/annotations{data}/{rel}.pickle",
            data=DATA,
            rel=get_rels(),
        ),
    output:
        path + f"/xgboost/annotations{DATA}/binary/binary.pkl",
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=30,
        tasks=2,
        mem_mb=40000,
    params:
        data=DATA,
        device=config["cuda_devices"],
        path=path
    conda:
        "envs/xgb.yml"
    script:
        "scripts/join_xgboost_results.py"

rule evolution_dataset:
    input:
        binary = f"{path}/xgboost/annotations{DATA}/binary/binary.pkl",
        parquet_files = expand(
            f"{path}/xgboost/annotations{DATA}/{{rel}}.parquet",
            rel=get_rels(),
        )
    output:
        f"{path}/xgboost/seqfiles_{DATA}/.EVOLUTION_DATASET_COMPLETE"
    params:
        path = path,
        data = DATA
    threads: 4
    resources:
        slurm_partition = "cpu,cpu_il",
        runtime         = 500,
        mem_mb          = 35000
    conda:
        "envs/xgb.yml"
    script:
        "scripts/create_evolution_dataset_snakemake.py"
