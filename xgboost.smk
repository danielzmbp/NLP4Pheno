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


# Helper function to get unique relationship types
input_df = f"{path}/preds{DATA}/REL_output/preds_strainselect_grouped.pqt"
def get_rels():
    df = pd.read_parquet(input_df)
    return df["rel"].unique()


rule all:
    input:
        f"{path}/xgboost/annotations{DATA}/binary/binary.pkl",
        f"{path}/xgboost/seqfiles_{DATA}/.EVOLUTION_DATASET_COMPLETE"

rule create_downloaded_strains_file:
    output:
        f"{path}/preds{DATA}/REL_output/strains_assemblies_downloaded.txt"
    resources:
        slurm_partition="cpu",
        runtime=60,
        mem_mb=10000
    run:
        filtered_assemblies = []
        for strain in glob(f"{path}/assemblies_{DATA}/*/"):
            for assembly in glob(f"{strain}/*/"):
                if len(os.listdir(assembly)) == 5:
                    ass = assembly.split("/")[-3] + "/" + assembly.split("/")[-2]
                    filtered_assemblies.append(ass)
        with open(output[0],"w") as f:
            for line in filtered_assemblies:
                f.write(line + "\n")

        
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
        slurm_partition="cpu",
        runtime=2000,
        mem_mb=150000,
    run:
        from pathlib import Path
        import concurrent.futures
        import pyarrow as pa

        # --- Helper function to load and process one annotation file ---
        def load_and_process_annotation(args):
            strain, assembly, base_path, cols_to_drop_list = args
            annotation_file = Path(base_path) / strain / assembly / "annotation.parquet"

            if not annotation_file.exists():
                return None # Return None for missing files

            annotation_df = pd.read_parquet(annotation_file) 
            # Drop unnecessary columns (handle missing columns gracefully)
            cols_present = [col for col in cols_to_drop_list if col in annotation_df.columns]
            if cols_present:
                annotation_df.drop(columns=cols_present, inplace=True)

            # Drop rows with missing InterPro accessions
            annotation_df.dropna(subset=["InterPro_accession"], inplace=True)

            if not annotation_df.empty:
                # Add identifiers back for merging
                annotation_df['strain'] = strain
                annotation_df['assembly'] = assembly

                essential_cols = ['InterPro_accession', 'Protein_ID', 'Length', 'Protein_accession', 'strain', 'assembly']
                cols_to_keep_final = [col for col in essential_cols if col in annotation_df.columns]
                return annotation_df[cols_to_keep_final]
            else:
                return None # Return None for empty or fully dropped dataframes



        # --- 1. Load Initial Data Efficiently ---
        # Load only required columns immediately

        df = pd.read_parquet(input.rel_file, columns=["StrainSelectID", "word_qc_group", "rel"])


        # Use read_csv for potentially faster parsing, handle empty file
        das = pd.read_csv(input.downloaded_strains, sep="/", header=None, names=["strain", "assembly"])
        downloaded_strains_set = set(das['strain'].unique())

        # Specify columns to load
        strainselect_vertices = pd.read_csv(
            input.strainselect_vertices,
            sep="\t",
            usecols=["StrainSelectID", "vertex_type", "vertex"]
        )
 

        # --- 2. Initial Filtering of Relation Data ---
        df.dropna(subset=["StrainSelectID"], inplace=True)
        df.drop_duplicates(inplace=True) 

        drel = df[(df.rel == wildcards.rel) & (df.StrainSelectID.isin(downloaded_strains_set))].copy()
        del df 


        # Keep only word_qc_groups appearing more than 2 times
        word_counts = drel["word_qc_group"].value_counts()
        drel = drel[drel["word_qc_group"].isin(word_counts[word_counts > 2].index)]


        # --- 3. Filter by Genus Diversity (Optimized) ---
        ss = strainselect_vertices[strainselect_vertices["vertex_type"] == "gss"].copy()
        del strainselect_vertices 
        ss.loc[:, "genus"] = ss['vertex'].str.split('.', n=1, expand=True)[0].astype('category') # Use category for genus
        ss = ss[["StrainSelectID", "genus"]]

        m = drel.merge(ss, on="StrainSelectID", how="left")
        del ss

        m['genus_count_in_group'] = m.groupby("word_qc_group")['genus'].transform('nunique')

        # Calculate the proportion of strains with the same genus within each group
        m['genus_count'] = m.groupby(["word_qc_group", "genus"])['StrainSelectID'].transform('count')
        m['total_strains_in_group'] = m.groupby("word_qc_group")['StrainSelectID'].transform('count')
        m['genus_proportion'] = m['genus_count'] / m['total_strains_in_group']

        # Filter groups where nunique > 4 first 
        # Filter out groups where any single genus makes up > 30% of the group
        m = m[m['genus_count_in_group'] > 4]
        m = m[m['genus_proportion'] <= 0.3]


        # Get the remaining valid word_qc_groups
        valid_groups = m['word_qc_group'].unique()
        drel = drel[drel["word_qc_group"].isin(valid_groups)] # Filter original drel

        del m 

        # --- 4. Prepare for Annotation Loading ---

        # Merge drel with available assemblies (das)
        drel_expanded = drel.merge(das, left_on="StrainSelectID", right_on="strain", how="inner")
        del drel
        del das

        # Identify unique strain/assembly pairs for which we need to load annotations
        unique_sa_to_load = drel_expanded[['strain', 'assembly']].drop_duplicates().reset_index(drop=True)


        # --- 5. Load and Process Annotations in Parallel ---
        all_processed_annotations = []
        # Prepare arguments for the parallel function
        tasks_args = [
            (row['strain'], row['assembly'], params.base_annotation_path, params.cols_to_drop)
            for _, row in unique_sa_to_load.iterrows()
        ]
        del unique_sa_to_load 

        # Use ThreadPoolExecutor for parallel I/O
        processed_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_sa = {executor.submit(load_and_process_annotation, arg): arg for arg in tasks_args}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_sa)):
                sa = future_to_sa[future]
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    processed_results.append(result_df)



        # Filter out potential None results if submit was used or errors occurred
        all_processed_annotations = [df for df in processed_results if df is not None]


        # --- 6. Combine Annotations and Merge ---
        # Concatenate all loaded annotation dataframes into one

        df_annotations_combined = pd.concat(all_processed_annotations, ignore_index=True)
        del all_processed_annotations 

        # Merge the combined annotations back to the expanded relation data
        # Ensure correct merge keys
        # Perform merge in chunks if df_annotations_combined or drel_expanded are massive to save memory
        final_df = drel_expanded.merge(
            df_annotations_combined,
            on=['strain', 'assembly'], # Corresponds to StrainSelectID and assembly
            how="inner" # Keep only rows where annotation data was successfully loaded and processed
        )
        del drel_expanded 
        del df_annotations_combined 


        # --- 7. Final Processing and Output ---
        # Add the combined identifier column 
        # Ensure columns exist before concatenation

        final_df.drop(columns=["rel","StrainSelectID"], inplace=True) 

        for col in final_df.columns:
            if final_df[col].dtype == 'object':
                # Attempt to convert to Pandas' Arrow-backed string type
                try:
                    # if final_df[col].dropna().apply(type).eq(str).all():
                    print(f"Converting column '{col}' to pd.StringDtype()...")
                    final_df[col] = final_df[col].astype(pd.StringDtype())
                except Exception as e:
                    print(f"Could not convert column '{col}' to pd.StringDtype(): {e}")

        
        final_df_pl = pl.from_pandas(final_df)

        final_df_pl = final_df_pl.with_columns(
            pl.concat_str(
                [
                    pl.col("strain").cast(pl.Utf8),
                    pl.lit("!"), # Literal string
                    pl.col("assembly").cast(pl.Utf8),
                    pl.lit("!"),
                    pl.col("word_qc_group").cast(pl.Utf8)
                ],
                separator=""
            ).alias("sa_ner")
        )

        # if 'strain' in final_df.columns and 'assembly' in final_df.columns and 'word_qc_group' in final_df.columns:
        #      final_df['sa_ner'] = final_df['strain'] + "!" + final_df['assembly'] + "!" + final_df['word_qc_group']
        # else:
        #      final_df['sa_ner'] = pd.NA


        # Final filter: ensure word_qc_groups still have more than two entry *after* merging with annotations
        # final_word_counts = final_df["word_qc_group"].value_counts()
        # final_df = final_df[final_df["word_qc_group"].isin(final_word_counts[final_word_counts > 2].index)]

        # # Select and order final columns for clarity and efficiency
        # # Define the exact columns needed in the output parquet file
        # output_columns = ['InterPro_accession', 'Protein_ID', 'Length', 'Protein_accession', 'sa_ner', 'word_qc_group']
        # # Ensure all expected columns exist, handle missing ones if necessary
        # final_output_columns = [col for col in output_columns if col in final_df.columns]


        # # Write only selected columns
        # final_df[final_output_columns].to_parquet(output.rel_output, index=False)
        # Final filter: ensure word_qc_groups still have more than two entries
        if "word_qc_group" in final_df_pl.columns:
            final_df_pl = final_df_pl.with_columns(
                pl.col("word_qc_group").count().over("word_qc_group").alias("temp_wqc_count")
            ).filter(
                pl.col("temp_wqc_count") > 2
            ).drop("temp_wqc_count") # Remove the temporary count column
        else:
            print("Warning: 'word_qc_group' column not found. Skipping group count filter.")


        # Select and order final columns for clarity and efficiency
        # Define the exact columns needed in the output parquet file
        output_columns_desired = ['InterPro_accession', 'Protein_ID', 'Length', 'Protein_accession', 'sa_ner', 'word_qc_group']

        # Ensure all expected columns exist in the Polars DataFrame
        current_polars_columns = final_df_pl.columns
        final_output_columns_pl = [col for col in output_columns_desired if col in current_polars_columns]

        # If no columns are selected (e.g., if final_df_pl became empty or desired columns don't exist),
        # handle this to avoid errors.
        if not final_output_columns_pl:
            print(f"Warning: No columns from '{output_columns_desired}' found in the DataFrame after processing. Parquet file will not be written or will be empty.")
        elif final_df_pl.is_empty():
            print(f"Warning: DataFrame is empty after filtering. Writing an empty Parquet file with selected columns: {final_output_columns_pl}")
            try:
                # Attempt to get schema from the (potentially empty) DataFrame for the selected columns
                schema_for_empty_df = {col: final_df_pl.schema[col] for col in final_output_columns_pl}
                pl.DataFrame(schema=schema_for_empty_df).write_parquet(output.rel_output)
            except Exception as e:
                print(f"Could not write empty parquet, possibly due to schema issues with an empty dataframe: {e}")
        else:
            # Select the columns and write to Parquet
            # Polars' write_parquet does not write an index by default.
            final_df_pl.select(final_output_columns_pl).write_parquet(output.rel_output)
            print(f"Successfully wrote selected columns to {output.rel_output}")




# Rule for creating pickle files, which includes the X, y, and index for input to XGBoost (features, labels, and index)
rule process_file:
    input:
        parquet_file=path + "/xgboost/annotations{data}/{rel}.parquet",
    output:
        pickle_file=path + "/xgboost/annotations{data}/{rel}.pkl",
    resources:
        slurm_partition="cpu",
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
        temp = tt.reset_index()["index"].str.split("!", expand=True)[2]
        tempdf = pd.concat([tt.reset_index(), temp], axis=1)
        tempdf = tempdf[tempdf[2].duplicated(keep=False)]
        tempdf.set_index("index", inplace=True)
        tempdf.drop(columns=[2], inplace=True)

        X = tempdf.to_numpy()
        y = tempdf.reset_index()["index"].str.split("!", expand=True)[2].to_numpy()

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
        slurm_partition="cpu",
        # slurm_extra="--gres=gpu:1",
        runtime=900,
        # tasks=5,
        mem_mb=50000,
    params:
        data=DATA,
        device=config["cuda_devices"],
        path=path
    conda:
        "xgb"
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
        slurm_partition="cpu",
        runtime=30,
        tasks=2,
        mem_mb=40000,
    params:
        data=DATA,
        device=config["cuda_devices"],
        path=path
    run:
        results = []
        for rel_file in input:
            with open(rel_file, "rb") as f:
                result = pickle.load(f)
                rel = rel_file.split("/")[-1].split(".")[0]
                results.append((rel, result))
        d = {}
        for rel, result in results:
            d[rel] = result

        with open(output[0], "wb") as f:
            pickle.dump(d, f)

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
    threads: 32
    resources:
        slurm_partition = "cpu",
        runtime         = 4300,
        mem_mb          = 250000
    script:
        "scripts/create_evolution_dataset_snakemake.py"
