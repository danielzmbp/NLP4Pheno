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
        f"{path}/xgboost/annotations{DATA}/binary.pkl",

rule create_downloaded_strains_file:
    output:
        f"{path}/preds{DATA}/REL_output/strains_assemblies_downloaded.txt"
    resources:
        slurm_partition="single",
        runtime=30,
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
rule process_rel:
    input:
        rel_file=input_df,
        downloaded_strains= f"{path}/preds{DATA}/REL_output/strains_assemblies_downloaded.txt",
        strainselect_vertices=f"{path}/preds{DATA}/strainselect/StrainSelect21_vertices.tab.txt",
    output:
        rel_output=path + "/xgboost/annotations{data}/{rel}.parquet",
    resources:
        slurm_partition="fat",
        runtime=100,
        mem_mb=300000,
        tasks=25,
    run:
        df = pd.read_parquet(input.rel_file)

        df = df[["StrainSelectID", "word_qc_group", "rel"]].dropna(subset="StrainSelectID").drop_duplicates()

        # Get downloaded strains
        with open(input.downloaded_strains, "r") as f:
            downloaded_assemblies = f.readlines()
        downloaded_assemblies = [l.strip() for l in downloaded_assemblies]

        das = pd.DataFrame(downloaded_assemblies)[0].str.split("/",expand=True)
        das.rename(columns={0:"strain",1:"assembly"}, inplace=True)

        drel = df[df.rel == wildcards.rel]
        word_counts = drel["word_qc_group"].value_counts()
        drel = drel[drel["word_qc_group"].isin(word_counts[word_counts > 1].index)]

        # Remove groups that include only strains from the same genus
        strainselect_vertices = pd.read_csv(input.strainselect_vertices, sep="\t")
        ss = strainselect_vertices[strainselect_vertices["vertex_type"] == "gss"]
        ss["genus"] = ss.vertex.str.split(".", expand=True)[0]
        ss = ss[["StrainSelectID", "genus"]]
        m = drel.merge(
            ss,
            on="StrainSelectID",
            how="left",
        )
        genus_groups = m.groupby("word_qc_group").apply(
            lambda x: x["genus"].nunique() > 1
        )
        valid_groups = genus_groups[genus_groups].index
        drel = drel[drel["word_qc_group"].isin(valid_groups)]

        rel_annotations = []
        for _, row in tqdm(drel.iterrows(), total=drel.shape[0]):
            strain = row.StrainSelectID
            word = row.word_qc_group
            if strain in das.strain.unique():
                annotations_for_assembly = das[das.strain == strain].assembly.to_list()
                for annotation in annotations_for_assembly:
                    annotation_df = pd.read_parquet(f"{path}/assemblies_{DATA}/{strain}/{annotation}/annotation.parquet")
                    annotation_df.drop(
                        columns=[
                            "Sequence_MD5_digest",
                            "Score",
                            "Sequence_length",
                            "Start_location",
                            "Stop_location",
                            "GO_annotations",
                            "Pathways_annotations",
                        ],
                        inplace=True,
                    )
                    annotation_df.dropna(
                        subset=["InterPro_accession"], inplace=True
                    )
                    annotation_df["sa_ner"] = strain + "!" + annotation + "!" + word
                    annotation_df["word_qc_group"] = word
                    rel_annotations.append(annotation_df)
        try:
            df_rel_annotations = pd.concat(rel_annotations)
            wcts = df_rel_annotations["word_qc_group"].value_counts()
            df_rel_annotations = df_rel_annotations[
                df_rel_annotations["word_qc_group"].isin(wcts[wcts > 1].index)
            ]
            df_rel_annotations.to_parquet(output.rel_output)
        except Exception as e:
            print(f"Error: {e}")



# Rule for creating pickle files, which includes the X, y, and index for input to XGBoost (features, labels, and index)
rule process_file:
    input:
        parquet_file=path + "/xgboost/annotations{data}/{rel}.parquet",
    output:
        pickle_file=path + "/xgboost/annotations{data}/{rel}.pkl",
    resources:
        slurm_partition="fat",
        runtime=30,
        mem_mb=220000,
        tasks=10,
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
            .pivot(values="count", index="InterPro_accession", columns="sa_ner")
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
    resources:
        slurm_partition="gpu_4",
        slurm_extra="--gres=gpu:1",
        runtime=1440,
        tasks=5,
    params:
        data=DATA,
        device=config["cuda_devices"],
        path=path
    conda:
        "xgb"
    script:
        "scripts/xgboost_binary_snakemake.py"


rule xgboost_binary_join:
    input:
        expand(
            path + "/xgboost/annotations{data}/{rel}.pickle",
            data=DATA,
            rel=get_rels(),
        ),
    output:
        path + f"/xgboost/annotations{DATA}/binary.pkl",
    resources:
        slurm_partition="single",
        runtime=30,
        tasks=2,
        mem_mb=20000,
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

