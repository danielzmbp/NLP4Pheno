import pandas as pd
from glob import glob
import os
import pickle
import polars as pl
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


configfile: "xgboost_config.yaml"


# Define constants
DATA = config["data"]
MAX_ASSEMBLY = config["max_assembly"]
MIN_SAMPLES = config["min_samples"]


# Helper function to get unique relationship types
def get_rels():
    df = pd.read_parquet(f"/home/gomez/gomez/preds{DATA}/REL_output/preds.parquet")
    return df[df["rel"] != "STRAIN-EFFECT:ASSOCIATED_WITH"]["rel"].unique()
    # return df["rel"].unique()


rule all:
    input:
        f"/home/gomez/gomez/xgboost/annotations{DATA}_{MAX_ASSEMBLY}/binary/binary_{MIN_SAMPLES}.pkl",


# Rule for processing relationship files for all the assemblies
rule process_rel:
    input:
        rel_file=f"/home/gomez/gomez/preds{DATA}/REL_output/preds.parquet",
    output:
        rel_output="/home/gomez/gomez/xgboost/annotations{data}_{max_assembly}/{rel}.parquet",
    run:
        df = pd.read_parquet(input.rel_file)

        df_small = df[["word_strain_qc", "word_qc", "rel"]]
        df_small = df_small.drop_duplicates()
        df_small.loc[:, "word_strain_qc"] = (
            df_small.word_strain_qc.str.replace("/", " ")
            .str.replace(")", " ")
            .str.replace("_", " ")
            .str.replace("(", " ")
            .str.replace("=", " ")
            .str.replace("'", " ")
            .str.replace(";", " ")
            .str.replace(",", " ")
            .str.replace("|", " ")
            .str.replace(".", " ")
            .str.replace("-", " ")
            .str.replace("   ", " ")
            .str.replace("  ", " ")
            .str.lstrip(" ")
            .str.rstrip(" ")
        )

        max_assembly = MAX_ASSEMBLY

        pred_strains = df_small.word_strain_qc.to_list()
        folders = glob(f"/home/gomez/gomez/assemblies_linkbert_{MAX_ASSEMBLY}/*/")
        assemblies = [f.split("/")[-2].replace("_", " ") for f in folders]
        annotations = glob(
            f"/home/gomez/gomez/assemblies_linkbert_{MAX_ASSEMBLY}/**/**/*.parquet"
        )
        pred_annotations = [
            f for f in annotations if f.split("/")[-3].replace("_", " ") in pred_strains
        ]

        annotation_counts = pd.Series(
            [i.split("/")[-3].replace("_", " ") for i in pred_annotations]
        ).value_counts()
        small_assemblies = annotation_counts[annotation_counts <= max_assembly]


        def process_rel(rel_file):
            rel = rel_file.split("/")[-1].replace(".parquet", "")
            drel = df_small[df_small.rel == rel]
            word_counts = drel["word_qc"].value_counts()
            drel = drel[drel["word_qc"].isin(word_counts[word_counts > 1].index)]

            rel_annotations = []
            for _, row in tqdm(drel.iterrows(), total=drel.shape[0]):
                strain = row.word_strain_qc
                word = row.word_qc
                rel = row.rel
                if strain in small_assemblies:
                    annotations_for_assembly = [
                        a
                        for a in pred_annotations
                        if a.split("/")[-3].replace("_", " ") == strain
                    ]
                    for annotation in annotations_for_assembly:
                        annotation_df = pd.read_parquet(annotation)
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
                        assembly = annotation.split("/")[-2]
                        annotation_df["sa_ner"] = strain + "_" + assembly + "_" + word
                        annotation_df["word_qc"] = word
                        rel_annotations.append(annotation_df)
            try:
                df_rel_annotations = pd.concat(rel_annotations)
                wcts = df_rel_annotations["word_qc"].value_counts()
                df_rel_annotations = df_rel_annotations[
                    df_rel_annotations["word_qc"].isin(wcts[wcts > 1].index)
                ]
                df_rel_annotations.to_parquet(rel_file)
            except Exception as e:
                print(f"Error: {e}")


        process_rel(output.rel_output)


# Rule for creating pickle files, which includes the X, y, and index for input to XGBoost (features, labels, and index)
rule process_file:
    input:
        parquet_file="/home/gomez/gomez/xgboost/annotations{data}_{max_assembly}/{rel}.parquet",
    output:
        pickle_file="/home/gomez/gomez/xgboost/annotations{data}_{max_assembly}/{rel}.pkl",
    run:
        def process_file(output_file):
            rel = output_file.split("/")[-1].replace(".parquet", "")

            d = pl.read_parquet(input.parquet_file)
            t = (
                d.group_by("InterPro_accession")
                .agg(pl.count("InterPro_accession").alias("count"))
                .sort("InterPro_accession")
            )
            d_tiny = d.select(["Protein_accession", "InterPro_accession", "sa_ner"])
            sa_unique = d_tiny.select("sa_ner").unique().to_pandas()["sa_ner"].tolist()
            for sa in tqdm(sa_unique):
                selected = d_tiny.filter(pl.col("sa_ner") == sa)
                s = selected.group_by(["Protein_accession", "InterPro_accession"]).agg(
                    pl.count("*").alias("count")
                )
                selected_valuecount = (
                    s.group_by("InterPro_accession")
                    .agg(pl.sum("count").alias(sa))
                    .sort("InterPro_accession")
                )
                t = t.join(selected_valuecount, on="InterPro_accession", how="left")
            t = t.to_pandas().set_index("InterPro_accession")
            ind = t.index.to_list()
            tt = t.transpose()
            temp = tt.reset_index()["index"].str.split("_", expand=True)[2]
            tempdf = pd.concat([tt.reset_index(), temp], axis=1)
            tempdf = tempdf[tempdf[2].duplicated(keep=False)]
            tempdf.set_index("index", inplace=True)
            tempdf.drop(columns=[2], inplace=True)
            X = tempdf.to_numpy()
            y = tempdf.reset_index()["index"].str.split("_", expand=True)[2].to_numpy()
            with open(output_file, "wb") as f:
                pickle.dump([X, y, ind], f)


        process_file(output.pickle_file)


# Run XGBoost on the binary classification task, outputs are the pickles containing the model and the predictions
rule xgboost_binary:
    input:
        expand(
            "/home/gomez/gomez/xgboost/annotations{data}_{max_assembly}/{rel}.pkl",
            data=DATA,
            max_assembly=MAX_ASSEMBLY,
            rel=get_rels(),
        ),
    output:
        f"/home/gomez/gomez/xgboost/annotations{DATA}_{MAX_ASSEMBLY}/binary/binary_{MIN_SAMPLES}.pkl",
    params:
        data=DATA,
        max_assembly=MAX_ASSEMBLY,
        min_samples=MIN_SAMPLES,
        device=config["device"],
    conda:
        "xgb"
    script:
        "scripts/xgboost_binary_snakemake.py"
