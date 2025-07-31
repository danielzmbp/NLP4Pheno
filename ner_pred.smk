import pandas as pd
import os
import json
import numpy as np
import re
from operator import itemgetter
from glob import glob
import itertools
import csv

configfile: "config.yaml"

labels_flat = config["ner_labels"][1:]
cutoff = config["cutoff_prediction"]
cuda = config["cuda_devices"]
corpus = "corpus" + str(config["dataset"])
preds = config["output_path"] + "/preds" + str(config["dataset"])

# Common resource configurations
COMMON_RESOURCES = {
    "slurm_partition": "single",
    "runtime": 30,
    "mem_mb": 5000
}

def merge_entities(entity_list):
    """Merge contiguous B-I entity sequences into single entities"""
    merged_list = []
    skip = False

    for i in range(len(entity_list)):
        if skip:
            skip = False
            continue

        current_entity = entity_list[i]
        current_entity.pop("word", None)  # Remove 'word' entry
        
        if current_entity["entity_group"] == "B":
            scores = [current_entity["score"]]
            # Look ahead to find contiguous 'I' entities
            next_idx = i + 1
            while (
                next_idx < len(entity_list)
                and entity_list[next_idx]["entity_group"] == "I"
                and (entity_list[next_idx]["start"] - current_entity["end"]) <= 4
            ):
                scores.append(entity_list[next_idx]["score"])
                current_entity["end"] = entity_list[next_idx]["end"]
                skip = True
                next_idx += 1

            # Average the scores
            current_entity["score"] = sum(scores) / len(scores)

        merged_list.append(current_entity)

    return merged_list

def process_ner_predictions(file_paths, cutoff_score):
    """Process NER prediction files and merge entities"""
    dataframes = []
    for file in file_paths:
        df = pd.read_parquet(file).explode("ner").dropna()
        grouped = df.groupby("text").agg({"ner": lambda x: list(x)}).reset_index()
        grouped["ner"] = grouped["ner"].apply(merge_entities)
        grouped = grouped.explode("ner")
        grouped = pd.concat([grouped.drop(columns="ner"), grouped.ner.apply(pd.Series)], axis=1)
        dataframes.append(grouped)
    
    result_df = pd.concat(dataframes)
    result_df["word"] = result_df.apply(lambda row: row["text"][row["start"] : row["end"]], axis=1)
    result_df["word"] = result_df["word"].str.lower()
    return result_df[result_df["score"] > cutoff_score]

def create_device_model_mapping(device_list, model_list):
    """Create device-model mapping for distributed processing"""
    return [f"{device} {model}" for device, model in zip(itertools.cycle([str(x) for x in device_list]), model_list)]


(PARTS,) = glob_wildcards(corpus + "/{part}.txt")


rule all:
    input:
        preds + "/NER_output/preds.parquet",


rule make_strain_file:
    output:
        preds + "/NER_output/device_strain.txt",
    resources:
        **COMMON_RESOURCES,
    run:
        # Create device-strain mapping
        device_strain_pairs = create_device_model_mapping(cuda, PARTS)
        with open(output[0], "w") as f:
            for pair in device_strain_pairs:
                f.write(f"{pair}\n")


rule run_strain_prediction:
    input:
        corpus_file=corpus + "/{strain}.txt",
        dev=preds + "/NER_output/device_strain.txt",
    output:
        preds + "/NER_output/STRAIN/{strain}.parquet",
    conda:
        "pytorch"
    resources:
        slurm_partition="gpu_4",
        slurm_extra="--gres=gpu:1",
        runtime=70,
        mem_mb=5000
    shell:
        """
        while read -r d s; do
            if [ "$s" == "{wildcards.strain}" ]; then
                export MODEL=NER_output/STRAIN
                python scripts/ner_prediction_corpus.py --model $MODEL --device $d --output {output} --corpus {input.corpus_file}
                break
            fi
        done < {input.dev}
        """


rule merge_strain_predictions:
    input:
        expand(preds + "/NER_output/STRAIN/{p}.parquet", p=PARTS),
    output:
        preds + "/NER_output/STRAIN/strains.parquet",
    resources:
        slurm_partition="single",
        runtime=200,
        mem_mb=40000,
    run:
        file_paths = glob(preds + "/NER_output/STRAIN/*.parquet")
        df = process_ner_predictions(file_paths, cutoff)
        df.to_parquet(output[0])


rule make_sentence_file:
    input:
        preds + "/NER_output/STRAIN/strains.parquet",
    output:
        preds + "/NER_output/strains.txt",
        preds + "/NER_output/device_models.txt",
    resources:
        slurm_partition="single",
        runtime=300,
        mem_mb=10000,
    run:
        df = pd.read_parquet(input[0])
        df.drop_duplicates(subset="text")["text"].to_csv(
            output[0], sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE
        )
        
        # Create device-model mapping
        device_model_pairs = create_device_model_mapping(cuda, labels_flat)
        with open(output[1], "w") as f:
            for pair in device_model_pairs:
                f.write(f"{pair}\n")


rule run_all_models:
    input:
        preds + "/NER_output/strains.txt",
        preds + "/NER_output/device_models.txt",
    output:
        preds + "/NER_output/{l,[A-Z]+}.parquet",
    conda:
        "pytorch"
    resources:
        slurm_partition="gpu_4",
        slurm_extra="--gres=gpu:1",
        runtime=800,
    shell:
        """
        while read -r d m; do
           if [ "$m" = "{wildcards.l}" ]; then
               export MODEL=NER_output/${{m}}
               python scripts/ner_prediction_corpus.py --model $MODEL --device ${{d}} --output {preds}/$MODEL.parquet --corpus {input[0]} 
           fi
        done < {input[1]} 
        """


rule agg_model_results:
    input:
        expand(preds + "/NER_output/{l}.parquet", l=labels_flat),
    output:
        preds + "/NER_output/strain_preds.parquet",
    resources:
        slurm_partition="single",
        runtime=300,
        mem_mb=20000
    run:
        dataframes = []
        for ner_file in input:
            df = pd.read_parquet(ner_file).explode("ner").dropna()
            grouped = df.groupby("text").agg({"ner": lambda x: list(x)}).reset_index()
            grouped["ner"] = grouped["ner"].apply(merge_entities)
            grouped = grouped.explode("ner")
            grouped = pd.concat(
                [grouped.drop(columns="ner"), grouped.ner.apply(pd.Series)], axis=1
            )
            grouped["ner"] = ner_file.split("/")[-1].split(".")[0]
            dataframes.append(grouped)

        df = pd.concat(dataframes)
        df["word"] = df.apply(lambda row: row["text"][row["start"] : row["end"]], axis=1)
        df["word"] = df["word"].str.lower()
        df.to_parquet(output[0])


rule merge_preds:
    input:
        preds + "/NER_output/STRAIN/strains.parquet",
        preds + "/NER_output/strain_preds.parquet",
    output:
        preds + "/NER_output/preds.parquet",
    resources:
        slurm_partition="single",
        runtime=300,
        mem_mb=20000
    run:
        strains = pd.read_parquet(input[0])
        others = pd.read_parquet(input[1])
        others = others[others["score"] > cutoff]

        suff = strains.iloc[:, 1:].add_suffix("_strain")
        strains = pd.concat([strains.iloc[:, 0], suff], axis=1)
        df = strains.merge(others, on="text", how="left")
        df = df.dropna(subset="word")
        df.to_parquet(output[0])
