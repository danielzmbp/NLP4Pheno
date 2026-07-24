import pandas as pd
import os
import json
import numpy as np
import re
from operator import itemgetter
from glob import glob
import itertools
import csv
import sys

sys.path.append("scripts")
from ner_postprocess import merge_entities as merge_entity_spans


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


labels_flat = config["ner_labels"][1:]
cutoff = config["cutoff_prediction"]
cuda = config["cuda_devices"]
corpus = "corpus" + str(config["dataset"])
preds = config["output_path"].rstrip("/") + "/preds" + str(config["dataset"])
parquet_file = config["pmc_parquet_file"]
CPU_PARTITION = config.get("slurm_cpu_partition", "cpu,cpu_il")
GPU_PARTITION = config.get(
    "slurm_gpu_partition", "gpu_h100,gpu_a100_il,gpu_h100_il"
)
GPU_GRES = config.get("slurm_gpu_gres", "gpu:1")
NER_PREDICTION_RUNTIME = int(config.get("ner_prediction_runtime", 420))
NER_PREDICTION_MEM_MB = int(config.get("ner_prediction_mem_mb", 32000))

STRAIN_PART_COUNT = 250

COMMON_RESOURCES = {
    "slurm_partition": CPU_PARTITION,
    "runtime": 30,
    "mem_mb": 3000,
    "cpus_per_task": 2,
}
PROCESSED_FILES_CACHE = {}
MERGED_ENTITIES_CACHE = {}


def merge_entities(entity_list):
    """Merge contiguous B-I entity sequences"""
    cache_key = str(entity_list)
    if cache_key in MERGED_ENTITIES_CACHE:
        return MERGED_ENTITIES_CACHE[cache_key]

    merged_list = merge_entity_spans(entity_list)

    MERGED_ENTITIES_CACHE[cache_key] = merged_list
    return merged_list


def process_ner_predictions(file_paths, cutoff_score):
    """Process NER prediction files and merge entities"""
    dataframes = []
    for file in file_paths:
        if file in PROCESSED_FILES_CACHE:
            df = PROCESSED_FILES_CACHE[file]
        else:
            df = pd.read_parquet(file).explode("ner").dropna()
            grouped = df.groupby("text").agg({"ner": lambda x: list(x)}).reset_index()
            grouped["ner"] = grouped["ner"].apply(merge_entities)
            grouped = grouped.explode("ner")
            grouped = pd.concat(
                [grouped.drop(columns="ner"), grouped.ner.apply(pd.Series)], axis=1
            )
            PROCESSED_FILES_CACHE[file] = grouped
            df = grouped
        dataframes.append(df)

    result_df = pd.concat(dataframes, ignore_index=True)
    # Handle non-string values in text column
    result_df = result_df.dropna(subset=["text", "start", "end"])
    result_df["text"] = result_df["text"].astype(str)
    result_df["start"] = pd.to_numeric(result_df["start"], errors="coerce")
    result_df["end"] = pd.to_numeric(result_df["end"], errors="coerce")
    result_df = result_df.dropna(subset=["start", "end"])
    result_df["start"] = result_df["start"].astype(int)
    result_df["end"] = result_df["end"].astype(int)
    result_df["word"] = result_df.apply(
        lambda row: str(row["text"])[int(row["start"]) : int(row["end"])].lower(),
        axis=1,
    )
    return result_df[result_df["score"] > cutoff_score]


def create_device_model_mapping(device_list, model_list):
    """Create device-model mapping for distributed processing"""
    return [
        f"{device} {model}"
        for device, model in zip(
            itertools.cycle([str(x) for x in device_list]), model_list
        )
    ]


PARTS = [f"{i:04d}" for i in range(STRAIN_PART_COUNT)]


rule all:
    input:
        preds + "/NER_output/preds.parquet",


rule generate_corpus:
    input:
        parquet_file,
    output:
        expand(corpus + "/{part}.txt", part=PARTS),
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=200,
        mem_mb=64000,
        cpus_per_task=4,
    script:
        "scripts/generate_corpus_optimized.py"


rule make_strain_file:
    input:
        expand(corpus + "/{part}.txt", part=PARTS),
    output:
        preds + "/NER_output/device_strain.txt",
    resources:
        **COMMON_RESOURCES,
    run:
        device_strain_pairs = create_device_model_mapping(cuda, PARTS)
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], "w") as f:
            for pair in device_strain_pairs:
                f.write(f"{pair}\n")


rule run_strain_prediction:
    input:
        corpus_file=corpus + "/{part}.txt",
        dev=preds + "/NER_output/device_strain.txt",
    output:
        temp(preds + "/NER_output/STRAIN/{part}.parquet"),
    conda:
        "envs/pytorch.yml"
    retries: 3
    resources:
        slurm_partition=GPU_PARTITION,
        gres=GPU_GRES,
        runtime=80,
        mem_mb=8000,
        cpus_per_task=4,
    shell:
        """
        mkdir -p {preds}/NER_output/STRAIN
        while read -r d s; do
            if [ "$s" == "{wildcards.part}" ]; then
                export MODEL=NER_output/STRAIN
                export CUDA_VISIBLE_DEVICES=$d
                python scripts/ner_prediction_corpus.py --model $MODEL --device 0 --output {output} --corpus {input.corpus_file}
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
        slurm_partition=CPU_PARTITION,
        runtime=120,
        mem_mb=32000,
        cpus_per_task=8,
    run:
        import os

        strain_dir = preds + "/NER_output/STRAIN/"
        file_paths = [os.path.join(strain_dir, f"{i:04d}.parquet") for i in range(STRAIN_PART_COUNT)]
        df = process_ner_predictions(file_paths, cutoff)
        df.to_parquet(output[0], compression="snappy")


rule make_sentence_file:
    input:
        preds + "/NER_output/STRAIN/strains.parquet",
    output:
        preds + "/NER_output/strains.txt",
        preds + "/NER_output/device_models.txt",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=300,
        mem_mb=32000,
        cpus_per_task=4,
    run:
        # The merged strain table contains millions of rows and several
        # prediction columns, but this rule only needs the sentence text.
        # Restricting the Parquet projection avoids materializing the much
        # larger entity/score/offset columns before de-duplication.
        df = pd.read_parquet(input[0], columns=["text"])
        df.drop_duplicates(subset="text")["text"].to_csv(
            output[0], sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE
        )
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
        "envs/pytorch.yml"
    resources:
        slurm_partition=GPU_PARTITION,
        gres=GPU_GRES,
        runtime=NER_PREDICTION_RUNTIME,
        mem_mb=NER_PREDICTION_MEM_MB,
    shell:
        """
        while read -r d m; do
           if [ "$m" = "{wildcards.l}" ]; then
               export MODEL=NER_output/${{m}}
               export CUDA_VISIBLE_DEVICES=${{d}}
               python scripts/ner_prediction_corpus.py --model $MODEL --device 0 --output {output} --corpus {input[0]} 
           fi
        done < {input[1]} 
        """


rule agg_model_results:
    input:
        expand(preds + "/NER_output/{l}.parquet", l=labels_flat),
    output:
        preds + "/NER_output/strain_preds.parquet",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=180,
        mem_mb=48000,
        cpus_per_task=12,
    run:
        dataframes = []
        for ner_file in input:
            if ner_file in PROCESSED_FILES_CACHE:
                grouped = PROCESSED_FILES_CACHE[ner_file]
            else:
                df = pd.read_parquet(ner_file).explode("ner").dropna()
                grouped = (
                    df.groupby("text").agg({"ner": lambda x: list(x)}).reset_index()
                )
                grouped["ner"] = grouped["ner"].apply(merge_entities)
                grouped = grouped.explode("ner")
                grouped = pd.concat(
                    [grouped.drop(columns="ner"), grouped.ner.apply(pd.Series)], axis=1
                )
                PROCESSED_FILES_CACHE[ner_file] = grouped

            grouped["ner"] = os.path.basename(ner_file).split(".")[0]
            dataframes.append(grouped)

        df = pd.concat(dataframes, ignore_index=True)
        # Handle non-string values in text column
        df = df.dropna(subset=["text", "start", "end"])
        df["text"] = df["text"].astype(str)
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["start", "end"])
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["word"] = df.apply(
            lambda row: str(row["text"])[int(row["start"]) : int(row["end"])].lower(),
            axis=1,
        )
        df.to_parquet(output[0], compression="snappy")


rule merge_preds:
    input:
        preds + "/NER_output/STRAIN/strains.parquet",
        preds + "/NER_output/strain_preds.parquet",
    output:
        preds + "/NER_output/preds.parquet",
    resources:
        slurm_partition=CPU_PARTITION,
        runtime=180,
        mem_mb=32000,
        cpus_per_task=8,
    run:
        strains = pd.read_parquet(input[0])
        others = pd.read_parquet(input[1])
        others = others[others["score"] > cutoff]

        strain_cols = strains.columns[1:]
        strain_renamed = strains[strain_cols].add_suffix("_strain")
        strains_processed = pd.concat([strains.iloc[:, [0]], strain_renamed], axis=1)

        df = strains_processed.merge(
            others.dropna(subset=["word"]), on="text", how="left"
        )
        df = df.dropna(subset=["word"])
        df.to_parquet(output[0], compression="snappy")
