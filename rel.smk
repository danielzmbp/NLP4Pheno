import pandas as pd
import os
import json
import numpy as np
import re
from itertools import permutations
from sklearn.model_selection import train_test_split
import jsonlines
import sys

sys.path.append("scripts")
from annotation_utils import load_annotations, load_unique_pmc_groups, source_group
from relation_data import build_relation_rows, relation_membership, split_by_task


configfile: "config.yaml"


labels = config["rel_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]
test_size = config["rel_test"]
annotation_pmc_matches_file = config.get("annotation_pmc_matches_file")

# Common resource configuration
COMMON_RESOURCES = {"slurm_partition": "cpu", "runtime": 260, "mem_mb": 8000}

# Cache strain catalog globally to avoid recomputation
STRAIN_CATALOG = None
JSON_DATA_CACHE = None


def load_json_data(input_file):
    """Load and cache JSON data to avoid repeated file reads"""
    global JSON_DATA_CACHE
    if JSON_DATA_CACHE is None:
        JSON_DATA_CACHE = load_annotations(input_file)
    return JSON_DATA_CACHE


def extract_strain_catalog(json_data):
    """Extract unique strain names from annotation data with caching"""
    global STRAIN_CATALOG
    if STRAIN_CATALOG is not None:
        return STRAIN_CATALOG

    strain_catalog = set()  # Use set for faster lookups
    for item in json_data:
        if item.get("annotations"):
            for annotation in item["annotations"]:
                if annotation.get("result"):
                    for result in annotation["result"]:
                        if (
                            "value" in result
                            and "labels" in result["value"]
                            and result["value"]["labels"][0] == "STRAIN"
                        ):
                            strain_catalog.add(result["value"]["text"])
    STRAIN_CATALOG = list(strain_catalog)
    return STRAIN_CATALOG


rule all:
    input:
        "REL_output/all_metrics.png",


rule parse_rels:
    input:
        input_file,
    output:
        "REL/parsed_rels.txt",
    resources:
        slurm_partition="cpu",
        runtime=60,
        mem_mb=12000,
    run:
        data = load_json_data(input[0])
        rel_df, stats = build_relation_rows(data)
        print(json.dumps(stats, sort_keys=True))
        rel_df.to_csv(output[0], sep="\t", index=False)


rule split_labels:
    input:
        "REL/parsed_rels.txt",
    output:
        expand("REL/{ENT}/all.tsv", ENT=labels),
    resources:
        **COMMON_RESOURCES,
    run:
        df = pd.read_csv(input[0], sep="\t")
        for label in labels:
            relation = label.split(":")[0]
            df[df["pair_type"] == relation].to_csv(
                f"REL/{label}/all.tsv", sep="\t", index=False
            )


rule split_sets:
    input:
        expand("REL/{ENT}/all.tsv", ENT=labels),
        input_file,
        *([annotation_pmc_matches_file] if annotation_pmc_matches_file else []),
    output:
        datasets=expand("REL/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
        summary="REL/split_summary.json",
    resources:
        **COMMON_RESOURCES,
    params:
        seed=config["seed"],
    run:
        summary = {}
        pmc_groups = load_unique_pmc_groups(annotation_pmc_matches_file)
        for label in labels:
            rel_label = label.split(":")[1]
            df = pd.read_csv(f"REL/{label}/all.tsv", sep="\t")
            df.loc[:, "binary_label"] = relation_membership(
                df["relations"], rel_label
            ).astype(int)
            df.loc[:, "split_group"] = df["task_id"].map(
                lambda task_id: source_group(task_id, pmc_groups)
            )
            data_sets = split_by_task(
                df,
                test_and_dev_size=test_size,
                seed=params.seed,
                group_column="split_group",
            )
            split_task_ids = {
                split: set(data["task_id"].unique()) for split, data in data_sets.items()
            }
            if (
                split_task_ids["train"] & split_task_ids["dev"]
                or split_task_ids["train"] & split_task_ids["test"]
                or split_task_ids["dev"] & split_task_ids["test"]
            ):
                raise RuntimeError(f"Task leakage detected while splitting {label}")
            split_source_groups = {
                split: set(data["split_group"].unique())
                for split, data in data_sets.items()
            }
            if (
                split_source_groups["train"] & split_source_groups["dev"]
                or split_source_groups["train"] & split_source_groups["test"]
                or split_source_groups["dev"] & split_source_groups["test"]
            ):
                raise RuntimeError(f"Source-article leakage detected while splitting {label}")
            summary[label] = {
                "all": {
                    "rows": int(len(df)),
                    "tasks": int(df["task_id"].nunique()),
                    "groups": int(df["split_group"].nunique()),
                    "pmc_linked_tasks": int(
                        df.loc[df["task_id"].astype(str).isin(pmc_groups), "task_id"].nunique()
                    ),
                    "positive": int(df["binary_label"].sum()),
                    "negative": int((df["binary_label"] == 0).sum()),
                },
                "splits": {},
            }

            for data_set, data in data_sets.items():
                summary[label]["splits"][data_set] = {
                    "rows": int(len(data)),
                    "tasks": int(data["task_id"].nunique()),
                    "groups": int(data["split_group"].nunique()),
                    "positive": int(data["binary_label"].sum()),
                    "negative": int((data["binary_label"] == 0).sum()),
                }
                data = (
                    data.reset_index(drop=True)
                    .reset_index()[["index", "task_id", "sentence", "binary_label"]]
                )
                with jsonlines.open(f"REL/{label}/{data_set}.json", mode="w") as writer:
                    for row in data.itertuples(index=False):
                        writer.write(
                            {
                                "id": row[0],
                                "task_id": row[1],
                                "sentence": row[2],
                                "label": row[3],
                            }
                        )
        with open(output.summary, "w") as handle:
            json.dump(
                {
                    "grouping": "unique_pmcid_else_task_id",
                    "pmc_linked_tasks": len(pmc_groups),
                    "relations": summary,
                },
                handle,
                indent=2,
                sort_keys=True,
            )


rule run_linkbert:
    input:
        expand("REL/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
    output:
        test=expand("REL_output/{ENT}/test_stats.json", ENT=labels),
        dev=expand("REL_output/{ENT}/dev_stats.json", ENT=labels),
    conda:
        "envs/pytorch.yml"
    params:
        epochs=config["rel_epochs"],
        cuda=lambda w: ",".join([str(i) for i in cuda]),
        model_type=config["model"],
        entities=" ".join(labels),
    resources:
        slurm_partition="gpu_h100,gpu_a100_il,gpu_h100_il",
        slurm_extra="--gres=gpu:1",
        runtime=250,
        mem_mb=32000,
    shell:
        """
        export CUDA_VISIBLE_DEVICES={params.cuda}
        export MODEL=BioLinkBERT-{params.model_type}
        export MODEL_PATH=michiyasunaga/$MODEL
        export USE_CODALAB=1
        python -c "import torch; print(f'Using GPU: {{torch.cuda.is_available()}}'); print(f'GPU Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}}')"
        if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
            echo "ERROR: No GPU detected by PyTorch; relation training is intentionally disabled on CPU."
            exit 1
        fi
        for entity in {params.entities};
        do
            datadir=REL/$entity
            outdir=REL_output/$entity
            mkdir -p $outdir
            python3 -u scripts/run_seqcls.py --model_name_or_path $MODEL_PATH \
            --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
            --do_train --do_eval --do_predict --report_to none --metric_name PRF1 \
            --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
            --learning_rate 3e-5 --num_train_epochs {params.epochs} --max_seq_length 512 \
            --save_strategy epoch --eval_strategy epoch --logging_strategy epoch --output_dir $outdir --overwrite_output_dir --load_best_model_at_end \
            --metric_for_best_model F1 --greater_is_better True \
            2>&1 | tee $outdir/log.txt
            rm -rf $outdir/checkpoint-*
        done
        """


rule merge_results:
    input:
        test="REL_output/{ENT}/test_stats.json",
        dev="REL_output/{ENT}/dev_stats.json",
    output:
        "REL_output/{ENT}/all_results.json",
    resources:
        **COMMON_RESOURCES,
    run:
        import json
        
        # Initialize merged results dictionary
        all_results = {}
        
        # Load test results (already have test_ prefix)
        if os.path.exists(input.test):
            with open(input.test, 'r') as f:
                test_data = json.load(f)
                # Don't add extra prefix - metrics already have test_ prefix
                all_results.update(test_data)
        
        # Load dev results (have eval_ prefix, not dev_)
        if os.path.exists(input.dev):
            with open(input.dev, 'r') as f:
                dev_data = json.load(f)
                # Don't add extra prefix - metrics already have eval_ prefix
                all_results.update(dev_data)
        
        # Write merged results
        with open(output[0], 'w') as f:
            json.dump(all_results, f, indent=2)


rule join_metrics:
    input:
        mets=expand("REL_output/{ENT}/all_results.json", ENT=labels),
    output:
        "REL_output/all_metrics.tsv",
    resources:
        **COMMON_RESOURCES,
    run:
        dfs = []
        for f in input.mets:
            with open(f, "r") as file:
                data = json.load(file)
            relation = f.split("/")[1]
            df = pd.DataFrame({relation: data})
            dfs.append(df)
        # Concatenate all dataframes side by side (columns are relations)
        result = pd.concat(dfs, axis=1)
        # The result now has metrics as rows and relations as columns
        result.to_csv(output[0], sep="\t")


rule plot_metrics:
    input:
        "REL_output/all_metrics.tsv",
    output:
        "REL_output/all_metrics.png",
    params:
        labels=labels,
    resources:
        **COMMON_RESOURCES,
    script:
        "scripts/rel_plot_performance.py"
