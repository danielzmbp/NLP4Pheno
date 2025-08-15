import pandas as pd
import os
import json
import numpy as np
import re
from itertools import permutations
from sklearn.model_selection import train_test_split
import jsonlines


configfile: "config.yaml"


labels = config["rel_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]
test_size = config["rel_test"]

# Common resource configuration
COMMON_RESOURCES = {"slurm_partition": "cpu", "runtime": 260, "mem_mb": 8000}

# Cache strain catalog globally to avoid recomputation
STRAIN_CATALOG = None
JSON_DATA_CACHE = None


def load_json_data(input_file):
    """Load and cache JSON data to avoid repeated file reads"""
    global JSON_DATA_CACHE
    if JSON_DATA_CACHE is None:
        with open(input_file) as f:
            JSON_DATA_CACHE = json.load(f)
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
        # Use cached JSON data
        data = load_json_data(input[0])
        ners = []
        rels = []
        for sentence in data:
            if sentence.get("annotations") and sentence["annotations"]:
                for annotation in sentence["annotations"][0]["result"]:
                    if annotation["type"] == "labels":
                        df = pd.json_normalize(annotation)
                        df["sentence_id"] = sentence["id"]
                        ners.append(df)
                    elif annotation["type"] == "relation":
                        df = pd.json_normalize(annotation)
                        df["sentence_id"] = sentence["id"]
                        rels.append(df)
        ner = pd.concat(ners, ignore_index=True) if ners else pd.DataFrame()
        rel = pd.concat(rels, ignore_index=True) if rels else pd.DataFrame()

        perms = ner.groupby("sentence_id").apply(
            lambda x: list(permutations(x["id"], 2))
        )
        perms.name = "perms"  # get every possible combination of entities in a sentence

        df = pd.json_normalize(data)
        d = ner.merge(perms, on="sentence_id").merge(
            df[["id", "data.text"]], left_on="sentence_id", right_on="id"
        )
        d = d.drop(columns="id_y").rename(columns={"id_x": "id"})

        sentences = []
        labels = []
        for id in d.sentence_id.drop_duplicates().to_list():
            sentence = d[d["sentence_id"] == id].loc[:, "data.text"].values[0]
            for perm in d[d["sentence_id"] == id].perms.to_list()[0]:
                text0 = d[d["id"] == perm[0]].loc[:, "value.text"].values[0]
                lab0 = str(d[d["id"] == perm[0]].loc[:, "value.labels"].values[0])
                text1 = d[d["id"] == perm[1]].loc[:, "value.text"].values[0]
                lab1 = str(d[d["id"] == perm[1]].loc[:, "value.labels"].values[0])
                replaced_sentence = sentence.replace(text0, f"@{lab0[2:-2]}$").replace(
                    text1, f"@{lab1[2:-2]}$"
                )
                sentences.append(replaced_sentence)
                rel_label = rel[
                    (rel["from_id"] == perm[0]) & (rel["to_id"] == perm[1])
                ].labels.values
                if rel_label.size == 1:
                    # Check if rel_label[0] is a valid list/array with content
                    try:
                        # Try to get the length - this will fail for NaN/None
                        rel_len = len(rel_label[0]) if hasattr(rel_label[0], '__len__') else 0
                        
                        if rel_len == 2:
                            sentences.append(replaced_sentence)
                            for lbl in rel_label[0]:
                                label = f"{lab0[2:-2]}-{lab1[2:-2]}:{str(lbl)}"
                                labels.append(label)
                        elif rel_len == 1:
                            label = f"{lab0[2:-2]}-{lab1[2:-2]}:{str(rel_label[0])[2:-2]}"
                            labels.append(label)
                        elif rel_len == 0:
                            # Empty list/array
                            label = f"{lab0[2:-2]}-{lab1[2:-2]}:"
                            labels.append(label)
                    except (TypeError, ValueError):
                        # Handle case where labels is NaN, None, or other non-iterable
                        print(f"WARNING: Empty/NaN relation label in sentence ID {id}")
                        print(f"  Sentence: {sentence[:100]}...")
                        print(f"  Entity pair: {text0} ({lab0[2:-2]}) -> {text1} ({lab1[2:-2]})")
                        print(f"  Raw label value: {rel_label[0]}")
                        label = f"{lab0[2:-2]}-{lab1[2:-2]}:"
                        labels.append(label)
                elif rel_label.size == 0:
                    label = f"{lab0[2:-2]}-{lab1[2:-2]}:"
                    labels.append(label)
        rel_df = pd.DataFrame({"sentence": sentences, "label": labels})
        # replace all hyphens in the data by spaces
        rel_df["sentence"] = rel_df["sentence"].str.replace(r"(?<=\w)-(?=\w)", " ")
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
            df[df["label"].str.startswith(relation)].to_csv(
                f"REL/{label}/all.tsv", sep="\t", index=False
            )


rule split_sets:
    input:
        expand("REL/{ENT}/all.tsv", ENT=labels),
        input_file,
    output:
        expand("REL/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
    resources:
        **COMMON_RESOURCES,
    params:
        seed=config["seed"],
    run:
        # Use cached JSON data and strain catalog
        data = load_json_data(input_file)
        strain_catalog = extract_strain_catalog(data)

        for label in labels:
            rel_label = label.split(":")[1]
            df = pd.read_csv(f"REL/{label}/all.tsv", sep="\t")

            df.rename(columns={"label": "l"}, inplace=True)

            df.loc[:, "label"] = np.where(df.l.str.endswith(rel_label), 1, 0)

            train, test_eval = train_test_split(
                df, test_size=test_size, stratify=df.label, random_state=params.seed
            )

            # Optimized data augmentation for positive samples
            positive_samples = train[train["label"] == 1]
            negative_samples = train[train["label"] == 0]
            num_to_generate = (len(negative_samples) - len(positive_samples)) // 5

            if num_to_generate > 0:
                augmented_rows = []
                strain_catalog_set = set(strain_catalog)  # Faster lookup

                for _ in range(num_to_generate):
                    random_sentence = positive_samples.sample(1).iloc[0]
                    sentence_text = random_sentence.sentence
                    for strain in strain_catalog:
                        if strain in sentence_text:
                            alternatives = list(strain_catalog_set - {strain})
                            if alternatives:
                                new_strain = np.random.choice(alternatives)
                                augmented_sentence = sentence_text.replace(
                                    strain, new_strain
                                )
                                augmented_row = random_sentence.copy()
                                augmented_row.sentence = augmented_sentence
                                augmented_rows.append(augmented_row)
                                break

                if augmented_rows:
                    train = pd.concat(
                        [train, pd.DataFrame(augmented_rows)], ignore_index=True
                    )

            test, evaluation = train_test_split(
                test_eval,
                test_size=0.5,
                stratify=test_eval.label,
                random_state=params.seed,
            )

            data_sets = {"test": test, "dev": evaluation, "train": train}

            for data_set, data in data_sets.items():
                data = (
                    data.reset_index()
                    .drop(columns="index")
                    .reset_index()[["index", "sentence", "label"]]
                )
                with jsonlines.open(f"REL/{label}/{data_set}.json", mode="w") as writer:
                    for row in data.itertuples(index=False):
                        writer.write(
                            {"id": row[0], "sentence": row[1], "label": row[2]}
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
            |& tee $outdir/log.txt
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
