import pandas as pd
import os
import json
import numpy as np
import re
import random
import copy
from operator import itemgetter
from sklearn.model_selection import train_test_split


configfile: "config.yaml"


labels = config["ner_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]
test_size = config["ner_test"]

# Common resource configuration
COMMON_RESOURCES = {"slurm_partition": "cpu", "runtime": 10, "mem_mb": 8000}

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


def replace_hyphens_in_data(sentences):
    """Replace hyphens in sentence data with spaces"""
    for sentence in sentences:
        for annotation in sentence["annotations"]:
            for result in annotation["result"]:
                if "value" in result and "text" in result["value"]:
                    result["value"]["text"] = re.sub(
                        r"(?<=\w)-(?=\w)", " ", result["value"]["text"]
                    )
        if "data" in sentence and "text" in sentence["data"]:
            sentence["data"]["text"] = re.sub(
                r"(?<=\w)-(?=\w)", " ", sentence["data"]["text"]
            )


rule all:
    input:
        "NER_output/aggregated_eval.png",
        expand("NER_output/{ENT}/overall_results.json", ENT=labels),


rule make_split:
    input:
        input_file,
    output:
        expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
    resources:
        **COMMON_RESOURCES,
    params:
        seed=config["seed"],
    run:
        seed = params.seed
        # Load JSON data once and reuse
        json_file = load_json_data(input_file)

        for label in labels:
            sentences = []
            ners = []
            for item in json_file:
                # Create a deep copy to avoid modifying the original data
                item_copy = copy.deepcopy(item)
                annotations = []
                indices_to_remove = []
                for a in item_copy["annotations"]:
                    for ind, r in enumerate(a["result"]):
                        try:
                            annotations.append(r["value"]["labels"][0])
                            if r["value"]["labels"][0] != label:
                                indices_to_remove.append(ind)
                        except (KeyError, IndexError):
                            pass
                if item_copy.get("annotations") and item_copy["annotations"]:
                    for index in sorted(indices_to_remove, reverse=True):
                        if len(item_copy["annotations"][0]["result"]) > index:
                            item_copy["annotations"][0]["result"].pop(index)
                sentences.append(item_copy)
                if label in annotations:
                    ners.append(1)
                else:
                    ners.append(0)
                    # count_positives = np.sum(ners)  # Unused variable
            t = list(zip(sentences, ners))
            sort = sorted(t, key=itemgetter(1))
            random.seed(seed)
            random.shuffle(sort)
            sentences, ners = zip(*sort)
            sentences = list(sentences)

            replace_hyphens_in_data(sentences)

            X_train, X_test_dev, _, y_test_dev = train_test_split(
                sentences,
                ners,
                test_size=test_size,
                random_state=params.seed,
                stratify=ners,
            )
            X_test, X_dev, _, _ = train_test_split(
                X_test_dev,
                y_test_dev,
                test_size=0.5,
                random_state=params.seed,
                stratify=y_test_dev,
            )
            sentence_split = (X_train, X_test, X_dev)
            for s, y in zip(model_sets, sentence_split):
                with open(f"NER/{label}/{s}.jsonls", "w") as f:
                    json.dump(list(y), f)
                    f.write("\n")


rule data_aug:
    input:
        input_file,
        expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
    output:
        expand("NER/{ENT}/{SET}.jsonla", ENT=labels, SET=model_sets),
    resources:
        **COMMON_RESOURCES,
    params:
        seed=config["seed"],
    run:
        # Use cached JSON data and strain catalog
        json_file = load_json_data(input_file)
        strain_catalog = extract_strain_catalog(json_file)

        for label in labels:
            if label == "STRAIN":
                for split in model_sets:
                    # Use more efficient file copying for STRAIN
                    import shutil

                    shutil.copy2(
                        f"NER/{label}/{split}.jsonls", f"NER/{label}/{split}.jsonla"
                    )
            else:
                for split in model_sets:
                    if split == "train":
                        with open(f"NER/{label}/{split}.jsonls") as infile:
                            data = json.load(infile)

                        items_without_annotations = []
                        items_with_annotations = []

                        for i in data:
                            has_valid_annotation = False
                            if i["annotations"]:
                                for j in i["annotations"]:
                                    if j["result"]:
                                        for result in j["result"]:
                                            if "value" in result:
                                                has_valid_annotation = True

                            if has_valid_annotation:
                                items_with_annotations.append(i)
                            else:
                                items_without_annotations.append(i)

                        no_annotations_count = len(items_without_annotations)
                        with_annotations_count = len(items_with_annotations)

                        def replace_entities_with_random(item, strain_catalog):
                            new_item = copy.deepcopy(item)

                            original_text = new_item["data"]["text"]
                            new_text = original_text
                            offset_adjustment = 0
                            replacement_made = False

                            # Search for substrings from strain_catalog in the text
                            for strain in strain_catalog:
                                if strain in new_text and not replacement_made:
                                    # Find the position of the strain in the text
                                    strain_pos = new_text.find(strain)

                                    # Replace the first occurrence of the strain with a random one from the catalog
                                    replacement = random.choice(strain_catalog)
                                    new_text = new_text.replace(strain, replacement, 1)

                                    # Calculate the offset adjustment (difference in length)
                                    offset_adjustment = len(replacement) - len(strain)

                                    # Update all annotation positions that come after the replacement
                                    if new_item.get("annotations"):
                                        for annotation in new_item["annotations"]:
                                            if annotation.get("result"):
                                                for result in annotation["result"]:
                                                    if (
                                                        "value" in result
                                                            and "start" in result["value"]
                                                            and "end" in result["value"]
                                                        ):
                                                            # If annotation starts after the replacement point, adjust its position
                                                            if (
                                                                result["value"]["start"]
                                                                > strain_pos
                                                            ):
                                                                result["value"][
                                                                "start"
                                                            ] += offset_adjustment
                                                                result["value"][
                                                                    "end"
                                                                ] += offset_adjustment
                                                    # If annotation contains the replacement point, adjust only the end
                                                    elif (
                                                        "value" in result
                                                            and "end" in result["value"]
                                                            and result["value"]["end"]
                                                            > strain_pos
                                                        ):
                                                        result["value"][
                                                            "end"
                                                        ] += offset_adjustment

                                    replacement_made = True
                                    break  # Only replace one entity and stop

                                    # Update the text in the item
                            new_item["data"]["text"] = new_text

                            return new_item
                            # Generate as many items as there are without annotations

                        num_to_generate = with_annotations_count
                        augmented_items = []
                        for _ in range(num_to_generate):
                            random_item = random.choice(items_with_annotations)
                            augmented_item = replace_entities_with_random(
                                random_item, strain_catalog
                            )
                            augmented_items.append(augmented_item)

                        all_items = []
                        all_items.extend(items_with_annotations)
                        all_items.extend(items_without_annotations)
                        all_items.extend(augmented_items)
                        random.shuffle(all_items)

                        with open(f"NER/{label}/{split}.jsonla", "w") as f:
                            json.dump(all_items, f, indent=2)

                    else:
                        # Copy the original file for non-train splits
                        with open(f"NER/{label}/{split}.jsonls") as infile:
                            with open(f"NER/{label}/{split}.jsonla", "w") as outfile:
                                for line in infile:
                                    outfile.write(line)


rule convert_splits:
    input:
        json=expand("NER/{ENT}/{SET}.jsonla", ENT=labels, SET=model_sets),
        config="label/config.xml",
    output:
        conll=expand("NER/{ENT}/{SET}.conll", ENT=labels, SET=model_sets),
    resources:
        slurm_partition="cpu",
        runtime=60,  # Increased from 10 to 60 minutes
        mem_mb=8000,
    shell:
        """
        for file in NER/**/*.jsonla; do
            output_dir="${{file%.jsonla}}"
            output_file="${{file%.jsonla}}.conll"
            label-studio-converter export -i "$file" -c {input.config} -f CONLL2003 -o "$output_dir"
            cat "$output_dir/result.conll" > "$output_file"
            rm -rf "$output_dir"
        done
        """


rule convert_to_bio:
    input:
        expand("NER/{ENT}/{SET}.conll", ENT=labels, SET=model_sets),
    output:
        expand("NER/{ENT}/{SET}.txt", ENT=labels, SET=model_sets),
    resources:
        **COMMON_RESOURCES,
    run:
        for label in labels:
            for split in model_sets:
                with open(f"NER/{label}/{split}.conll") as infile:
                    with open(f"NER/{label}/{split}.txt", "w") as outfile:
                        for line in infile:
                            if line.endswith(f"{label}\n"):
                                outfile.write(
                                    line[: -len(f"{label}") - 2] + "\n"
                                )  ## if the label is the last word, remove it and leave only the BIO tag

                            elif line.endswith("O\n"):
                                outfile.write(line)
                            elif line == "\n":
                                outfile.write(line)
                            else:
                                outfile.write(re.sub(r" [BI]-.*", " O", line))


rule convert_to_json:
    input:
        expand("NER/{ENT}/{SET}.txt", ENT=labels, SET=model_sets),
    output:
        expand("NER/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
    resources:
        slurm_partition="cpu",
        runtime=60,  # Increased from 10 to 60 minutes
        mem_mb=8000,
    shell:
        """
        for file in NER/**/*.txt; do
            output_file="${{file%.txt}}.json"
            python scripts/conll2003_to_jsonl.py "$file" "$output_file"
        done
        """


rule run_linkbert:
    input:
        expand("NER/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
    output:
        results=expand("NER_output/{ENT}/all_results.json", ENT=labels),
        predictions=expand("NER_output/{ENT}/test_predictions.txt", ENT=labels),
    conda:
        "envs/pytorch.yml"
    params:
        epochs=config["ner_epochs"],
        cuda=lambda w: ",".join([str(i) for i in cuda]),
        model_type=config["model"],
        entities=" ".join(labels),
    resources:
        slurm_partition="gpu_a100_il",
        slurm_extra="--gres=gpu:1",
        runtime=120,
        mem_mb=32000,
    shell:
        """
        export MODEL_PATH=michiyasunaga/BioLinkBERT-{params.model_type}
        export MODEL=BioLinkBERT-{params.model_type}
        export CUDA_VISIBLE_DEVICES={params.cuda}
        export EPOCHS={params.epochs}
        export TOKENIZERS_PARALLELISM=true
        # Detailed GPU check
        python -c "import torch; print(f'PyTorch version: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'CUDA version: {{torch.version.cuda if torch.cuda.is_available() else \"N/A\"}}'); print(f'Device count: {{torch.cuda.device_count() if torch.cuda.is_available() else 0}}')"
        nvidia-smi || echo "nvidia-smi not available"
        
        # Force exit if no GPU
        if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
            echo "ERROR: No GPU detected by PyTorch! Training would be extremely slow on CPU."
            echo "Please check PyTorch CUDA installation."
            exit 1
        fi
        
        for entity in {params.entities};
        do
            datadir=NER/$entity
            outdir=NER_output/$entity
            mkdir -p $outdir
            python3 -u scripts/run_ner.py --model_name_or_path $MODEL_PATH \
            --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
            --do_train --do_eval --do_predict --report_to none \
            --per_device_train_batch_size 64 --gradient_accumulation_steps 2 --fp16 \
            --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs $EPOCHS --max_seq_length 512 \
            --save_strategy epoch --eval_strategy epoch --logging_strategy epoch --save_total_limit 2 \
            --output_dir $outdir --overwrite_output_dir --load_best_model_at_end --metric_for_best_model eval_loss \
            |& tee $outdir/log.txt 
            rm -rf $outdir/checkpoint-*
        done
        """


rule run_nervaluate:
    input:
        predictions=expand("NER_output/{ENT}/test_predictions.txt", ENT=labels),
        ground_truth=expand("NER/{ENT}/test.txt", ENT=labels),
        config="config.yaml",
    output:
        overall=expand("NER_output/{ENT}/overall_results.json", ENT=labels),
        per_tag=expand("NER_output/{ENT}/results_per_tag.json", ENT=labels),
        comparison=expand("NER_output/{ENT}/comparison_report.json", ENT=labels),
    resources:
        **COMMON_RESOURCES,
    shell:
        """
        python scripts/run_nervaluate.py
        """


rule aggregate_data:
    input:
        expand("NER_output/{ENT}/all_results.json", ENT=labels),
    output:
        "NER_output/aggregated_eval.tsv",
    resources:
        slurm_partition="cpu",
        runtime=10,
        mem_mb=8000,
    run:
        dfs = []
        for label in labels:
            with open(f"NER_output/{label}/all_results.json", "r") as f:
                data = json.load(f)
            df = pd.DataFrame({label: data})
            dft = df.transpose()
            dfs.append(dft)
        full = pd.concat(dfs)
        full.to_csv(output[0], sep="\t")


rule plot:
    input:
        "NER_output/aggregated_eval.tsv",
    output:
        "NER_output/aggregated_eval.png",
    params:
        labels=labels,
    resources:
        slurm_partition="cpu",
        runtime=30,
        mem_mb=8000,
    script:
        "scripts/ner_plot_performance.py"
