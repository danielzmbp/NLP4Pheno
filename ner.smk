import pandas as pd
import os
import json
import numpy as np
import random
import copy
import sys
from operator import itemgetter

sys.path.append("scripts")
from annotation_utils import load_annotations, merged_pmc_groups, source_group
from ner_data import build_dataset
from split_utils import three_way_group_split


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


labels = config["ner_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]
test_size = config["ner_test"]
annotation_pmc_matches_file = config.get("annotation_pmc_matches_file")
split_inputs = [input_file] + (
    [annotation_pmc_matches_file] if annotation_pmc_matches_file else []
)
CPU_PARTITION = config.get("slurm_cpu_partition", "cpu")
GPU_PARTITION = config.get(
    "slurm_gpu_partition", "gpu_a100_il,gpu_h100_il,gpu_a100"
)
GPU_GRES = config.get("slurm_gpu_gres", "gpu:1")
PRETRAINED_MODEL = config.get("pretrained_model_path") or (
    f"michiyasunaga/BioLinkBERT-{config['model']}"
)

# Common resource configuration
COMMON_RESOURCES = {
    "slurm_partition": CPU_PARTITION,
    "runtime": 10,
    "mem_mb": 8000,
}

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
    STRAIN_CATALOG = sorted(strain_catalog)
    return STRAIN_CATALOG


rule all:
    input:
        "NER_output/aggregated_eval.png",
        expand("NER_output/{ENT}/overall_results.json", ENT=labels),


rule make_split:
    input:
        split_inputs,
    output:
        datasets=expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
        summary="NER/split_summary.json",
    resources:
        **COMMON_RESOURCES,
    params:
        seed=config["seed"],
    run:
        seed = params.seed
        summary = {}
        # Load JSON data once and reuse
        json_file = load_json_data(input_file)
        pmc_groups = merged_pmc_groups(json_file, annotation_pmc_matches_file)

        for label in labels:
            sentences = []
            ners = []
            for item in json_file:
                # Create a deep copy to avoid modifying the original data
                item_copy = copy.deepcopy(item)
                annotations = []
                for annotation in item_copy.get("annotations", []):
                    retained_results = []
                    for result in annotation.get("result", []):
                        if result.get("type") != "labels":
                            continue
                        result_labels = result.get("value", {}).get("labels", [])
                        annotations.extend(result_labels)
                        if label in result_labels:
                            retained_results.append(result)
                    annotation["result"] = retained_results
                sentences.append(item_copy)
                if label in annotations:
                    ners.append(1)
                else:
                    ners.append(0)
                    # count_positives = np.sum(ners)  # Unused variable
            source_groups = [
                source_group(item.get("id"), pmc_groups) for item in sentences
            ]
            split_indices = three_way_group_split(
                ners,
                source_groups,
                test_and_dev_size=test_size,
                seed=params.seed,
            )
            sentence_split = tuple(
                [sentences[index] for index in split_indices[split]]
                for split in model_sets
            )
            label_split = tuple(
                [ners[index] for index in split_indices[split]]
                for split in model_sets
            )
            task_sets = [set(item.get("id") for item in split) for split in sentence_split]
            source_group_sets = [
                {source_group(item.get("id"), pmc_groups) for item in split}
                for split in sentence_split
            ]
            if (
                task_sets[0] & task_sets[1]
                or task_sets[0] & task_sets[2]
                or task_sets[1] & task_sets[2]
            ):
                raise RuntimeError(f"Task leakage detected while splitting {label}")
            if (
                source_group_sets[0] & source_group_sets[1]
                or source_group_sets[0] & source_group_sets[2]
                or source_group_sets[1] & source_group_sets[2]
            ):
                raise RuntimeError(f"Source-article leakage detected while splitting {label}")
            summary[label] = {}
            for s, y, split_labels in zip(model_sets, sentence_split, label_split):
                summary[label][s] = {
                    "tasks": len(y),
                    "groups": len({source_group(item.get("id"), pmc_groups) for item in y}),
                    "pmc_linked_tasks": sum(str(item.get("id")) in pmc_groups for item in y),
                    "positive_tasks": int(sum(split_labels)),
                    "negative_tasks": int(len(split_labels) - sum(split_labels)),
                }
                with open(f"NER/{label}/{s}.jsonls", "w") as f:
                    json.dump(list(y), f)
                    f.write("\n")
        with open(output.summary, "w") as handle:
            json.dump(
                {
                    "grouping": "unique_pmcid_else_task_id",
                    "pmc_linked_tasks": len(pmc_groups),
                    "labels": summary,
                },
                handle,
                indent=2,
                sort_keys=True,
            )


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
        random.seed(params.seed)
        json_file = load_json_data(input_file)
        strain_catalog = extract_strain_catalog(json_file)
        normalized_strain_catalog = sorted(set(strain_catalog))
        strain_spans_by_task = {}
        for task in json_file:
            spans = []
            for annotation in task.get("annotations", []):
                for result in annotation.get("result", []):
                    value = result.get("value", {})
                    if "STRAIN" in value.get("labels", []):
                        spans.append((int(value["start"]), int(value["end"])))
            strain_spans_by_task[str(task.get("id"))] = spans

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

                        def replace_entities_with_random(item):
                            new_item = copy.deepcopy(item)
                            target_spans = [
                                (result["value"]["start"], result["value"]["end"])
                                for annotation in new_item.get("annotations", [])
                                for result in annotation.get("result", [])
                                if "value" in result
                            ]
                            candidates = [
                                span
                                for span in strain_spans_by_task.get(str(new_item.get("id")), [])
                                if all(
                                    span[1] <= target_start or span[0] >= target_end
                                    for target_start, target_end in target_spans
                                )
                            ]
                            if not candidates:
                                return None
                            strain_start, strain_end = random.choice(candidates)
                            text = new_item["data"]["text"]
                            original_strain = text[strain_start:strain_end]
                            alternatives = [
                                strain
                                for strain in normalized_strain_catalog
                                if strain != original_strain
                            ]
                            if not alternatives:
                                return None
                            replacement = random.choice(alternatives)
                            adjustment = len(replacement) - len(original_strain)
                            new_item["data"]["text"] = (
                                text[:strain_start] + replacement + text[strain_end:]
                            )
                            for annotation in new_item.get("annotations", []):
                                for result in annotation.get("result", []):
                                    value = result.get("value", {})
                                    if value.get("start", -1) >= strain_end:
                                        value["start"] += adjustment
                                        value["end"] += adjustment
                            new_item["id"] = f"{new_item.get('id')}:aug:{label}"
                            return new_item

                        augmented_items = []
                        for random_item in items_with_annotations:
                            augmented_item = replace_entities_with_random(random_item)
                            if augmented_item is not None:
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


rule build_ner_datasets:
    input:
        expand("NER/{ENT}/{SET}.jsonla", ENT=labels, SET=model_sets),
    output:
        json=expand("NER/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
        bio=expand("NER/{ENT}/{SET}.txt", ENT=labels, SET=model_sets),
        summary="NER/dataset_summary.json",
    resources:
        **COMMON_RESOURCES,
    run:
        stats = {}
        for label in labels:
            stats[label] = {}
            for split in model_sets:
                stats[label][split] = build_dataset(
                    f"NER/{label}/{split}.jsonla",
                    label=label,
                    json_output=f"NER/{label}/{split}.json",
                    bio_output=f"NER/{label}/{split}.txt",
                )
        with open(output.summary, "w") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)


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
        model_path=PRETRAINED_MODEL,
        entities=" ".join(labels),
    resources:
        slurm_partition=GPU_PARTITION,
        gres=GPU_GRES,
        runtime=int(config.get("ner_training_runtime", 120)),
        mem_mb=32000,
        cpus_per_task=4,
    shell:
        """
        export MODEL_PATH="{params.model_path}"
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
            2>&1 | tee $outdir/log.txt
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
        slurm_partition=CPU_PARTITION,
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
        slurm_partition=CPU_PARTITION,
        runtime=30,
        mem_mb=8000,
    script:
        "scripts/ner_plot_performance.py"
