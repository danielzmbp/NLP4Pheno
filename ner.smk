import pandas as pd
import os
import json
import numpy as np
import re
import random
from operator import itemgetter
from sklearn.model_selection import train_test_split


configfile: "config.yaml"


labels = config["ner_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]
test_size = config["ner_test"]


rule all:
    input:
        "NER_output/aggregated_eval.png",

# rule make_input_json:
#     resources:
#         slurm_partition="single",
#         runtime=10,
#     output:
#         input_file,
#     run:
#         "python scripts/convert_label.py"

rule make_split:
    input:
        input_file,
    output:
        expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=8000,
    params:
        seed=config["seed"],
    run:
        seed = params.seed
        for label in labels:
            with open(input_file) as f:
                json_file = json.load(f)
            sentences = []
            ners = []
            for item in json_file:
                annotations = []
                indices_to_remove = []
                for a in item["annotations"]:
                    for ind, r in enumerate(a["result"]):
                        try:
                            annotations.append(r["value"]["labels"][0])
                            if r["value"]["labels"][0] != label:
                                indices_to_remove.append(ind)
                        except:
                            pass
                for index in sorted(indices_to_remove, reverse=True):
                    item["annotations"][0]["result"].pop(index)
                sentences.append(item)
                if label in annotations:
                    ners.append(1)
                else:
                    ners.append(0)
            count_positives = np.sum(ners)
            t = list(zip(sentences, ners))
            sort = sorted(t, key=itemgetter(1))
            # get x times more negatives than positives
            # sort = sort[-(count_positives * 10) :]
            random.seed(seed)
            random.shuffle(sort)
            sentences, ners = zip(*sort)
            sentences = list(sentences)

            # replace all hyphens in the data by spaces
            for sentence in sentences:
                for annotation in sentence["annotations"]:
                    for result in annotation["result"]:
                        if "value" in result:
                            if "text" in result["value"]:
                                result["value"]["text"] = re.sub(r'(?<=\w)-(?=\w)', ' ', result["value"]["text"])
                if "data" in sentence:
                    if "text" in sentence["data"]:
                        sentence["data"]["text"] = re.sub(r'(?<=\w)-(?=\w)', ' ', sentence["data"]["text"])
            
            X_train, X_test_dev, _, y_test_dev = train_test_split(
                sentences, ners, test_size=test_size, random_state=1, stratify=ners
            )
            X_test, X_dev, _, _ = train_test_split(
                X_test_dev,
                y_test_dev,
                test_size=0.5,
                random_state=1,
                stratify=y_test_dev,
            )
            sentence_split = (X_train, X_test, X_dev)
            for s, y in zip(model_sets, sentence_split):
                with open(f"NER/{label}/{s}.jsonls", "w") as f:
                    json.dump(list(y), f)
                    f.write("\n")

# # this swaps the current entities randomly of the same kind
# rule data_aug:
#     input:
#         expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
#     output:
#         expand("NER/{ENT}/{SET}.jsonla", ENT=labels, SET=model_sets),
#     resources:
#         slurm_partition="single",
#         runtime=30,
#         mem_mb=8000,
#     params:
#         seed=config["seed"],
#     run:
#         for label in labels:
#             for split in model_sets:
#                 if split == "train":
#                     with open(f"NER/{label}/{split}.jsonls") as infile:
#                         data = json.load(infile)
                    
#                     items_without_annotations = []
#                     items_with_annotations = []
#                     all_texts = []

#                     for i in data:
#                         has_valid_annotation = False
#                         if i["annotations"]:
#                             for j in i["annotations"]:
#                                 if j["result"]:
#                                     for result in j["result"]:
#                                         if "value" in result:
#                                             has_valid_annotation = True
#                                             if "text" in result["value"]:
#                                                 all_texts.append(result["value"]["text"])
                        
#                         if has_valid_annotation:
#                             items_with_annotations.append(i)
#                         else:
#                             items_without_annotations.append(i)

#                     no_annotations_count = len(items_without_annotations)
#                     with_annotations_count = len(items_with_annotations)

#                     def replace_entities_with_random(item, all_texts):
#                         # Create a deep copy to avoid modifying the original
#                         new_item = copy.deepcopy(item)
                        
#                         # Get the original text
#                         original_text = new_item["data"]["text"]
#                         new_text = original_text
                        
#                         # Sort annotations by their position in reverse order (to avoid offset issues)
#                         entity_positions = []
#                         for annotation in new_item["annotations"]:
#                             for result in annotation["result"]:
#                                 if "value" in result and "start" in result["value"] and "end" in result["value"]:
#                                     entity_positions.append({
#                                         "start": result["value"]["start"],
#                                         "end": result["value"]["end"],
#                                         "text": result["value"]["text"],
#                                         "labels": result["value"]["labels"]
#                                     })
                        
#                         # Sort in reverse order (from end to start)
#                         entity_positions.sort(key=lambda x: x["start"], reverse=True)
                        
#                         # Replace each entity with a random one from all_texts
#                         for entity in entity_positions:
#                             # Choose a random replacement text
#                             replacement = random.choice(all_texts)
                            
#                             # Replace in the text
#                             new_text = new_text[:entity["start"]] + replacement + new_text[entity["end"]:]
                            
#                             # Update the annotation
#                             for annotation in new_item["annotations"]:
#                                 for result in annotation["result"]:
#                                     if "value" in result and "start" in result["value"] and "end" in result["value"]:
#                                         if result["value"]["start"] == entity["start"] and result["value"]["end"] == entity["end"]:
#                                             result["value"]["text"] = replacement
                        
#                         # Update the text in the item
#                         new_item["data"]["text"] = new_text
                        
#                         return new_item
#                     # Generate as many items as there are without annotations
#                     all_texts = list(set(all_texts))
#                     #num_to_generate = no_annotations_count - with_annotations_count 
#                     num_to_generate = with_annotations_count // 5
#                     augmented_items = []
#                     for _ in range(num_to_generate):
#                         random_item = random.choice(items_with_annotations)
#                         augmented_item = replace_entities_with_random(random_item, all_texts)
#                         augmented_items.append(augmented_item)

#                     all_items = []
#                     all_items.extend(items_with_annotations)
#                     all_items.extend(items_without_annotations)
#                     all_items.extend(augmented_items)
#                     random.shuffle(all_items) 

#                     with open(f"NER/{label}/{split}.jsonla", 'w') as f:
#                         json.dump(all_items, f, indent=2)


#                 else:
#                     # Copy the original file for non-train splits
#                     with open(f"NER/{label}/{split}.jsonls") as infile:
#                         with open(f"NER/{label}/{split}.jsonla", "w") as outfile:
#                             for line in infile:
#                                 outfile.write(line)


rule data_aug:
    input:
        input_file,
        expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
    output:
        expand("NER/{ENT}/{SET}.jsonla", ENT=labels, SET=model_sets),
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=8000,
    params:
        seed=config["seed"],
    run:
        with open(input_file) as f:
            json_file = json.load(f)
        strain_catalog = []
        for i in json_file:
            if i["annotations"]:
                for j in i["annotations"]:
                    if j["result"]:
                        for result in j["result"]:
                            if "value" in result and "labels" in result["value"]:
                                if result["value"]["labels"][0] == "STRAIN":
                                    strain_catalog.append(result["value"]["text"])
        strain_catalog = list(set(strain_catalog))

        for label in labels:
            if label == "STRAIN":
                for split in model_sets:
                    with open(f"NER/{label}/{split}.jsonls") as infile:
                        with open(f"NER/{label}/{split}.jsonla", "w") as outfile:
                            outfile.write(infile.read()) 
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
                            
                            # Search for substrings from strain_catalog in the text
                            for strain in strain_catalog:
                                if strain in new_text:
                                    # Replace the first occurrence of the strain with a random one from the catalog
                                    replacement = random.choice(strain_catalog)
                                    new_text = new_text.replace(strain, replacement, 1)
                                    break  # Only replace one entity and stop
                            
                            # Update the text in the item
                            new_item["data"]["text"] = new_text
                            
                            return new_item
                        # Generate as many items as there are without annotations
                        num_to_generate = no_annotations_count - with_annotations_count 
                        num_to_generate = with_annotations_count
                        augmented_items = []
                        for _ in range(num_to_generate):
                            random_item = random.choice(items_with_annotations)
                            augmented_item = replace_entities_with_random(random_item, strain_catalog)
                            augmented_items.append(augmented_item)

                        all_items = []
                        all_items.extend(items_with_annotations)
                        all_items.extend(items_without_annotations)
                        all_items.extend(augmented_items)
                        random.shuffle(all_items) 

                        with open(f"NER/{label}/{split}.jsonla", 'w') as f:
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
        config="config.xml",
    output:
        conll=expand("NER/{ENT}/{SET}.conll", ENT=labels, SET=model_sets),
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=8000,
    shell:
        """
        for f in NER/**/*.jsonla
        do label-studio-converter export -i $f -c {input.config} -f CONLL2003 -o ${{f%.jsonla}}
        cat ${{f%.jsonla}}/result.conll > ${{f%jsonla}}conll
        rm -rf ${{f%.jsonla}}
        done
        """


rule convert_to_bio:
    input:
        expand(
            "NER/{ENT}/{SET}.conll",
            ENT=labels,
            SET=model_sets,
        ),
    output:
        expand(
            "NER/{ENT}/{SET}.txt",
            ENT=labels,
            SET=model_sets,
        ),
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=8000,
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
        expand(
            "NER/{ENT}/{SET}.txt",
            ENT=labels,
            SET=model_sets,
        ),
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=8000,
    output:
        expand(
            "NER/{ENT}/{SET}.json",
            ENT=labels,
            SET=model_sets,
        ),
    shell:
        """
        for f in NER/**/*.txt
        do python scripts/conll2003_to_jsonl.py $f ${{f%.txt}}.json
        done
        """


rule run_linkbert:
    input:
        expand("NER/{ENT}/{SET}.json", ENT=labels, SET=model_sets),
    output:
        expand("NER_output/{ENT}/all_results.json", ENT=labels),
    conda:
        "l"
    params:
        epochs=config["ner_epochs"],
        cuda=lambda w: ",".join([str(i) for i in cuda]),
        model_type=config["model"],
    resources:
        slurm_partition="gpu_4",
        slurm_extra="--gres=gpu:1",
        runtime=500,
        mem_mb=32000,
    shell:
        """
        export WANDB_DISABLED=true
        export MODEL_PATH=michiyasunaga/BioLinkBERT-{params.model_type}
        export MODEL=BioLinkBERT-{params.model_type}
        export CUDA_VISIBLE_DEVICES={params.cuda}
        export EPOCHS={params.epochs}
        export TOKENIZERS_PARALLELISM=true
        for entity in {labels};
        do
            datadir=NER/$entity
            outdir=NER_output/$entity
            mkdir -p $outdir
            python3 -u scripts/run_ner.py --model_name_or_path $MODEL_PATH \
            --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
            --do_train --do_eval --do_predict \
            --per_device_train_batch_size 64 --gradient_accumulation_steps 2 --fp16 \
            --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs $EPOCHS --max_seq_length 512 \
            --save_strategy epoch --evaluation_strategy epoch --logging_strategy epoch --output_dir $outdir --overwrite_output_dir --load_best_model_at_end \
            |& tee $outdir/log.txt 
            rm -rf $outdir/checkpoint-*
        done
        """


rule aggregate_data:
    input:
        expand("NER_output/{ENT}/all_results.json", ENT=labels),
    output:
        "NER_output/aggregated_eval.tsv",
    resources:
        slurm_partition="single",
        runtime=50,
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
        slurm_partition="single",
        runtime=30,
        mem_mb=8000,
    script:
        "scripts/ner_plot_performance.py"
