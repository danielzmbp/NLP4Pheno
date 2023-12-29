import pandas as pd
import os
import json
import numpy as np
import re
import random
from operator import itemgetter
from sklearn.model_selection import train_test_split


configfile: "config.yaml"


labels_flat = config["ner_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]
test_size = config["test_size"]


rule all:
    input:
        "NER_output/aggregated_eval.csv",


rule make_split:
    input:
        input_file,
    output:
        expand("NER/{ENT}/{SET}.jsonls", ENT=labels_flat, SET=model_sets[:3]),
    run:
        json_file = json.load(open(input[0]))
        for label in labels_flat:
            sentences = []
            labels = []
            for item in json_file:
                annotations = []
                for a in item["annotations"]:
                    for r in a["result"]:
                        try:
                            annotations.append(r["value"]["labels"][0])
                        except:
                            pass
                sentences.append(item)
                if label in annotations:
                    labels.append(1)
                else:
                    labels.append(0)
            count_positives = np.sum(labels)
            t = list(zip(sentences, labels))
            sort = sorted(t, key=itemgetter(1))
            # get x times more negatives than positives
            # sort = sort[-(count_positives * 10) :]
            random.seed(42)
            random.shuffle(sort)
            sentences, labels = zip(*sort)
            sentences = list(sentences)
            X_train, X_test_dev, _, y_test_dev = train_test_split(
                sentences, labels, test_size=test_size, random_state=42, stratify=labels
            )
            X_test, X_dev, _, _ = train_test_split(
                X_test_dev,
                y_test_dev,
                test_size=0.5,
                random_state=42,
                stratify=y_test_dev,
            )
            sentence_split = (X_train, X_test, X_dev)
            for s, y in zip(model_sets, sentence_split):
                with open(f"NER/{label}/{s}.jsonls", "w") as f:
                    json.dump(list(y), f)
                    f.write("\n")



rule convert_splits:
    input:
        json=expand("NER/{ENT}/{SET}.jsonls", ENT=labels_flat, SET=model_sets),
        config="/home/gomez/biobert-pytorch/my_dataset/config.xml",
    output:
        conll=temp(expand("NER/{ENT}/{SET}.conll", ENT=labels_flat, SET=model_sets))
    shell:
        """
        for f in NER/**/*.jsonls
        do label-studio-converter export -i $f -c {input.config} -f CONLL2003 -o ${{f%.jsonls}}
        cat ${{f%.jsonls}}/result.conll > ${{f%jsonls}}conll
        rm -rf ${{f%.jsonls}}
        done
        """


rule convert_to_bio:
    input:
        expand(
            "NER/{ENT}/{SET}.conll",
            ENT=labels_flat,
            SET=model_sets,
        ),
    output:
        temp(expand(
            "NER/{ENT}/{SET}.txt",
            ENT=labels_flat,
            SET=model_sets,
        )),
    run:
        for label in labels_flat:
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
            ENT=labels_flat,
            SET=model_sets,
        ),
    output:
        expand(
            "NER/{ENT}/{SET}.json",
            ENT=labels_flat,
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
        expand("NER/{ENT}/{SET}.json", ENT=labels_flat, SET=model_sets),
    output:
        expand("NER_output/{ENT}/all_results.json", ENT=labels_flat),
    conda:
        "linkbert"
    params:
        epochs=config["epochs"],
        cuda=lambda w: ",".join([str(i) for i in cuda]),
    shell:
        """
        export MODEL=BioLinkBERT-large
        export MODEL_PATH=michiyasunaga/$MODEL
        export CUDA_VISIBLE_DEVICES={params.cuda}
        export EPOCHS={params.epochs}
        export TOKENIZERS_PARALLELISM=true
        for entity in {labels_flat};
        do
            datadir=NER/$entity
            outdir=NER_output/$entity
            mkdir -p $outdir
            python3 -u scripts/run_ner.py --model_name_or_path $MODEL_PATH \
            --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
            --do_train --do_eval --do_predict \
            --per_device_train_batch_size 32 --gradient_accumulation_steps 2 --fp16 \
            --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs $EPOCHS --max_seq_length 512 \
            --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
            |& tee $outdir/log.txt 
        done
        """


rule aggregate_data:
    input:
        expand("NER_output/{ENT}/all_results.json", ENT=labels_flat),
    output:
        "NER_output/aggregated_eval.csv",
    run:
        dfs = []
        for label in labels_flat:
            df = pd.read_json(
                f"NER_output/{label}/all_results.json",
            )
            df.loc[:, "label"] = label
            dfs.append(df)
        full = pd.concat(dfs, axis=0)
        # full.rename(columns={0: "metric", 1: "value"}, inplace=True)
        # m = full.pivot(
        #     index=["metric", "group"], values="value", columns="label"
        # ).reset_index()
        full.to_csv(output[0], index=False)


rule plot:
    input:
        "NER_output/aggregated_eval.csv",
    output:
        "NER_output/aggregated_eval.png",
    params:
        labels=labels_flat,
    script:
        "scripts/ner_plot_performance.py"
