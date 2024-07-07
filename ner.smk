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


rule convert_splits:
    input:
        json=expand("NER/{ENT}/{SET}.jsonls", ENT=labels, SET=model_sets),
        config="config.xml",
    output:
        conll=expand("NER/{ENT}/{SET}.conll", ENT=labels, SET=model_sets),
    resources:
        slurm_partition="single",
        runtime=30,
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
        runtime=250,
    shell:
        """
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
    script:
        "scripts/ner_plot_performance.py"
