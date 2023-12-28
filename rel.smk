import pandas as pd
import os
import json
import numpy as np
import re
from itertools import permutations
from sklearn.model_selection import train_test_split


configfile: "config.yaml"


labels_flat = config["rel_labels"]
model_sets = config["model_sets"]
input_file = config["input_file"]
cuda = config["cuda_devices"]


rule all:
    input:
        "REL_output/all_metrics.png",


rule parse_rels:
    input:
        input_file,
    output:
        "REL/parsed_rels.txt",
    run:
        data = json.load(open(input[0]))
        ners = []
        rels = []
        for sentence in data:
            for annotation in sentence["annotations"][0]["result"]:
                if annotation["type"] == "labels":
                    df = pd.json_normalize(annotation)
                    df["sentence_id"] = sentence["id"]
                    ners.append(df)
                elif annotation["type"] == "relation":
                    df = pd.json_normalize(annotation)
                    df["sentence_id"] = sentence["id"]
                    rels.append(df)
        ner = pd.concat(ners)
        rel = pd.concat(rels)

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
                try:
                    rel_label = str(
                        rel[
                            (rel["from_id"] == perm[0]) & (rel["to_id"] == perm[1])
                        ].labels.values[0]
                    )
                except:
                    rel_label = ""
                label = f"{lab0[2:-2]}-{lab1[2:-2]}:{rel_label[2:-2]}"
                labels.append(label)
        pd.DataFrame({"sentence": sentences, "label": labels}).to_csv(
            output[0], sep="\t", index=False
        )


rule split_labels:
    input:
        "REL/parsed_rels.txt",
    output:
        expand(
            "REL/{ENT}/all.tsv",
            ENT=labels_flat,
        ),
    run:
        df = pd.read_csv(input[0], sep="\t")
        for label in labels_flat:
            relation = label.split(":")[0]
            df[df["label"].str.startswith(relation)].to_csv(
                f"REL/{label}/all.tsv", sep="\t", index=False
            )


rule split_sets:
    input:
        expand(
            "REL/{ENT}/all.tsv",
            ENT=labels_flat,
        ),
    output:
        expand(
            "REL/{ENT}/{SPLIT}/{SET}.tsv",
            ENT=labels_flat,
            SET=model_sets,
            SPLIT=splits,
        ),
    run:
        

        for label in labels_flat:
            rel_label = label.split(":")[1]
            df = pd.read_csv(f"REL/{label}/all.tsv", sep="\t")

            df.rename(columns={"label": "l"}, inplace=True)

            df.loc[:, "label"] = np.where(df.l.str.endswith(rel_label), 1, 0)

            train, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)

            test_data = test.reset_index().drop(columns="index").reset_index()[["index", "sentence", "label"]]
            test_original_data = test.reset_index().drop(columns="index").reset_index()[["index", "sentence", "label"]]
            train_data = train.reset_index().drop(columns="index").reset_index()[["index", "sentence", "label"]]

            with jsonlines.open(f"REL/{label}/test.jsonl", mode='w') as writer:
                for row in test_data.itertuples(index=False):
                    writer.write({"id": row[0], "sentence": row[1], "label": row[2]})

            with jsonlines.open(f"REL/{label}/test_original.jsonl", mode='w') as writer:
                for row in test_original_data.itertuples(index=False):
                    writer.write({"id": row[0], "sentence": row[1], "label": row[2]})

            with jsonlines.open(f"REL/{label}/train.jsonl", mode='w') as writer:
                for row in train_data.itertuples(index=False):
                    writer.write({"id": row[0], "sentence": row[1], "label": row[2]})


rule run_biobert:
    input:
        expand(
            "REL/{ENT}/{SPLIT}/{SET}.tsv",
            ENT=labels_flat,
            SET=model_sets,
            SPLIT=splits,
        ),
    output:
        expand(
            "REL_output/{ENT}-{SPLIT}/test_results.txt",
            ENT=labels_flat,
            SPLIT=splits,
        ),
    conda:
        "bb3"
    params:
        epochs=config["epochs"],
        cuda=lambda w: ",".join([str(i) for i in cuda]),
    shell:
        """
        export CUDA_VISIBLE_DEVICES={params.cuda}
        export SAVE_DIR=./REL_output

        export MAX_LENGTH=384
        export BATCH_SIZE=32
        export NUM_EPOCHS={params.epochs}
        export SAVE_STEPS=1000
        export SEED=1
        
        for entity in {labels_flat};
        do
        for split in {{1..{nsplits}}}
            do
            export DATA_DIR=REL/${{entity}}/${{split}}
                datadir=REL/$entity
                outdir=runs/$entity/$MODEL
                mkdir -p $outdir
                python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
                --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
                --do_train --do_eval --do_predict --metric_name PRF1 \
                --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 \
                --learning_rate 3e-5 --num_train_epochs 10 --max_seq_length 256 \
                --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
                |& tee $outdir/log.txt
            done
        done
        """


rule get_metrics:
    input:
        results=expand(
            "REL_output/{ENT}-{SPLIT}/test_results.txt",
            ENT=labels_flat,
            SPLIT=splits,
        ),
        test_original=expand(
            "REL/{ENT}/{SPLIT}/{SET}.tsv",
            ENT=labels_flat,
            SET=["test_original"],
            SPLIT=splits,
        ),
    output:
        expand(
            "REL_output/{ENT}-{SPLIT}/metrics.txt",
            ENT=labels_flat,
            SPLIT=splits,
        ),
    shell:
        """
        export SAVE_DIR=./REL_output
        for ENTITY in {labels_flat};
        do
            for SPLIT in {{1..{nsplits}}}
            do
                export DATA_DIR=REL/${{ENTITY}}/${{SPLIT}}
                python ./scripts/re_eval.py --output_path=${{SAVE_DIR}}/${{ENTITY}}-${{SPLIT}}/test_results.txt --answer_path=${{DATA_DIR}}/test_original.tsv > ${{SAVE_DIR}}/${{ENTITY}}-${{SPLIT}}/metrics.txt
            done
        done
        """


rule join_metrics:
    input:
        mets=expand(
            "REL_output/{ENT}-{SPLIT}/metrics.txt",
            ENT=labels_flat,
            SPLIT=splits,
        ),
    output:
        "REL_output/all_metrics.txt",
    run:
        dfs = []
        for f in input.mets:
            print(f)
            df = pd.read_csv(f, sep="\s+:\s", engine="python", header=None)
            df.rename(columns={0: "metric", 1: "score"}, inplace=True)
            df = df.replace("%", "", regex=True)
            df.loc[:, "score"] = df.score.astype(float) * 0.01
            df.loc[:, "relation"] = f.rsplit("-", 1)[0].split("/")[1]
            df.loc[:, "split"] = f.split("-")[-1].split("/")[0]
            dfs.append(df)
        pd.concat(dfs).to_csv(output[0], sep="\t", index=False)


rule plot_metrics:
    input:
        "REL_output/all_metrics.txt",
    output:
        "REL_output/all_metrics.png",
        "REL_output/best_splits.txt",
    params:
        labels=labels_flat,
    script:
        "scripts/rel_plot_performance.py"
