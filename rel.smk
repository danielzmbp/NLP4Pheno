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


rule all:
    input:
        "REL_output/all_metrics.png",


rule parse_rels:
    input:
        input_file,
    output:
        "REL/parsed_rels.txt",
    resources:
        slurm_partition="single",
        runtime=30,
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
            ENT=labels,
        ),
    resources:
        slurm_partition="single",
        runtime=30,
    run:
        df = pd.read_csv(input[0], sep="\t")
        for label in labels:
            relation = label.split(":")[0]
            df[df["label"].str.startswith(relation)].to_csv(
                f"REL/{label}/all.tsv", sep="\t", index=False
            )


rule split_sets:
    input:
        expand(
            "REL/{ENT}/all.tsv",
            ENT=labels,
        ),
    output:
        expand(
            "REL/{ENT}/{SET}.json",
            ENT=labels,
            SET=model_sets,
        ),
    resources:
        slurm_partition="single",
        runtime=30,
    params:
        seed=config["seed"],
    run:
        for label in labels:
            rel_label = label.split(":")[1]
            df = pd.read_csv(f"REL/{label}/all.tsv", sep="\t")

            df.rename(columns={"label": "l"}, inplace=True)

            df.loc[:, "label"] = np.where(df.l.str.endswith(rel_label), 1, 0)

            train, test_eval = train_test_split(
                df, test_size=test_size, stratify=df.label, random_state=params.seed
            )
            test, evaluation = train_test_split(
                test_eval,
                test_size=0.6,
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
        expand(
            "REL/{ENT}/{SET}.json",
            ENT=labels,
            SET=model_sets,
        ),
    output:
        expand(
            "REL_output/{ENT}/all_results.json",
            ENT=labels,
        ),
    conda:
        "l"
    params:
        epochs=config["rel_epochs"],
        cuda=lambda w: ",".join([str(i) for i in cuda]),
        model_type=config["model"],
    resources:
        slurm_partition="gpu_4",
        slurm_extra="--gres=gpu:1",
        runtime=500,
    shell:
        """
        export CUDA_VISIBLE_DEVICES={params.cuda}
        export MODEL=BioLinkBERT-{params.model_type}
        export MODEL_PATH=michiyasunaga/$MODEL
        for entity in {labels};
        do
            datadir=REL/$entity
            outdir=REL_output/$entity
            mkdir -p $outdir
            python3 -u scripts/run_seqcls.py --model_name_or_path $MODEL_PATH \
            --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
            --do_train --do_eval --do_predict --metric_name PRF1 \
            --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
            --learning_rate 3e-5 --num_train_epochs {params.epochs} --max_seq_length 384 \
            --save_strategy epoch --evaluation_strategy epoch --logging_strategy epoch --output_dir $outdir --overwrite_output_dir --load_best_model_at_end \
            --metric_for_best_model F1 --greater_is_better True \
            |& tee $outdir/log.txt
            rm -rf $outdir/checkpoint-*
        done
        """


rule join_metrics:
    input:
        mets=expand(
            "REL_output/{ENT}/all_results.json",
            ENT=labels,
        ),
    output:
        "REL_output/all_metrics.tsv",
    resources:
        slurm_partition="single",
        runtime=30,
    run:
        import json
        import pandas as pd

        dfs = []
        for f in input.mets:
            with open(f, "r") as file:
                data = json.load(file)
            relation = f.split("/")[1]  # f.rsplit("-", 1)[0].split("/")[1]
            df = pd.DataFrame({relation: data})
            dft = df.transpose()
            dfs.append(dft)
        result_df = pd.concat(dfs)
        result_df.to_csv(output[0], sep="\t")


rule plot_metrics:
    input:
        "REL_output/all_metrics.tsv",
    output:
        "REL_output/all_metrics.png",
    params:
        labels=labels,
    resources:
        slurm_partition="single",
        runtime=30,
    script:
        "scripts/rel_plot_performance.py"
