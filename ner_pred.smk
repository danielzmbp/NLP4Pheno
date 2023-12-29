import pandas as pd
import os
import json
import numpy as np
import re
from operator import itemgetter
from glob import glob
import itertools
import csv


configfile: "config.yaml"


labels_flat = config["ner_labels"][1:]
cutoff = config["cutoff_prediction"]
cuda = config["cuda_devices"]
corpus = "corpus" + config["dataset"]
preds = "/home/gomez/gomez/preds" + config["dataset"]


(PARTS,) = glob_wildcards(corpus + "/{part}.txt")


rule all:
    input:
        preds + "/NER_output/preds.parquet",


rule make_strain_file:
    output:
        preds + "/NER_output/device_strain.txt",
    run:
        # add device to each strain
        device = cuda
        dev = [str(x) for x in device]
        models = [x + " " + y for x, y in zip(itertools.cycle(dev), PARTS)]
        with open(output[0], "w") as f:
            for i in models:
                f.write(f"{i}\n")


rule run_strain_prediction:
    input:
        expand(corpus + "/{p}.txt", p=PARTS),
        dev=preds + "/NER_output/device_strain.txt",
    output:
        expand(preds + "/NER_output/STRAIN/{p}.parquet", p=PARTS),
    conda:
        "torch"
    shell:
        """
        for i in {cuda};do
            while read -r d m; do
            if [ ${{d}} -eq ${{i}} ]; then
                export MODEL=NER_output/STRAIN
                python scripts/ner_prediction_corpus.py --model $MODEL --device ${{d}} --output {preds}/$MODEL/$m.parquet --corpus {corpus}/$m.txt
            fi
            done < {input.dev} &
        done
        wait
        """


rule merge_strain_predictions:
    input:
        expand(preds + "/NER_output/STRAIN/{p}.parquet", p=PARTS),
    output:
        preds + "/NER_output/STRAIN/strains.parquet",
    run:
        l = []
        for file in glob(preds + "/NER_output/STRAIN/*.parquet"):
            d = pd.read_parquet(file).explode("ner").dropna()

            singletons = d.drop_duplicates(
                "text", keep=False
            )  # sentences appearing once

            dups = (
                d[d.index.isin(singletons.index) == False]
                .reset_index()
                .drop(columns="index")
            )  # sentences appearing more than once
            # this part is to fix the problem of splitting tokens
            indeces = []
            # find indeces of duplicates
            for i in range(dups.shape[0] - 1):
                if dups.iloc[i, 0] == dups.iloc[i + 1, 0]:
                    if (dups.iloc[i, 1]["end"] == dups.iloc[i + 1, 1]["start"]) or (
                        dups.iloc[i, 1]["end"] + 1 == dups.iloc[i + 1, 1]["start"]
                    ):
                        indeces.append(i)
                        indeces.append(i + 1)
            split = dups.iloc[indeces, :]
            dups_fixed = []
            # join split tokens from the same sentence (start_i = end_i+1)
            for i in range(split.shape[0] - 1):
                if dups.iloc[i, 0] == dups.iloc[i + 1, 0]:  # same sentence
                    if (dups.iloc[i, 1]["end"] == dups.iloc[i + 1, 1]["start"]) or (
                        dups.iloc[i, 1]["end"] + 1 == dups.iloc[i + 1, 1]["start"]
                    ):
                        start = dups.iloc[i, 1]["start"]
                        end = dups.iloc[i + 1, 1]["end"]
                        text = dups.iloc[i, 0]
                        word = text[start:end].lower()
                        score = np.max(
                            [dups.iloc[i, 1]["score"], dups.iloc[i + 1, 1]["score"]]
                        )
                        temporary = pd.DataFrame(
                            {
                                "text": text,
                                "start": start,
                                "end": end,
                                "word": word,
                                "score": score,
                            },
                            index=[0],
                        )
                        dups_fixed.append(temporary)
            c = pd.concat([dups.drop(index=indeces, axis=0), singletons])
            c = pd.concat([d.drop(columns="ner"), d.ner.apply(pd.Series)], axis=1)
            c.loc[:, "ner"] = file.split("/")[-2]
            c.drop(columns="entity_group", inplace=True)
            if len(dups_fixed) > 0:
                output_df = pd.concat([c, pd.concat(dups_fixed)])
            else:
                output_df = c
            l.append(c)

        df = pd.concat(l)

        df[df["score"] > cutoff].to_parquet(output[0])


rule make_sentence_file:
    input:
        preds + "/NER_output/STRAIN/strains.parquet",
    output:
        preds + "/NER_output/strains.txt",
        preds + "/NER_output/device_models.txt",
    run:
        df = pd.read_parquet(input[0])
        df.drop_duplicates(subset="text")["text"].to_csv(
            output[0], sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE
        )
        # add device to each strain
        dev = [str(x) for x in cuda]
        models = [x + " " + y for x, y in zip(itertools.cycle(dev), labels_flat)]
        with open(output[1], "w") as f:
            for i in models:
                f.write(f"{i}\n")


rule run_all_models:
    input:
        preds + "/NER_output/strains.txt",
        preds + "/NER_output/device_models.txt",
    output:
        expand(preds + "/NER_output/{l}.parquet", l=labels_flat),
    conda:
        "torch"
    shell:
        """
        for i in {cuda};do
            while read -r d m; do
            if [ ${{d}} -eq ${{i}} ]; then
                export MODEL=NER_output/${{m}}
                python scripts/ner_prediction_corpus.py --model $MODEL --device ${{d}} --output {preds}/$MODEL.parquet --corpus {input[0]} 
            fi
            done < {input[1]} &
        done
        wait
        """


rule agg_model_results:
    input:
        expand(preds + "/NER_output/{l}.parquet", l=labels_flat),
    output:
        preds + "/NER_output/strain_preds.parquet",
    run:
        l = []
        for ner in input:
            d = pd.read_parquet(ner).explode("ner").dropna()
            singletons = d.drop_duplicates(
                "text", keep=False
            )  # sentences appearing once
            dups = (
                d[d.index.isin(singletons.index) == False]
                .reset_index()
                .drop(columns="index")
            )  # sentences appearing more than once

            # this part is to fix the problem of splitting tokens
            indeces = []
            # find indeces of duplicates
            for i in range(dups.shape[0] - 1):
                if dups.iloc[i, 0] == dups.iloc[i + 1, 0]:
                    if (dups.iloc[i, 1]["end"] == dups.iloc[i + 1, 1]["start"]) or (
                        dups.iloc[i, 1]["end"] + 1 == dups.iloc[i + 1, 1]["start"]
                    ):
                        indeces.append(i)
                        indeces.append(i + 1)
            split = dups.iloc[indeces, :]
            dups_fixed = []
            # join split tokens from the same sentence (start_i = end_i+1)
            for i in range(split.shape[0] - 1):
                if dups.iloc[i, 0] == dups.iloc[i + 1, 0]:  # same sentence
                    if (dups.iloc[i, 1]["end"] == dups.iloc[i + 1, 1]["start"]) or (
                        dups.iloc[i, 1]["end"] + 1 == dups.iloc[i + 1, 1]["start"]
                    ):
                        start = dups.iloc[i, 1]["start"]
                        end = dups.iloc[i + 1, 1]["end"]
                        text = dups.iloc[i, 0]
                        word = text[start:end].lower()
                        score = np.max(
                            [dups.iloc[i, 1]["score"], dups.iloc[i + 1, 1]["score"]]
                        )
                        temporary = pd.DataFrame(
                            {
                                "text": text,
                                "start": start,
                                "end": end,
                                "word": word,
                                "score": score,
                            },
                            index=[0],
                        )
                        dups_fixed.append(temporary)
            c = pd.concat([dups.drop(index=indeces, axis=0), singletons])
            c = pd.concat([c.drop(columns="ner"), c.ner.apply(pd.Series)], axis=1)
            c.drop(columns="entity_group", inplace=True)
            if len(dups_fixed) > 0:
                output_df = pd.concat([c, pd.concat(dups_fixed)])
            else:
                output_df = c
            output_df.loc[:, "ner"] = ner.split("/")[-1].split(".")[0]
            l.append(output_df)
        df = pd.concat(l)

        df.to_parquet(output[0])


rule merge_preds:
    input:
        preds + "/NER_output/STRAIN/strains.parquet",
        preds + "/NER_output/strain_preds.parquet",
    output:
        preds + "/NER_output/preds.parquet",
    run:
        strains = pd.read_parquet(input[0])
        others = pd.read_parquet(input[1])
        others = others[others["score"] > cutoff]

        strains.drop(columns="ner", inplace=True)
        suff = strains.iloc[:, 1:].add_suffix("_strain")
        strains = pd.concat([strains.iloc[:, 0], suff], axis=1)
        df = strains.merge(others, on="text", how="left")
        df = df.dropna(subset="ner")
        df.to_parquet(output[0])
