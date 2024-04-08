from Bio import Entrez
import os
import pandas as pd
from tqdm import tqdm
import subprocess
import argparse
import shutil
from glob import glob
import numpy as np

Entrez.email = "d.gomez@lmu.de"
Entrez.api_key = "71c734bb92382389e17af918de877c12b308"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_assemblies",
    type=int,
    default=500,
    help="Maximum number of assemblies to download",
)
parser.add_argument("--data", type=int, help="Dataset value size")
parser.add_argument(
    "--word_size_limit", type=int, default=3, help="Assembly word name size limit"
)
parser.add_argument(
    "--min_samples", type=int, default=3, help="Assemblies to filter per strain"
)
args = parser.parse_args()

min_samples = args.min_samples
max_assemblies = args.max_assemblies
data = args.data
word_size_limit = args.word_size_limit
path = f"/home/gomez/gomez/assemblies/{data}/{max_assemblies}"

## Download assemblies
st = pd.read_parquet(f"../../gomez/preds{data}/REL_output/preds.parquet")

nostrains = set()
if os.path.exists("scripts/strain_cache.txt"):
    with open("scripts/strain_cache.txt", "r") as f:
        nostrains = set(line.strip() for line in f)
else:
    os.system("touch scripts/strain_cache.txt")
    nostrains = []


def get_assembly_summary(id):
    """Get esummary for an entrez id"""
    esummary_handle = Entrez.esummary(db="assembly", id=id, report="full")
    esummary_record = Entrez.read(esummary_handle, validate=False)
    return esummary_record


def get_assemblies(term, download=True, path=path):
    """
    Download genbank assemblies for a given search term.

    Args:
        term (str): The search term, usually the organism name.
        download (bool, optional): Whether to download the results. Defaults to True.
        path (str, optional): The folder to save the assemblies to. Defaults to '/home/gomez/gomez/assemblies/{max_assemblies}'.

    Returns:
        list: A list of FTP links to the downloaded assemblies.
    """
    # check if path exists
    if term not in nostrains:
        if not (os.path.exists(f'{path}/{term.replace(" ","_")}/')):
            handle = Entrez.esearch(
                db="assembly",
                term=f"({term} [ORGN]) AND (Bacteria [ORGN])",
                retmax=str(max_assemblies),
            )
            record = Entrez.read(handle, validate=False)
            ids = record["IdList"]
            if len(ids) == 0:
                if term not in nostrains:
                    with open("scripts/strain_cache.txt", "a") as f:
                        f.write(f"{term}\n")
            elif len(ids) < max_assemblies:
                links = []
                for id in tqdm(ids, desc="ids", position=1, leave=False):
                    # get summary
                    summary = get_assembly_summary(id)
                    # get ftp link
                    try:
                        url = summary["DocumentSummarySet"]["DocumentSummary"][0][
                            "FtpPath_GenBank"
                        ]
                        if url == "":
                            continue
                        label = os.path.basename(url)
                        # get the fasta link - change this to get other formats
                        link = os.path.join(url, label + "_genomic.fna.gz")
                        print(f"\n{term}->{id}")
                        links.append(link)
                        term = (
                            term.replace("/", "_")
                            .replace(")", "_")
                            .replace(":", "_")
                            .replace(" ", "_")
                            .replace("(", "_")
                            .replace("=", "_")
                            .replace("'", "_")
                            .replace(";", "_")
                            .replace(",", "_")
                            .replace("|", "_")
                            .replace(".", "_")
                            .replace("-", "_")
                            .replace("^", "_")
                            .replace("*", "_")
                            .replace('"', "_")
                            .replace("___", "_")
                            .replace("__", "_")
                            .lstrip("_")
                            .rstrip("_")
                        )
                        os.makedirs(f"{path}/{term}/{id}/", exist_ok=True)
                        if download == True:
                            if os.path.exists(f"{path}/{term}/{id}/{label}.fna.gz"):
                                continue
                            # download link
                            else:
                                subprocess.Popen(
                                    [
                                        "wget",
                                        "-q",
                                        "-O",
                                        f"{path}/{term}/{id}/{label}.fna.gz",
                                        link,
                                    ]
                                )
                    except:
                        pass
                return links
            else:
                print(f"\n {term} <- {len(ids)}")
        else:
            print(f"\n {term} <- already exists")
    else:
        print(f"\n {term} <- no results")


strains = list(st.word_strain_qc.value_counts().index)

ss = [i for i in strains if len(i) > word_size_limit]

ss.sort()

for i in tqdm(ss, desc="strains", position=0):
    try:
        get_assemblies(i)
    except:
        pass

## Filter assemblies
filtered_path = f"{path}_{min_samples}"
os.makedirs(filtered_path, exist_ok=True)

for strain in tqdm(os.listdir(path)):
    os.makedirs(f"{filtered_path}/{strain}", exist_ok=True)
    assemblies = os.listdir(f"{path}/{strain}")
    if len(assemblies) > max_assemblies:
        selected_assemblies = np.random.choice(
            assemblies, max_assemblies, replace=False
        )
        for assembly in selected_assemblies:
            assembly_path = f"{path}/{strain}/{assembly}"
            # Check file size in bytes
            if os.path.getsize(glob(f"{assembly_path}/*.fna.gz")[0]) < 5120:
                available_assemblies = [
                    a
                    for a in assemblies
                    if os.path.getsize(glob(f"{path}/{strain}/{a}/*.fna.gz")[0]) >= 5120
                ]
                if available_assemblies:
                    assembly = np.random.choice(available_assemblies)
                    assembly_path = f"{path}/{strain}/{assembly}"
                else:
                    continue
            shutil.copytree(
                assembly_path,
                f"{filtered_path}/{strain}/{assembly}",
                dirs_exist_ok=True,
            )
    else:
        for assembly in assemblies:
            assembly_path = f"{path}/{strain}/{assembly}"
            # Check file size in bytes
            if os.path.getsize(glob(f"{assembly_path}/*.fna.gz")[0]) < 5120:
                available_assemblies = [
                    a
                    for a in assemblies
                    if os.path.getsize(glob(f"{path}/{strain}/{a}/*.fna.gz")[0]) >= 5120
                ]
                if available_assemblies:
                    assembly = np.random.choice(available_assemblies)
                    assembly_path = f"{path}/{strain}/{assembly}"
                else:
                    continue
            shutil.copytree(
                assembly_path,
                f"{filtered_path}/{strain}/{assembly}",
                dirs_exist_ok=True,
            )
