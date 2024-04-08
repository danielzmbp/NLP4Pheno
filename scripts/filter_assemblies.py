import os
import argparse
import numpy as np
from tqdm import tqdm
import shutil
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_assemblies", type=int, default=5, help="Maximum number of assemblies"
)
parser.add_argument("--data", type=int, help="Data value")
parser.add_argument(
    "--original_path",
    type=str,
    default="/home/gomez/gomez/assemblies_linkbert_500",
    help="Original path of assemblies",
)

args = parser.parse_args()

max_assemblies = args.max_assemblies
data = args.data
original_path = args.original_path

filtered_path = f"{original_path}_filtered_{max_assemblies}"
os.makedirs(filtered_path, exist_ok=True)

for strain in tqdm(os.listdir(original_path)):
    os.makedirs(f"{filtered_path}/{strain}", exist_ok=True)
    assemblies = os.listdir(f"{original_path}/{strain}")
    if len(assemblies) > max_assemblies:
        selected_assemblies = np.random.choice(
            assemblies, max_assemblies, replace=False
        )
        for assembly in selected_assemblies:
            assembly_path = f"{original_path}/{strain}/{assembly}"
            # Check file size in bytes
            if os.path.getsize(glob(f"{assembly_path}/*.fna.gz")[0]) < 5120:
                available_assemblies = [
                    a
                    for a in assemblies
                    if os.path.getsize(
                        glob(f"{original_path}/{strain}/{a}/*.fna.gz")[0]
                    )
                    >= 5120
                ]
                if available_assemblies:
                    assembly = np.random.choice(available_assemblies)
                    assembly_path = f"{original_path}/{strain}/{assembly}"
                else:
                    continue
            shutil.copytree(
                assembly_path,
                f"{filtered_path}/{strain}/{assembly}",
                dirs_exist_ok=True,
            )
    else:
        for assembly in assemblies:
            assembly_path = f"{original_path}/{strain}/{assembly}"
            # Check file size in bytes
            if os.path.getsize(glob(f"{assembly_path}/*.fna.gz")[0]) < 5120:
                available_assemblies = [
                    a
                    for a in assemblies
                    if os.path.getsize(
                        glob(f"{original_path}/{strain}/{a}/*.fna.gz")[0]
                    )
                    >= 5120
                ]
                if available_assemblies:
                    assembly = np.random.choice(available_assemblies)
                    assembly_path = f"{original_path}/{strain}/{assembly}"
                else:
                    continue
            shutil.copytree(
                assembly_path,
                f"{filtered_path}/{strain}/{assembly}",
                dirs_exist_ok=True,
            )
