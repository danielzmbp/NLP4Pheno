"""
This script filters assemblies based on the number of assemblies per strain by randomly selecting a subset of assemblies for each strain.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--max_assemblies", type=int, default=5,
                    help="Maximum number of assemblies")
parser.add_argument("--data", type=int, help="Data value")

args = parser.parse_args()

max_assemblies = args.max_assemblies
data = args.data

path = '/home/gomez/gomez/assemblies_linkbert_500'

os.system(f"mkdir -p {path}_filtered_{max_assemblies}")

for strain in os.listdir(path):
    os.system(f"mkdir -p {path}_filtered_{max_assemblies}/{strain}")
    assemblies = os.listdir(f"{path}/{strain}")
    if len(assemblies) > max_assemblies:
        selected_assemblies = np.random.choice(
            assemblies, max_assemblies, replace=False)
        for assembly in selected_assemblies:
            os.system(
                f"cp {path}/{strain}/{assembly} {path}_filtered_{max_assemblies}/{strain}/{assembly}")
    else:
        for assembly in assemblies:
            os.system(
                f"cp {path}/{strain}/{assembly} {path}_filtered_{max_assemblies}/{strain}/{assembly}")
