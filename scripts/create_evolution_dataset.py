import pandas as pd
from tqdm import tqdm
import os
from glob import glob
from Bio import SeqIO
import pickle
from concurrent.futures import ThreadPoolExecutor
import polars as pl


path = ".."
data = "1100"
outdir = f"{path}/seqfiles_{data}"


def load_pickle(path, data):
    with open(f"{path}/xgboost/annotations{data}/binary/binary.pkl", "rb") as f:
        # with open(f"{path}/binary.pkl", 'rb') as f:
        pickle_file = pickle.load(f)

    l = []
    rels = pickle_file.keys()
    for rel in tqdm(rels):
        for i in range(len(pickle_file[rel])):
            s = pd.Series(pickle_file[rel][i][2].get_score(importance_type="gain"))
            if len(s) > 0:
                importance_values = s.sort_values(ascending=False).values
                genes = s.sort_values(ascending=False).index.to_list()
                importance_ranking = (
                    s.sort_values(ascending=False).rank(ascending=False).values
                )
                accuracy = pickle_file[rel][i][1]
                ner = pickle_file[rel][i][0][0]
                for j in range(len(importance_values)):
                    l.append(
                        [
                            rel,
                            ner,
                            genes[j],
                            importance_values[j],
                            importance_ranking[j],
                            accuracy,
                        ]
                    )
    df = pd.DataFrame(
        l,
        columns=[
            "rel",
            "ner",
            "gene",
            "importance_values",
            "importance_ranking",
            "accuracy",
        ],
    )
    return df


def process_strain(strain, folder_path, protein_ids):
    output_faa = []
    output_fna = []
    protein_ids_seen = set()  # Keep track of protein IDs already seen

    if os.path.exists(folder_path):
        for assembly in os.listdir(folder_path):
            faa_files = glob(f"{folder_path}/{assembly}/*.faa")
            fna_files = glob(f"{folder_path}/{assembly}/*.cds")

            if faa_files:
                faa_file = faa_files[0]
                for record in SeqIO.parse(faa_file, "fasta"):
                    if record.id in protein_ids and record.id not in protein_ids_seen:
                        record.description = ""
                        output_faa.append(record)
                        protein_ids_seen.add(record.id)

            if fna_files:
                fna_file = fna_files[0]
                for record in SeqIO.parse(fna_file, "fasta"):
                    if "protein_id=" in record.description:
                        protein_id = record.description.split("protein_id=")[1].split(
                            "]"
                        )[0]
                        if (
                            protein_id in protein_ids
                            and protein_id not in protein_ids_seen
                        ):
                            record.id = protein_id
                            record.description = ""
                            output_fna.append(record)
                            protein_ids_seen.add(protein_id)

    return output_faa, output_fna


# @profile
def create_evolution_dataset(df, path, data, outdir):
    os.makedirs(outdir, exist_ok=True)

    df = pl.from_pandas(df)
    df = df.filter(pl.col("importance_ranking") == 1)

    for rel in tqdm(df["rel"].unique().to_list()):
        filtered_df = df.filter(pl.col("rel") == rel)

        filtered_df = filtered_df.sample(n=10, with_replacement=True)

        parq = pl.read_parquet(f"{path}/xgboost/annotations{data}/{rel}.parquet")

        for row in tqdm(
            filtered_df.iter_rows(named=True), total=len(filtered_df), leave=False
        ):
            sa_ner_df = parq.filter(pl.col("word_qc_group") == row["ner"])
            if not sa_ner_df.is_empty():
                strains = (
                    sa_ner_df["sa_ner"].str.split("!").list.get(0).unique().to_list()
                )
                new_rel = row["rel"].replace(":", "_")
                new_ner = (
                    row["ner"]
                    .replace(" ", "_")
                    .replace("'", "")
                    .replace("(", "_")
                    .replace(")", "_")
                )
                sa_ner = f"first_{new_rel}_{new_ner}"
                protein_ids = set(
                    parq.filter(pl.col("InterPro_description") == row["gene"])[
                        "Protein_accession"
                    ]
                    .unique()
                    .to_list()
                )

                output_faa = []
                output_fna = []

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(
                            process_strain,
                            s,
                            f"{path}/assemblies_{data}/{s}",
                            protein_ids,
                        ): s
                        for s in strains
                    }

                    for future in futures:
                        faa, fna = future.result()
                        output_faa.extend(faa)
                        output_fna.extend(fna)

                if output_faa and output_fna:
                    os.makedirs(f"{outdir}/{sa_ner}", exist_ok=True)
                    with open(f"{outdir}/{sa_ner}/seq.faa", "w") as f:
                        SeqIO.write(output_faa, f, "fasta")
                    with open(f"{outdir}/{sa_ner}/seq.fna", "w") as f:
                        SeqIO.write(output_fna, f, "fasta")


# def create_evolution_dataset(df, path, data, outdir):
# 	os.makedirs(outdir, exist_ok=True)

# 	df = df[df['importance_ranking'] == 1]
# 	# Find the smallest rel based on the number of occurrences
# 	smallest_rel = df['rel'].value_counts().idxmin()
# 	smallest_rel_count = df['rel'].value_counts().min()

# 	for rel in tqdm(df['rel'].unique()):
# 		filtered_df = df[(df['rel'] == rel)]

# 		# Subsample the rels to match the smallest rel count
# 		if rel != smallest_rel:
# 			filtered_df = filtered_df.sample(n=smallest_rel_count, replace=True)

# 		parq = pd.read_parquet(f"{path}/xgboost/annotations{data}/{rel}.parquet")

# 		for row in tqdm(filtered_df.iterrows(), total=len(filtered_df), leave=False):
# 			sa_ner = parq[parq["word_qc_group"] == row[1]['ner']].sa_ner
# 			if not sa_ner.empty:
# 				strains = sa_ner.str.split("!", expand=True)[0].unique()
# 				new_rel = row[1]['rel'].replace(':', '_')
# 				new_ner = row[1]['ner'].replace(' ', '_').replace("'", '').replace('(', '_').replace(')', '_')
# 				sa_ner = f"first_{new_rel}_{new_ner}"
# 				protein_ids = parq[parq["InterPro_description"] == row[1]['gene']].Protein_accession.unique()
# 				output_faa = []
# 				output_fna = []

# 				for s in strains:
# 					folder_path = f"{path}/assemblies_{data}/{s}"
# 					if os.path.exists(folder_path):
# 						for assembly in os.listdir(folder_path):
# 							faa_files = glob(f"{folder_path}/{assembly}/*.faa")
# 							fna_files = glob(f"{folder_path}/{assembly}/*.cds")

# 							if faa_files:
# 								faa_file = faa_files[0]
# 								for record in SeqIO.parse(faa_file, "fasta"):
# 									if record.id in protein_ids:
# 										record.description = ""
# 										output_faa.append(record)

# 							if fna_files:
# 								fna_file = fna_files[0]
# 								for record in SeqIO.parse(fna_file, "fasta"):
# 									if "protein_id=" in record.description:
# 										protein_id = record.description.split("protein_id=")[1].split("]")[0]
# 										if protein_id in protein_ids:
# 											record.id = protein_id
# 											record.description = ""
# 											output_fna.append(record)

# 				if output_faa and output_fna:
# 					os.makedirs(f"{outdir}/{sa_ner}", exist_ok=True)
# 					with open(f"{outdir}/{sa_ner}/seq.faa", "w") as f:
# 						SeqIO.write(output_faa, f, "fasta")
# 					with open(f"{outdir}/{sa_ner}/seq.fna", "w") as f:
# 						SeqIO.write(output_fna, f, "fasta")

# def create_evolution_test_set(df,path,data,outdir):
# 	for rel in tqdm(df['rel'].unique()):
# 		filtered_df = df[df['rel'] == rel]
# 		parq = pd.read_parquet(f"{path}/xgboost/annotations{data}/{rel}.parquet")
# 		for row in tqdm(filtered_df.iterrows(), total=len(filtered_df), leave=False):
# 			sa_ner = parq[parq["word_qc"]==row[1]['ner']].sa_ner
# 			if not sa_ner.empty:  # Use .empty to check if the Series is empty
# 				strains = sa_ner.str.split("!",expand=True)[0].unique()
# 				new_rel = row[1]['rel'].replace(':','_')
# 				new_ner = row[1]['ner'].replace(' ','_').replace("'",'').replace('(','_').replace(')','_')
# 				sa_ner = f"last_{new_rel}_{new_ner}"
# 				protein_ids = parq[parq["InterPro_description"]== row[1]['gene']].Protein_accession.unique()
# 				output_faa = []
# 				output_fna = []
# 				for s in strains:
# 					strain = s.replace(" ","_")
# 					folder_path = f"{path}/assemblies_{data}/{strain}"
# 					if os.path.exists(folder_path):
# 						for assembly in os.listdir(folder_path):
# 							faa_files = glob(f"{folder_path}/{assembly}/*.faa")
# 							fna_files = glob(f"{folder_path}/{assembly}/*.fasta")
# 							if faa_files:
# 								faa_file = faa_files[0]
# 								for record in SeqIO.parse(faa_file, "fasta"):
# 									if record.id in protein_ids:
# 										record.description = ""
# 										output_faa.append(record)
# 							if fna_files:
# 								fna_file = fna_files[0]
# 								for record in SeqIO.parse(fna_file, "fasta"):
# 									if record.id in protein_ids:
# 										output_fna.append(record)
# 				os.makedirs(f"{outdir}/{sa_ner}",exist_ok=True)
# 				with open(f"{outdir}/{sa_ner}/seq.faa","w") as f:
# 					SeqIO.write(output_faa, f, "fasta")
# 				with open(f"{outdir}/{sa_ner}/seq.fna","w") as f:
# 					SeqIO.write(output_fna, f, "fasta")


def deduplicate_dataset(path, data):
    # Set the directory where the files are located
    directory = f"{path}/seqfiles_{data}"

    # Iterate over all files in the directory
    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            # Check if the file is a fasta file
            if file.endswith(".faa") or file.endswith(".fna"):
                # Get the full path of the file
                file_path = os.path.join(root, file)

                # Create a temporary file to store the deduplicated sequences
                temp_file = file_path + ".temp"

                # Check if the file exists
                if not os.path.exists(file_path):
                    continue

                # Run seqkit rmdup command to delete duplicate sequences
                os.system(f"seqkit rmdup -n -o {temp_file} {file_path}")

                # Replace the original file with the deduplicated file
                os.replace(temp_file, file_path)


def main():
    df = load_pickle(path, data)
    create_evolution_dataset(df, path, data, outdir)
    deduplicate_dataset(path, data)


if __name__ == "__main__":
    main()
