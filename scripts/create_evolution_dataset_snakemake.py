from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
from Bio import SeqIO
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
import polars as pl


PATH   = snakemake.params.path
DATA   = snakemake.params.data
DONE   = Path(snakemake.output[0])
OUTDIR = f"{PATH}/xgboost/seqfiles_{DATA}"


def load_pickle(path, data):
    with open(f"{path}/xgboost/annotations{data}/binary/binary.pkl", "rb") as f:
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
    protein_ids_seen = set()
    cds_ids_seen = set()
    
    # Convert to set for faster lookup
    protein_ids_set = set(protein_ids)

    if not os.path.exists(folder_path):
        return output_faa, output_fna

    try:
        for assembly in os.listdir(folder_path):
            assembly_path = f"{folder_path}/{assembly}"
            
            # Process FAA files
            faa_files = glob(f"{assembly_path}/*.faa")
            if faa_files:
                faa_file = faa_files[0]
                try:
                    for record in SeqIO.parse(faa_file, "fasta"):
                        if record.id in protein_ids_set and record.id not in protein_ids_seen:
                            record.description = ""
                            output_faa.append(record)
                            protein_ids_seen.add(record.id)
                            # Early termination if we have all proteins
                            if len(protein_ids_seen) == len(protein_ids_set):
                                break
                except Exception as e:
                    print(f"Error processing FAA file {faa_file}: {e}")

            # Process FNA files
            fna_files = glob(f"{assembly_path}/*.cds")
            if fna_files:
                fna_file = fna_files[0]
                try:
                    for record in SeqIO.parse(fna_file, "fasta"):
                        if "protein_id=" in record.description:
                            protein_id = record.description.split("protein_id=")[1].split("]")[0]
                            if protein_id in protein_ids_set and protein_id not in cds_ids_seen:
                                record.id = protein_id
                                record.description = ""
                                output_fna.append(record)
                                cds_ids_seen.add(protein_id)
                                # Early termination if we have all proteins
                                if len(cds_ids_seen) == len(protein_ids_set):
                                    break
                except Exception as e:
                    print(f"Error processing FNA file {fna_file}: {e}")

    except Exception as e:
        print(f"Error processing strain {strain}: {e}")

    return output_faa, output_fna


def select_rank_1_entry(df_group, ip_names_pl):
    """
    Select the rank 1 entry and join with InterPro information.

    Args:
        df_group: DataFrame group with same rel/ner combination
        ip_names_pl: Polars DataFrame with InterPro entry information including ENTRY_TYPE

    Returns:
        Single row with rank 1 entry including ENTRY_TYPE and ENTRY_AC
    """
    # Get rank 1 entry
    rank_1_entry = df_group.filter(pl.col("importance_ranking") == 1).head(1)

    if rank_1_entry.is_empty():
        # Fallback to best available ranking
        rank_1_entry = df_group.filter(pl.col("importance_ranking") == df_group["importance_ranking"].min()).head(1)

    # Join with InterPro entry types to get ENTRY_TYPE and ENTRY_AC
    entry_with_types = rank_1_entry.join(ip_names_pl, left_on="gene", right_on="ENTRY_NAME", how="left")

    return entry_with_types


# @profile
def create_evolution_dataset(df, path, data, outdir, ip_names_pl):
    """Create evolution dataset with optimized processing"""
    os.makedirs(outdir, exist_ok=True)

    df = pl.from_pandas(df)

    # Initialize summary tracking
    summary_records = []

    # Select rank 1 entries only
    # Group by rel and ner, then select rank 1 entry per group
    grouped_results = []
    unique_combinations = df.select(["rel", "ner"]).unique()

    print(f"Processing {len(unique_combinations)} unique rel-ner combinations")

    for row in unique_combinations.iter_rows(named=True):
        rel, ner = row["rel"], row["ner"]
        group_df = df.filter((pl.col("rel") == rel) & (pl.col("ner") == ner))

        print(f"Processing {rel} - {ner} ({len(group_df)} candidates)")

        # Select rank 1 entry for this rel/ner combination
        rank_1_entry = select_rank_1_entry(group_df, ip_names_pl)
        grouped_results.append(rank_1_entry)

        # Track selection for summary
        if len(rank_1_entry) > 0:
            entry_data = rank_1_entry.row(0, named=True)
            summary_records.append({
                "rel": rel,
                "ner": ner,
                "selected_gene": entry_data.get("gene", "Unknown"),
                "entry_type": entry_data.get("ENTRY_TYPE", "Unknown"),
                "entry_ac": entry_data.get("ENTRY_AC", "Unknown"),
                "importance_ranking": entry_data.get("importance_ranking", "Unknown"),
                "importance_value": entry_data.get("importance_values", "Unknown"),
                "accuracy": entry_data.get("accuracy", "Unknown")
            })

    # Combine all selected entries
    if grouped_results:
        df = pl.concat(grouped_results)
        # The helper function already joins with InterPro data, no need to join again
    else:
        # Fallback to empty dataframe with proper schema
        df = pl.DataFrame(schema=df.schema)

    # Save selection summary
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = f"{outdir}/selection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved selection summary to {summary_path}")

        # Print summary statistics
        family_count = sum(1 for r in summary_records if r["entry_type"] == "Family")
        total_count = len(summary_records)
        print(f"Rank 1 Selection Summary: {family_count}/{total_count} ({family_count/total_count*100:.1f}%) Family entries at rank 1")

        # Print entry type distribution
        type_counts = summary_df["entry_type"].value_counts()
        print("Entry type distribution (rank 1 entries):")
        for entry_type, count in type_counts.items():
            print(f"  {entry_type}: {count} ({count/total_count*100:.1f}%)")
    
    # Process relationships in batches for better memory management
    unique_rels = df["rel"].unique().to_list()
    print(f"Processing {len(unique_rels)} relationships")
    
    for rel in tqdm(unique_rels, desc="Processing relationships"):
        try:
            filtered_df = df.filter(pl.col("rel") == rel)
            
            # Load the parquet file once per relationship
            parquet_path = f"{path}/xgboost/annotations{data}/{rel}.parquet"
            if not os.path.exists(parquet_path):
                print(f"Parquet file not found: {parquet_path}")
                continue
                
            parq = pl.read_parquet(parquet_path)
            
            for row in tqdm(
                filtered_df.iter_rows(named=True), 
                total=len(filtered_df), 
                leave=False,
                desc=f"Processing {rel} entries"
            ):
                try:
                    # Filter data for this specific entry
                    sa_ner_df = parq.filter(pl.col("word_qc_group") == row["ner"])
                    if sa_ner_df.is_empty():
                        continue

                    # Use the ENTRY_AC from the selected entry (now properly preserved)
                    entry_ac = row.get("ENTRY_AC")
                    if entry_ac is None:
                        print(f"  Warning: No ENTRY_AC found for {row['gene']}, skipping")
                        continue

                    strain_filter = parq.filter(
                        (pl.col("InterPro_accession") == entry_ac) &
                        (pl.col("word_qc_group") == row["ner"])
                    )

                    if strain_filter.is_empty():
                        print(f"  Warning: No data found for InterPro {entry_ac} with NER {row['ner']}")
                        continue

                    # Log what we're processing
                    entry_type = row.get("ENTRY_TYPE", "Unknown")
                    gene_name = row.get("gene", "Unknown")
                    ranking = row.get("importance_ranking", "Unknown")
                    print(f"  Processing rank 1 {entry_type} entry '{gene_name}' ({entry_ac})")
                        
                    strains = (strain_filter["sa_ner"]
                              .str.split("!")
                              .list.get(0)
                              .unique()
                              .to_list())
                    
                    if not strains:
                        continue
                        
                    # Create sanitized names
                    new_rel = row["rel"].replace(":", "_")
                    new_ner = (row["ner"]
                              .replace(" ", "_").replace("'", "").replace("(", "_")
                              .replace(")", "_").replace("/", "_").replace(":", "_")
                              .replace("&", "_").replace(",", "_")
                              .replace("[", "_").replace("]", "_").replace("^", "_")
                              .replace("½", "half").replace("¼", "quarter").replace("¾", "three_quarters")
                              .replace("⅓", "one_third").replace("⅔", "two_thirds").replace("⅛", "one_eighth"))
                    sa_ner = f"first_{new_rel}_{new_ner}"
                    
                    protein_ids = set(strain_filter["Protein_accession"].unique().to_list())
                    
                    if not protein_ids:
                        continue

                    output_faa = []
                    output_fna = []

                    # Process strains in parallel with controlled batch size
                    batch_size = min(len(strains), int(snakemake.threads))
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
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
                            try:
                                faa, fna = future.result(timeout=60)  # 1 minute timeout per strain
                                output_faa.extend(faa)
                                output_fna.extend(fna)
                            except Exception as e:
                                print(f"Error processing strain {futures[future]}: {e}")

                    # Write output files if we have sequences
                    if output_faa and output_fna:
                        output_dir = f"{outdir}/{sa_ner}"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        try:
                            with open(f"{output_dir}/seq.faa", "w") as f:
                                SeqIO.write(output_faa, f, "fasta")
                            with open(f"{output_dir}/seq.fna", "w") as f:
                                SeqIO.write(output_fna, f, "fasta")
                        except Exception as e:
                            print(f"Error writing sequences for {sa_ner}: {e}")
                            
                except Exception as e:
                    print(f"Error processing row in {rel}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing relationship {rel}: {e}")
            continue


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
    """Deduplicate FASTA sequences using seqkit with better error handling"""
    directory = f"{path}/xgboost/seqfiles_{data}"
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist, skipping deduplication")
        return

    fasta_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".faa", ".fna")):
                fasta_files.append(os.path.join(root, file))
    
    print(f"Found {len(fasta_files)} FASTA files to deduplicate")
    
    for file_path in tqdm(fasta_files, desc="Deduplicating FASTA files"):
        if not os.path.exists(file_path):
            continue
            
        temp_file = file_path + ".temp"
        
        try:
            # Use subprocess for better error handling
            result = subprocess.run(
                ["seqkit", "rmdup", "-n", "-o", temp_file, file_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            if result.returncode == 0:
                os.replace(temp_file, file_path)
            else:
                print(f"Error deduplicating {file_path}: {result.stderr}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except subprocess.TimeoutExpired:
            print(f"Timeout deduplicating {file_path}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except FileNotFoundError:
            print("seqkit not found, skipping deduplication")
            break
        except Exception as e:
            print(f"Error deduplicating {file_path}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

ip_names = pd.read_csv(
    "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list",
    sep="\t",
    header=0,
)
ip_names.set_index("ENTRY_AC", inplace=True)

ip_names["ENTRY_NAME"] = (
    ip_names["ENTRY_NAME"]
    .str.replace("[", "_")
    .str.replace("]", "_")
    .str.replace("<", "_")
)

# Keep entry types for family prioritization
ip_names_pl = pl.from_pandas(ip_names.reset_index())

df = load_pickle(PATH, DATA)
create_evolution_dataset(df, PATH, DATA, OUTDIR, ip_names_pl)
deduplicate_dataset(PATH, DATA)

# Print final summary if selection_summary.csv was created
summary_path = f"{OUTDIR}/selection_summary.csv"
if os.path.exists(summary_path):
    print("\n" + "="*60)
    print("FINAL SELECTION SUMMARY")
    print("="*60)

    summary_df = pd.read_csv(summary_path)
    total_selections = len(summary_df)
    family_selections = len(summary_df[summary_df["entry_type"] == "Family"])

    print(f"Total rank 1 selections made: {total_selections}")
    print(f"Family entries at rank 1: {family_selections} ({family_selections/total_selections*100:.1f}%)")
    print(f"Non-family entries at rank 1: {total_selections - family_selections} ({(total_selections - family_selections)/total_selections*100:.1f}%)")

    print("\nEntry type breakdown (rank 1 entries):")
    type_counts = summary_df["entry_type"].value_counts().sort_values(ascending=False)
    for entry_type, count in type_counts.items():
        print(f"  {entry_type}: {count} ({count/total_selections*100:.1f}%)")

    print(f"\nRanking verification (should all be 1):")
    ranking_counts = summary_df["importance_ranking"].value_counts().sort_index()
    for ranking, count in ranking_counts.items():
        print(f"  Rank {ranking}: {count} entries")

    print(f"\nDetailed summary saved to: {summary_path}")
    print("="*60)

# Touch the flag-file **only if everything succeeded**
if DONE is not None:
    DONE.parent.mkdir(parents=True, exist_ok=True)
    DONE.touch()