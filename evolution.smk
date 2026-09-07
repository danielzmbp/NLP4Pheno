import pandas as pd
import types

configfile: "config.yaml"

data = config["dataset"]
path = f"{config['output_path'].rstrip('/')}/xgboost/seqfiles_{data}"
hyphy_remove_duplicates = config.get(
    "hyphy_remove_duplicates",
    "/home/tu/tu_tu/tu_kmpaj01/hyphy-analyses/remove-duplicates/remove-duplicates.bf",
)
(R,) = glob_wildcards(path + "/{rel}/seq.faa")

# Common resource configurations
COMMON_RESOURCES = {
    "slurm_partition": "cpu,cpu_il",
    "runtime": 2000,
    "mem_mb": 8192
}

HEAVY_RESOURCES = {
    "slurm_partition": "cpu,cpu_il",
    "runtime": 4320,
    "mem_mb": 32768
}

localrules: deduplicate, align, codonaln, remove_dups, final, filter_matching, sample_sequences, summary_stats


rule final:
    input:
        expand(
            path + "/{rel}/seq.json",
            rel=R,
        ),
        path + "/evolution_summary.txt"


rule deduplicate:
    input:
        path + "/{rel}/seq.faa",
    output:
        path + "/{rel}/seq.dedup.faa",
    run:
        from collections import defaultdict

        sequences = []

        # Read sequences
        with open(input[0], 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        seq = ''.join(current_seq)
                        sequences.append((current_id, seq))
                    current_id = line.strip()
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id:
                seq = ''.join(current_seq)
                sequences.append((current_id, seq))

        # Keep only unique sequences (by name+sequence combination)
        unique_seqs = []
        seen_combos = set()
        for name, seq in sequences:
            combo = (name, seq)
            if combo not in seen_combos:
                unique_seqs.append((name, seq))
                seen_combos.add(combo)

        # Write deduplicated sequences
        with open(output[0], 'w') as f:
            for name, seq in unique_seqs:
                f.write(name + '\n')
                # Write sequence in 80-character lines
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + '\n')


rule filter_matching:
    input:
        faa=path + "/{rel}/seq.dedup.faa",
        fna=path + "/{rel}/seq.fna",
    output:
        faa=path + "/{rel}/seq.filtered.faa",
        fna=path + "/{rel}/seq.filtered.fna",
        marker=path + "/{rel}/seq.sufficient",
    run:
        # Read protein sequences and get IDs
        protein_ids = set()
        with open(input.faa, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    protein_ids.add(line.strip()[1:])

        # Read CDS sequences and get IDs
        cds_ids = set()
        cds_sequences = []
        with open(input.fna, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id and current_id in protein_ids:
                        cds_sequences.append((f">{current_id}", ''.join(current_seq)))
                    current_id = line.strip()[1:]
                    cds_ids.add(current_id)
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            # Handle the last sequence
            if current_id and current_id in protein_ids:
                cds_sequences.append((f">{current_id}", ''.join(current_seq)))

        # Find common IDs
        common_ids = protein_ids & cds_ids
        excluded_proteins = protein_ids - cds_ids
        excluded_cds = cds_ids - protein_ids

        # Log filtering results
        print(f"Processing {wildcards.rel}:")
        print(f"  Protein sequences: {len(protein_ids)}")
        print(f"  CDS sequences: {len(cds_ids)}")
        print(f"  Common sequences: {len(common_ids)}")
        if excluded_proteins:
            print(f"  Excluded proteins (no CDS): {list(excluded_proteins)[:5]}{'...' if len(excluded_proteins) > 5 else ''}")
        if excluded_cds:
            print(f"  Excluded CDS (no protein): {list(excluded_cds)[:5]}{'...' if len(excluded_cds) > 5 else ''}")

        # Check if we have sufficient sequences for evolution analysis
        if len(common_ids) < 2:
            print(f"  INSUFFICIENT: Only {len(common_ids)} sequences, skipping evolution analysis")
            # Create empty output files to satisfy snakemake dependencies
            with open(output.faa, 'w') as f:
                pass
            with open(output.fna, 'w') as f:
                pass
            # Create marker file indicating insufficient sequences
            with open(output.marker, 'w') as f:
                f.write(f"insufficient\t{len(common_ids)}\n")
            return

        # Write filtered protein sequences
        with open(input.faa, 'r') as infile, open(output.faa, 'w') as outfile:
            current_id = None
            current_seq = []
            include_seq = False

            for line in infile:
                if line.startswith('>'):
                    # Write previous sequence if it should be included
                    if current_id and include_seq:
                        outfile.write(f">{current_id}\n")
                        for i in range(0, len(''.join(current_seq)), 80):
                            outfile.write(''.join(current_seq)[i:i+80] + '\n')

                    current_id = line.strip()[1:]
                    include_seq = current_id in common_ids
                    current_seq = []
                else:
                    if include_seq:
                        current_seq.append(line.strip())

            # Handle the last sequence
            if current_id and include_seq:
                outfile.write(f">{current_id}\n")
                for i in range(0, len(''.join(current_seq)), 80):
                    outfile.write(''.join(current_seq)[i:i+80] + '\n')

        # Write filtered CDS sequences
        with open(output.fna, 'w') as outfile:
            for header, seq in cds_sequences:
                outfile.write(header + '\n')
                for i in range(0, len(seq), 80):
                    outfile.write(seq[i:i+80] + '\n')

        # Create marker file indicating sufficient sequences
        with open(output.marker, 'w') as f:
            f.write(f"sufficient\t{len(common_ids)}\n")


rule sample_sequences:
    input:
        faa=path + "/{rel}/seq.filtered.faa",
        fna=path + "/{rel}/seq.filtered.fna",
        marker=path + "/{rel}/seq.sufficient",
    output:
        faa=path + "/{rel}/seq.sampled.faa",
        fna=path + "/{rel}/seq.sampled.fna",
    run:
        import random

        # Check if we have sufficient sequences
        with open(input.marker, 'r') as f:
            status, count = f.read().strip().split('\t')

        if status == "insufficient":
            print(f"Skipping sampling for {wildcards.rel}: insufficient sequences")
            # Create empty output files
            with open(output.faa, 'w') as f:
                pass
            with open(output.fna, 'w') as f:
                pass
            return

        # Read protein sequences
        protein_sequences = []
        with open(input.faa, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        protein_sequences.append((current_id, ''.join(current_seq)))
                    current_id = line.strip()[1:]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id:
                protein_sequences.append((current_id, ''.join(current_seq)))

        # Read nucleotide sequences
        nucleotide_sequences = {}
        with open(input.fna, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        nucleotide_sequences[current_id] = ''.join(current_seq)
                    current_id = line.strip()[1:]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id:
                nucleotide_sequences[current_id] = ''.join(current_seq)

        # Sample sequences if we have more than 100
        num_sequences = len(protein_sequences)
        if num_sequences > 100:
            print(f"Sampling {wildcards.rel}: {num_sequences} sequences -> 100 sequences")
            # Set random seed for reproducibility
            random.seed(42)
            sampled_proteins = random.sample(protein_sequences, 100)
        else:
            print(f"Keeping all sequences for {wildcards.rel}: {num_sequences} sequences")
            sampled_proteins = protein_sequences

        # Write sampled protein sequences
        with open(output.faa, 'w') as f:
            for seq_id, seq in sampled_proteins:
                f.write(f">{seq_id}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + '\n')

        # Write corresponding nucleotide sequences
        with open(output.fna, 'w') as f:
            for seq_id, seq in sampled_proteins:
                if seq_id in nucleotide_sequences:
                    nucl_seq = nucleotide_sequences[seq_id]
                    f.write(f">{seq_id}\n")
                    for i in range(0, len(nucl_seq), 80):
                        f.write(nucl_seq[i:i+80] + '\n')


rule align:
    input:
        path + "/{rel}/seq.sampled.faa",
    output:
        path + "/{rel}/seq.aln",
    threads: 1
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=600,
        mem_mb=2048
    shell:
        """
        set +e  # Disable exit on error for grep checks
        # Check if we have sufficient sequences
        MARKER_FILE="{path}/{wildcards.rel}/seq.sufficient"
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping alignment for {wildcards.rel}: insufficient sequences"
            echo "" > "{output}"
        elif grep -q ">" "{input}" 2>/dev/null; then
            set -e  # Re-enable exit on error
            mafft --auto --thread {threads} "{input}" > "{output}" 2>/dev/null || {{
                echo "MAFFT failed for {wildcards.rel}, creating empty alignment"
                echo "" > "{output}"
            }}
        else
            echo "Skipping alignment for {wildcards.rel}: empty input"
            echo "" > "{output}"
        fi
        """


rule fasttree:
    input:
        path + "/{rel}/seq.aln",
    output:
        path + "/{rel}/seq.tree",
    threads: 16
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=1200,
        mem_mb=8192
    shell:
        """
        set +e  # Disable exit on error for grep checks
        # Check if we have sufficient sequences
        MARKER_FILE="{path}/{wildcards.rel}/seq.sufficient"
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping FastTree for {wildcards.rel}: insufficient sequences"
            echo "# Insufficient sequences for tree construction" > "{output}"
        elif grep -q ">" "{input}" 2>/dev/null; then
            set -e  # Re-enable exit on error
            fasttree -nosupport "{input}" > "{output}"
        else
            echo "Skipping FastTree for {wildcards.rel}: empty alignment"
            echo "# Empty alignment file" > "{output}"
        fi
        """


rule codonaln:
    input:
        pro_align=path + "/{rel}/seq.aln",
        nucl_seq=path + "/{rel}/seq.sampled.fna",
    output:
        alignment=path + "/{rel}/seq.aln.codon",
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=600,
        mem_mb=2048
    shell:
        """
        set +e  # Disable exit on error for grep checks
        # Check if we have sufficient sequences
        MARKER_FILE="{path}/{wildcards.rel}/seq.sufficient"
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping pal2nal for {wildcards.rel}: insufficient sequences"
            echo "# Insufficient sequences for codon alignment" > "{output.alignment}"
        elif grep -q ">" "{input.pro_align}" 2>/dev/null && grep -q ">" "{input.nucl_seq}" 2>/dev/null; then
            set -e  # Re-enable exit on error
            pal4nal.pl "{input.pro_align}" "{input.nucl_seq}" -output fasta -nomismatch -o "{output.alignment}" || {{
                echo "pal2nal failed for {wildcards.rel}, creating placeholder"
                echo "# pal2nal failed due to sequence mismatches" > "{output.alignment}"
            }}
        else
            echo "Skipping pal2nal for {wildcards.rel}: empty input files"
            echo "# Empty input files" > "{output.alignment}"
        fi
        """


rule remove_dups:
    input:
        aln_codon=path + "/{rel}/seq.aln.codon",
        tree=path + "/{rel}/seq.tree",
    output:
        path + "/{rel}/seq.nxh",
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=1200,
        mem_mb=8192
    shell:
        """
        set +e  # Disable exit on error for grep checks
        # Check if we have sufficient sequences
        MARKER_FILE="{path}/{wildcards.rel}/seq.sufficient"
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping remove_dups for {wildcards.rel}: insufficient sequences"
            echo "# Insufficient sequences for duplicate removal" > "{output}"
        elif grep -q "^#" "{input.aln_codon}" || grep -q "^#" "{input.tree}"; then
            echo "Skipping remove_dups for {wildcards.rel}: invalid input files"
            echo "# Invalid input files" > "{output}"
        else
            set -e  # Re-enable exit on error
            hyphy {hyphy_remove_duplicates} --msa "{input.aln_codon}" --tree "{input.tree}" --output "{output}" || {{
                echo "HyPhy remove_dups failed for {wildcards.rel}"
                echo "# HyPhy remove_dups failed" > "{output}"
            }}
        fi
        """


def check_sufficient_sequences(wildcards):
    """Check if we have sufficient sequences for processing"""
    marker_file = f"{path}/{wildcards.rel}/seq.sufficient"
    try:
        with open(marker_file, 'r') as f:
            status, count = f.read().strip().split('\t')
        return status == "sufficient"
    except FileNotFoundError:
        # If marker doesn't exist, default to running (old behavior)
        return True

def get_busted_input(wildcards):
    """Only run BUSTED if we have sufficient sequences"""
    if check_sufficient_sequences(wildcards):
        return f"{path}/{wildcards.rel}/seq.nxh"
    else:
        return []  # Empty list means skip this rule

rule busted:
    input:
        get_busted_input
    output:
        json=path + "/{rel}/seq.json",
        log=path + "/{rel}/seq.log",
    threads: 32
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu,cpu_il",
        runtime=4320,
        mem_mb=32 * 1024,
    shell:
        """
        set +e  # Disable exit on error for grep checks
        if grep -q "NTAX = 1" "{input}" 2>/dev/null; then
            echo "Skipping BUSTED for {wildcards.rel}: only 1 taxon (insufficient)"
            echo '{{"analysis": {{"info": "Only 1 taxon - insufficient for BUSTED analysis"}}}}' > "{output.json}"
            echo "# Only 1 taxon - insufficient for BUSTED analysis" > "{output.log}"
        elif grep -q -v "^#" "{input}" 2>/dev/null && [ -s "{input}" ]; then
            set -e  # Re-enable exit on error
            ENV=TOLERATE_NUMERICAL_ERRORS=1
            CPU={threads}
            hyphy busted --alignment "{input}" --output "{output.json}" > "{output.log}" || {{
                echo "BUSTED failed for {wildcards.rel}"
                echo '{{"analysis": {{"info": "BUSTED analysis failed"}}}}' > "{output.json}"
                echo "# BUSTED analysis failed" > "{output.log}"
            }}
        else
            echo "Skipping BUSTED for {wildcards.rel}: insufficient sequences"
            echo '{{"analysis": {{"info": "Insufficient sequences for BUSTED analysis"}}}}' > "{output.json}"
            echo "# Insufficient sequences for BUSTED analysis" > "{output.log}"
        fi
        """


rule summary_stats:
    input:
        markers=expand(path + "/{rel}/seq.sufficient", rel=R),
        jsons=expand(path + "/{rel}/seq.json", rel=R),
    output:
        summary=path + "/evolution_summary.txt",
        csv=path + "/evolution_summary.csv",
    run:
        import json
        import os
        from collections import defaultdict

        print("\n" + "="*80)
        print("EVOLUTION PIPELINE SUMMARY STATISTICS")
        print("="*80)

        # Initialize counters
        stats = {
            "insufficient": [],      # <2 sequences
            "normal": [],           # 2-100 sequences
            "sampled": [],          # >100 sequences (sampled to 100)
            "failed_analysis": [],  # BUSTED analysis failed
            "successful": [],       # Completed successfully
        }

        total_relationships = len(R)

        # Process each relationship
        for rel in R:
            marker_file = f"{path}/{rel}/seq.sufficient"
            json_file = f"{path}/{rel}/seq.json"

            # Read marker file to get sequence count status
            try:
                with open(marker_file, 'r') as f:
                    status, count = f.read().strip().split('\t')
                    count = int(count)

                if status == "insufficient":
                    stats["insufficient"].append((rel, count))
                elif count > 100:
                    stats["sampled"].append((rel, count))
                else:
                    stats["normal"].append((rel, count))

            except (FileNotFoundError, ValueError):
                # If marker file doesn't exist or is malformed, check if sequences exist
                faa_file = f"{path}/{rel}/seq.faa"
                if os.path.exists(faa_file):
                    stats["normal"].append((rel, "unknown"))
                else:
                    stats["insufficient"].append((rel, 0))

            # Check BUSTED analysis success
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                # Check if it's a successful BUSTED analysis
                if "test results" in json_data and "p-value" in json_data["test results"]:
                    stats["successful"].append(rel)
                else:
                    # Check for error messages
                    if "analysis" in json_data and "info" in json_data["analysis"]:
                        stats["failed_analysis"].append((rel, json_data["analysis"]["info"]))
                    else:
                        stats["failed_analysis"].append((rel, "Unknown error"))

            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                stats["failed_analysis"].append((rel, "JSON file missing or corrupted"))

        # Print console summary
        print(f"Total relationships analyzed: {total_relationships}")
        print()

        print(f"SEQUENCE COUNT DISTRIBUTION:")
        print(f"  Insufficient sequences (<2): {len(stats['insufficient'])} ({len(stats['insufficient'])/total_relationships*100:.1f}%)")
        print(f"  Normal range (2-100): {len(stats['normal'])} ({len(stats['normal'])/total_relationships*100:.1f}%)")
        print(f"  Large datasets (>100, sampled): {len(stats['sampled'])} ({len(stats['sampled'])/total_relationships*100:.1f}%)")
        print()

        print(f"ANALYSIS SUCCESS RATES:")
        print(f"  Successful BUSTED analyses: {len(stats['successful'])} ({len(stats['successful'])/total_relationships*100:.1f}%)")
        print(f"  Failed analyses: {len(stats['failed_analysis'])} ({len(stats['failed_analysis'])/total_relationships*100:.1f}%)")
        print()

        # Show examples of each category
        if stats["insufficient"]:
            print(f"INSUFFICIENT SEQUENCES (examples):")
            for rel, count in stats["insufficient"][:5]:
                print(f"  {rel}: {count} sequences")
            if len(stats["insufficient"]) > 5:
                print(f"  ... and {len(stats['insufficient'])-5} more")
            print()

        if stats["sampled"]:
            print(f"LARGE DATASETS SAMPLED (examples):")
            for rel, count in sorted(stats["sampled"], key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {rel}: {count} sequences -> 100")
            if len(stats["sampled"]) > 5:
                print(f"  ... and {len(stats['sampled'])-5} more")
            print()

        if stats["failed_analysis"]:
            print(f"FAILED ANALYSES (examples):")
            for rel, error in stats["failed_analysis"][:5]:
                print(f"  {rel}: {error}")
            if len(stats["failed_analysis"]) > 5:
                print(f"  ... and {len(stats['failed_analysis'])-5} more")
            print()

        # Create detailed CSV report
        csv_data = []
        for rel in R:
            # Get sequence info
            marker_file = f"{path}/{rel}/seq.sufficient"
            json_file = f"{path}/{rel}/seq.json"

            seq_status = "unknown"
            seq_count = "unknown"
            analysis_status = "unknown"
            error_message = ""

            # Parse sequence information
            try:
                with open(marker_file, 'r') as f:
                    status, count = f.read().strip().split('\t')
                    seq_count = int(count)
                    if status == "insufficient":
                        seq_status = "insufficient"
                    elif seq_count > 100:
                        seq_status = "sampled"
                    else:
                        seq_status = "normal"
            except:
                seq_status = "unknown"

            # Parse analysis results
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                if "test results" in json_data and "p-value" in json_data["test results"]:
                    analysis_status = "success"
                else:
                    analysis_status = "failed"
                    if "analysis" in json_data and "info" in json_data["analysis"]:
                        error_message = json_data["analysis"]["info"]
            except:
                analysis_status = "failed"
                error_message = "JSON file missing or corrupted"

            csv_data.append({
                "relationship": rel,
                "sequence_count": seq_count,
                "sequence_status": seq_status,
                "analysis_status": analysis_status,
                "error_message": error_message
            })

        # Save text summary
        with open(output.summary, 'w') as f:
            f.write("EVOLUTION PIPELINE SUMMARY STATISTICS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total relationships: {total_relationships}\n\n")
            f.write(f"Sequence distribution:\n")
            f.write(f"  Insufficient (<2): {len(stats['insufficient'])}\n")
            f.write(f"  Normal (2-100): {len(stats['normal'])}\n")
            f.write(f"  Sampled (>100): {len(stats['sampled'])}\n\n")
            f.write(f"Analysis results:\n")
            f.write(f"  Successful: {len(stats['successful'])}\n")
            f.write(f"  Failed: {len(stats['failed_analysis'])}\n\n")

            f.write("Detailed breakdown:\n\n")
            f.write("INSUFFICIENT SEQUENCES:\n")
            for rel, count in stats["insufficient"]:
                f.write(f"  {rel}: {count}\n")

            f.write("\nLARGE DATASETS SAMPLED:\n")
            for rel, count in stats["sampled"]:
                f.write(f"  {rel}: {count} -> 100\n")

            f.write("\nFAILED ANALYSES:\n")
            for rel, error in stats["failed_analysis"]:
                f.write(f"  {rel}: {error}\n")

        # Save CSV
        import csv
        with open(output.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["relationship", "sequence_count", "sequence_status", "analysis_status", "error_message"])
            writer.writeheader()
            writer.writerows(csv_data)

        print("="*80)
        print(f"Summary saved to: {output.summary}")
        print(f"Detailed CSV saved to: {output.csv}")
        print("="*80)
