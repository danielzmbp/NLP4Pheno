import pandas as pd
import types

configfile: "config.yaml"

data = config["dataset"]
path = f"/home/tu/tu_tu/tu_kmpaj01/link/xgboost/seqfiles_{data}"
(R,) = glob_wildcards(path + "/{rel}/seq.faa")

# Common resource configurations
COMMON_RESOURCES = {
    "slurm_partition": "cpu",
    "runtime": 2000,
    "mem_mb": 8192
}

HEAVY_RESOURCES = {
    "slurm_partition": "cpu",
    "runtime": 4320,
    "mem_mb": 32768
}

localrules: deduplicate, align, codonaln, remove_dups, final, filter_matching


rule final:
    input:
        expand(
            path + "/{rel}/seq.json",
            rel=R,
        )


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


rule align:
    input:
        path + "/{rel}/seq.filtered.faa",
    output:
        path + "/{rel}/seq.aln",
    threads: 1
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu",
        runtime=600,
        mem_mb=2048
    shell:
        """
        # Check if we have sufficient sequences
        MARKER_FILE={path}/{wildcards.rel}/seq.sufficient
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping alignment for {wildcards.rel}: insufficient sequences"
            echo "" > {output}
        elif grep -q ">" {input} 2>/dev/null; then
            mafft --auto --thread {threads} {input} > {output} 2>/dev/null || {{
                echo "MAFFT failed for {wildcards.rel}, creating empty alignment"
                echo "" > {output}
            }}
        else
            echo "Skipping alignment for {wildcards.rel}: empty input"
            echo "" > {output}
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
        slurm_partition="cpu",
        runtime=1200,
        mem_mb=8192
    shell:
        """
        # Check if we have sufficient sequences
        MARKER_FILE={path}/{wildcards.rel}/seq.sufficient
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping FastTree for {wildcards.rel}: insufficient sequences"
            echo "# Insufficient sequences for tree construction" > {output}
        elif grep -q ">" {input} 2>/dev/null; then
            fasttree -nosupport {input} > {output}
        else
            echo "Skipping FastTree for {wildcards.rel}: empty alignment"
            echo "# Empty alignment file" > {output}
        fi
        """


rule codonaln:
    input:
        pro_align=path + "/{rel}/seq.aln",
        nucl_seq=path + "/{rel}/seq.filtered.fna",
    output:
        alignment=path + "/{rel}/seq.aln.codon",
    conda:
        "envs/evolution.yml"
    resources:
        slurm_partition="cpu",
        runtime=600,
        mem_mb=2048
    shell:
        """
        # Check if we have sufficient sequences
        MARKER_FILE={path}/{wildcards.rel}/seq.sufficient
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping pal2nal for {wildcards.rel}: insufficient sequences"
            echo "# Insufficient sequences for codon alignment" > {output.alignment}
        elif grep -q ">" {input.pro_align} 2>/dev/null && grep -q ">" {input.nucl_seq} 2>/dev/null; then
            pal4nal.pl {input.pro_align} {input.nucl_seq} -output fasta -nomismatch -o {output.alignment} || {{
                echo "pal2nal failed for {wildcards.rel}, creating placeholder"
                echo "# pal2nal failed due to sequence mismatches" > {output.alignment}
            }}
        else
            echo "Skipping pal2nal for {wildcards.rel}: empty input files"
            echo "# Empty input files" > {output.alignment}
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
        slurm_partition="cpu",
        runtime=1200,
        mem_mb=8192
    shell:
        """
        # Check if we have sufficient sequences
        MARKER_FILE={path}/{wildcards.rel}/seq.sufficient
        if [ -f "$MARKER_FILE" ] && grep -q "insufficient" "$MARKER_FILE"; then
            echo "Skipping remove_dups for {wildcards.rel}: insufficient sequences"
            echo "# Insufficient sequences for duplicate removal" > {output}
        elif grep -q "^#" {input.aln_codon} || grep -q "^#" {input.tree}; then
            echo "Skipping remove_dups for {wildcards.rel}: invalid input files"
            echo "# Invalid input files" > {output}
        else
            hyphy /home/tu/tu_tu/tu_kmpaj01/hyphy-analyses/remove-duplicates/remove-duplicates.bf --msa {input.aln_codon} --tree {input.tree} --output {output} || {{
                echo "HyPhy remove_dups failed for {wildcards.rel}"
                echo "# HyPhy remove_dups failed" > {output}
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
        if grep -q "NTAX = 1" "{input}" 2>/dev/null; then
            echo "Skipping BUSTED for {wildcards.rel}: only 1 taxon (insufficient)"
            echo '{{"analysis": {{"info": "Only 1 taxon - insufficient for BUSTED analysis"}}}}' > {output.json}
            echo "# Only 1 taxon - insufficient for BUSTED analysis" > {output.log}
        elif grep -q -v "^#" "{input}" 2>/dev/null && [ -s "{input}" ]; then
            ENV=TOLERATE_NUMERICAL_ERRORS=1
            CPU={threads}
            hyphy busted --alignment {input} --output {output.json} > {output.log} || {{
                echo "BUSTED failed for {wildcards.rel}"
                echo '{{"analysis": {{"info": "BUSTED analysis failed"}}}}' > {output.json}
                echo "# BUSTED analysis failed" > {output.log}
            }}
        else
            echo "Skipping BUSTED for {wildcards.rel}: insufficient sequences"
            echo '{{"analysis": {{"info": "Insufficient sequences for BUSTED analysis"}}}}' > {output.json}
            echo "# Insufficient sequences for BUSTED analysis" > {output.log}
        fi
        """
