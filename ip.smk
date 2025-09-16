import pandas as pd
from Bio import Entrez
from Bio import SeqIO
import os
import subprocess

configfile: "config.yaml"

data = config["dataset"]
path = config["output_path"]
output_path = path + f"/assemblies_{data}/"

# Common resource configurations
COMMON_RESOURCES = {
    "slurm_partition": "cpu_il,cpu",
    "runtime": 300,
    "mem_mb": 4096
}

DOWNLOAD_RESOURCES = {
    "slurm_partition": "cpu_il,cpu",
    "runtime": 60,
    "mem_mb": 2048
}

INTERPROSCAN_RESOURCES = {
    "slurm_partition": "cpu_il,cpu",
    "runtime": 3600,
    "mem_mb": 16384
}

# InterProScan configuration
INTERPROSCAN_PATH = "/home/tu/tu_tu/tu_kmpaj01/ip/interproscan-5.75-106.0/interproscan.sh"
INTERPROSCAN_APPS = "Pfam"  # Can be expanded: SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam

# Load strain/assembly mappings
assemblies = []
strains = []
strains_assemblies_file = f"{path}/preds{data}/REL_output/strains_assemblies.txt"

if os.path.exists(strains_assemblies_file):
    with open(strains_assemblies_file, "r") as f:
        for line in f:
            s, a = line.strip().split("/")
            strains.append(s)
            assemblies.append(a)
else:
    # File doesn't exist yet - this is expected when rel_pred.smk hasn't run
    print(f"Warning: {strains_assemblies_file} not found. Run rel_pred.smk first.")

def get_interproscan_headers():
    """Return standard InterProScan TSV headers"""
    return [
        "Protein_accession",
        "Sequence_MD5_digest",
        "Sequence_length",
        "Analysis",
        "Signature_accession",
        "Signature_description",
        "Start_location",
        "Stop_location",
        "Score",
        "Status",
        "Date",
        "InterPro_accession",
        "InterPro_description",
        "GO_annotations",
        "Pathways_annotations",
    ]


def get_successful_targets(wildcards):
    """Get targets for successfully downloaded assemblies"""
    checkpoint_output = checkpoints.attempt_all_downloads.get(**wildcards).output[0]
    
    successful_pairs = []
    with open(checkpoint_output, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                successful_pairs.append(line)
    
    # Generate target files for successful assemblies
    annotation_targets = []
    genomic_targets = []
    
    for pair in successful_pairs:
        strain, assembly = pair.split("/")
        annotation_targets.append(f"{output_path}{strain}/{assembly}/annotation.parquet")
        genomic_targets.append(f"{output_path}{strain}/{assembly}/genomic.fna.gz")
    
    return annotation_targets + genomic_targets

rule final:
    input:
        get_successful_targets,
        f"{path}/preds{data}/REL_output/strains_assemblies_downloaded.txt"


localrules: download, unzip

rule download:
    output:
        output_path + "{strain}/{assembly}.zip",
        output_path + "{strain}/{assembly}.download_success",
    resources:
        **DOWNLOAD_RESOURCES,
    shell:
        """
        mkdir -p $(dirname {output[1]})
        
        # Try download up to 3 times with increasing delays
        for attempt in 1 2 3; do
            echo "Download attempt $attempt for {wildcards.assembly}"
            
            if datasets download genome accession {wildcards.assembly} --include gff3,cds,protein,genome,seq-report --filename {output[0]} --assembly-version 'latest' --api-key $(cat .ncbi_api_key) --fast-zip-validation --no-progressbar; then
                echo "Successfully downloaded {wildcards.assembly} on attempt $attempt"
                touch {output[1]}
                exit 0
            else
                echo "Attempt $attempt failed for {wildcards.assembly}" >&2
                if [ $attempt -lt 3 ]; then
                    sleep_time=$((attempt * 5))
                    echo "Waiting $sleep_time seconds before retry..." >&2
                    sleep $sleep_time
                    # Clean up partial download
                    rm -f {output[0]}
                fi
            fi
        done
        
        echo "All download attempts failed for {wildcards.assembly}" >&2
        touch {output[0]}  # Create empty zip file for failed downloads
        echo "DOWNLOAD_FAILED" > {output[1]}
        exit 0
        """ 

rule unzip:
    input:
        zip_file=output_path + "{strain}/{assembly}.zip",
        success_marker=output_path + "{strain}/{assembly}.download_success",
    output:
        output_path + "{strain}/{assembly}/protein.faa",
        temp(output_path + "{strain}/{assembly}/genomic.fna"),
        output_path + "{strain}/{assembly}/genomic.cds",
        temp(output_path + "{strain}/{assembly}/genomic.gff"),
    resources:
        **COMMON_RESOURCES,
    shell:
        """
        # Check if download was successful
        if [ ! -f {input.success_marker} ] || grep -q "DOWNLOAD_FAILED" {input.success_marker}; then
            echo "Creating empty placeholder files for {wildcards.assembly} - download failed"
            mkdir -p {output_path}{wildcards.strain}/{wildcards.assembly}
            touch {output[0]} {output[1]} {output[2]} {output[3]}
            exit 0
        fi
        
        # Create output directory
        mkdir -p {output_path}{wildcards.strain}/{wildcards.assembly}
        
        # Extract all files to output directory
        unzip -o {input.zip_file} -d {output_path}{wildcards.strain}/{wildcards.assembly}/
        
        # Find the actual assembly directory (handles version suffixes like .1, .2, etc.)
        assembly_dir=$(find {output_path}{wildcards.strain}/{wildcards.assembly}/ncbi_dataset/data -maxdepth 1 -type d -name "{wildcards.assembly}*" | head -1)
        
        if [ -z "$assembly_dir" ]; then
            echo "Warning: Could not find assembly directory for {wildcards.assembly}"
            touch {output[0]} {output[1]} {output[2]} {output[3]}
            exit 0
        fi
        
        # Handle protein.faa
        if [ -f "$assembly_dir/protein.faa" ]; then
            cp "$assembly_dir/protein.faa" {output[0]}
        else
            touch {output[0]}
        fi
        
        # Handle genomic.fna - find the actual genomic file
        genomic_file=$(find "$assembly_dir" -name "*_genomic.fna" | head -1)
        if [ -n "$genomic_file" ] && [ -f "$genomic_file" ]; then
            cp "$genomic_file" {output[1]}
        else
            touch {output[1]}
        fi
        
        # Handle cds_from_genomic.fna -> genomic.cds
        if [ -f "$assembly_dir/cds_from_genomic.fna" ]; then
            cp "$assembly_dir/cds_from_genomic.fna" {output[2]}
        else
            touch {output[2]}
        fi
        
        # Handle genomic.gff
        if [ -f "$assembly_dir/genomic.gff" ]; then
            cp "$assembly_dir/genomic.gff" {output[3]}
        else
            touch {output[3]}
        fi
        
        # Clean up extracted directory structure
        rm -rf {output_path}{wildcards.strain}/{wildcards.assembly}/ncbi_dataset
        rm -f {output_path}{wildcards.strain}/{wildcards.assembly}/README.md
        rm -f {output_path}{wildcards.strain}/{wildcards.assembly}/md5sum.txt
        """
        
rule compress_fna_gff:
    input:
        output_path + "{strain}/{assembly}/genomic.fna",
        output_path + "{strain}/{assembly}/genomic.gff",
    output:
        output_path + "{strain}/{assembly}/genomic.gff.gz",
        output_path + "{strain}/{assembly}/genomic.fna.gz",
    resources:
        **COMMON_RESOURCES,
    shell:
        """
        # Check if files are empty (from failed downloads)
        if [ ! -s {input[0]} ] || [ ! -s {input[1]} ]; then
            echo "Creating empty compressed files for {wildcards.assembly} - original files were empty"
            touch {output[0]} {output[1]}
        else
            gzip {input[0]}; gzip {input[1]}
        fi
        """


rule ip:
    input:
        output_path + "{strain}/{assembly}/protein.faa",
    output:
        temp(output_path + "{strain}/{assembly}/annotation.tsv"),
    threads: 4
    resources:
        **INTERPROSCAN_RESOURCES,
    shell:
        """
        # Check if protein file is empty (from failed downloads)
        if [ ! -s {input} ]; then
            echo "Creating empty annotation file for {wildcards.assembly} - protein file was empty"
            touch {output}
        else
            {INTERPROSCAN_PATH} -T $TMPDIR -goterms -dra --iprlookup --cpu {threads} -i {input} -o {output} -f TSV -appl {INTERPROSCAN_APPS}
        fi
        """


rule convert_to_parquet:
    input:
        output_path + "{strain}/{assembly}/annotation.tsv",
    output:
        output_path + "{strain}/{assembly}/annotation.parquet",
    threads: 1
    resources:
        **COMMON_RESOURCES,
    run:
        import os
        headers = get_interproscan_headers()
        
        # Check if annotation file is empty (from failed downloads)
        if os.path.getsize(input[0]) == 0:
            print(f"Creating empty parquet for {wildcards.assembly} - annotation file was empty")
            # Create empty DataFrame with correct schema
            df = pd.DataFrame(columns=headers)
            df.drop(columns=["Status", "Date"], inplace=True)
        else:
            df = pd.read_csv(input[0], sep="\t", na_values=["-", "None"], names=headers)
            df.drop(columns=["Status", "Date"], inplace=True)
        
        df.to_parquet(output[0])


checkpoint attempt_all_downloads:
    input:
        expand(
            output_path + "{strain}/{assembly}.download_success",
            zip,
            strain=strains,
            assembly=assemblies,
        ),
    output:
        f"{path}/preds{data}/REL_output/strains_assemblies_downloaded.txt"
    resources:
        **COMMON_RESOURCES,
    run:
        successful_assemblies = []
        failed_assemblies = []
        
        for strain, assembly in zip(strains, assemblies):
            success_file = f"{output_path}{strain}/{assembly}.download_success"
            if os.path.exists(success_file):
                with open(success_file, 'r') as f:
                    content = f.read().strip()
                    if content != "DOWNLOAD_FAILED":
                        # Check if zip file was successfully downloaded
                        zip_file = f"{output_path}{strain}/{assembly}.zip"
                        if os.path.exists(zip_file) and os.path.getsize(zip_file) > 0:
                            successful_assemblies.append(f"{strain}/{assembly}")
                        else:
                            failed_assemblies.append(f"{strain}/{assembly}")
                    else:
                        failed_assemblies.append(f"{strain}/{assembly}")
            else:
                failed_assemblies.append(f"{strain}/{assembly}")
        
        # Write successful assemblies file
        with open(output[0], "w") as f:
            for assembly in successful_assemblies:
                f.write(f"{assembly}\n")
        
        # Log results
        print(f"Successfully downloaded: {len(successful_assemblies)} assemblies")
        print(f"Failed downloads: {len(failed_assemblies)} assemblies")
        if failed_assemblies:
            print("Failed assemblies:", failed_assemblies[:10])  # Show first 10
