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
    "slurm_partition": "cpu",
    "runtime": 300,
    "mem_mb": 4096
}

DOWNLOAD_RESOURCES = {
    "slurm_partition": "cpu",
    "runtime": 600,
    "mem_mb": 2048
}

INTERPROSCAN_RESOURCES = {
    "slurm_partition": "cpu",
    "runtime": 3600,
    "mem_mb": 16384
}

# InterProScan configuration
INTERPROSCAN_PATH = "/home/tu/tu_tu/tu_kmpaj01/ip/interproscan-5.74-105.0/interproscan.sh"
INTERPROSCAN_APPS = "Pfam"  # Can be expanded: SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam

# Load strain/assembly mappings
assemblies = []
strains = []
with open(f"{path}/preds{data}/REL_output/strains_assemblies.txt", "r") as f:
    for line in f:
        s, a = line.strip().split("/")
        strains.append(s)
        assemblies.append(a)

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

rule final:
    input:
        expand(
            output_path + "{strain}/{assembly}/annotation.parquet",
            zip,
            strain=strains,
            assembly=assemblies,
        ),
        expand(
            output_path + "{strain}/{assembly}/genomic.fna.gz",
            zip,
            strain=strains,
            assembly=assemblies,
        ),


rule download:
    output:
        temp(output_path + "{strain}/{assembly}.zip"),
    resources:
        **DOWNLOAD_RESOURCES,
    shell:
        "datasets download genome accession {wildcards.assembly} --include gff3,cds,protein,genome,seq-report --filename {output} --assembly-version 'latest' --api-key $(cat .ncbi_api_key) --fast-zip-validation --no-progressbar" 

rule unzip:
    input:
        output_path + "{strain}/{assembly}.zip",
    output:
        output_path + "{strain}/{assembly}/protein.faa",
        temp(output_path + "{strain}/{assembly}/genomic.fna"),
        output_path + "{strain}/{assembly}/genomic.cds",
        temp(output_path + "{strain}/{assembly}/genomic.gff"),
    resources:
        **COMMON_RESOURCES,
    shell:
        """
        unzip -o -j {input} 'ncbi_dataset/data/*/*.faa' 'ncbi_dataset/data/*/*.fna' 'ncbi_dataset/data/*/*.gff' -d {output_path}/{wildcards.strain}/{wildcards.assembly}
        mv {output_path}/{wildcards.strain}/{wildcards.assembly}/cds_from_genomic.fna {output_path}/{wildcards.strain}/{wildcards.assembly}/genomic.cds
        mv {output_path}/{wildcards.strain}/{wildcards.assembly}/*_genomic.fna {output_path}/{wildcards.strain}/{wildcards.assembly}/genomic.fna
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
        "gzip {input[0]}; gzip {input[1]}"


rule ip:
    input:
        output_path + "{strain}/{assembly}/protein.faa",
    output:
        temp(output_path + "{strain}/{assembly}/annotation.tsv"),
    threads: 4
    resources:
        **INTERPROSCAN_RESOURCES,
    shell:
        "{INTERPROSCAN_PATH} -T $TMPDIR -goterms -dra --iprlookup --cpu {threads} -i {input} -o {output} -f TSV -appl {INTERPROSCAN_APPS}"


rule convert_to_parquet:
    input:
        output_path + "{strain}/{assembly}/annotation.tsv",
    output:
        output_path + "{strain}/{assembly}/annotation.parquet",
    threads: 1
    resources:
        **COMMON_RESOURCES,
    run:
        headers = get_interproscan_headers()
        df = pd.read_csv(input[0], sep="\t", na_values=["-", "None"], names=headers)
        df.drop(columns=["Status", "Date"], inplace=True)
        df.to_parquet(output[0])
