import pandas as pd
from Bio import Entrez
from Bio import SeqIO
import os
import subprocess


configfile: "config.yaml"


data = config["dataset"]
max_assemblies = config["max_assemblies"]
min_samples = config["min_samples"]
word_size_limit = config["word_size_limit"]

path = f"../preds{data}/REL_output/strains_assemblies.txt"

assemblies = []
strains = []
with open(f"../preds{data}/REL_output/strains_assemblies.txt", "r") as f:
    for line in f:
        s,a = line.strip().split("/")
        strains.append(s)
        assemblies.append(a)


rule final:
    input:
        expand("../assemblies/{strain}/{assembly}/annotation.parquet", zip , strain=strains,assembly=assemblies)

rule download:
    output:
        temp("../assemblies/{strain}/{assembly}.zip"),
    shell:
        "datasets download genome accession {wildcards.assembly} --include gff3,cds,protein,genome,seq-report --filename {output} --assembly-version 'latest'"
        
    

rule unzip:
    input:
        "../assemblies/{strain}/{assembly}.zip",
    output:
        "../assemblies/{strain}/{assembly}/protein.faa",
        "../assemblies/{strain}/{assembly}/genomic.fna",
        "../assemblies/{strain}/{assembly}/genomic.cds",
        "../assemblies/{strain}/{assembly}/genomic.gff",
    shell:
        "unzip -j {input} 'ncbi_dataset/data/*/*.faa' 'ncbi_dataset/data/*/*.fna' 'ncbi_dataset/data/*/*.gff' -d ../assemblies/{wildcards.strain}/{wildcards.assembly}; mv ../assemblies/{wildcards.strain}/{wildcards.assembly}/cds_from_genomic.fna ../assemblies/{wildcards.strain}/{wildcards.assembly}/genomic.cds; mv ../assemblies/{wildcards.strain}/{wildcards.assembly}/*_genomic.fna ../assemblies/{wildcards.strain}/{wildcards.assembly}/genomic.fna"


rule ip:
    input:
        "../assemblies/{strain}/{assembly}/protein.faa",
    output:
        temp("../assemblies/{strain}/{assembly}/annotation.tsv"),
    threads: 2
    shell:
        "/home/tu/tu_tu/tu_bbpgo01/ip/interproscan-5.67-99.0/interproscan.sh -goterms -dra --iprlookup --cpu {threads} -i {input} -o {output} -f TSV -appl Pfam # SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam"


rule convert_to_parquet:
    input:
        "../assemblies/{strain}/{assembly}/annotation.tsv",
    output:
        "../assemblies/{strain}/{assembly}/annotation.parquet",
    threads: 1
    run:
        headers = [
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
        df = pd.read_csv(input[0], sep="\t", na_values=["-", "None"], names=headers)
        df.drop(columns=["Status", "Date"], inplace=True)
        df.to_parquet(output[0])
