import pandas as pd
from Bio import Entrez
from Bio import SeqIO
import os
import subprocess


configfile: "config.yaml"


data = config["dataset"]
# path = ".."
path = config["output_path"]

assemblies = []
strains = []
with open(f"{path}/preds{data}/REL_output/strains_assemblies.txt", "r") as f:
    for line in f:
        s, a = line.strip().split("/")
        strains.append(s)
        assemblies.append(a)


rule final:
    input:
        expand(
            path + "/assemblies/{strain}/{assembly}/annotation.parquet",
            zip,
            strain=strains,
            assembly=assemblies,
        ),


rule download:
    output:
        temp(path + "/assemblies/{strain}/{assembly}.zip"),
    shell:
        "datasets download genome accession {wildcards.assembly} --include gff3,cds,protein,genome,seq-report --filename {output} --assembly-version 'latest' --api-key 71c734bb92382389e17af918de877c12b308"

rule unzip:
    input:
        path + "/assemblies/{strain}/{assembly}.zip",
    output:
        path + "/assemblies/{strain}/{assembly}/protein.faa",
        path + "/assemblies/{strain}/{assembly}/genomic.fna",
        path + "/assemblies/{strain}/{assembly}/genomic.cds",
        path + "/assemblies/{strain}/{assembly}/genomic.gff",
    shell:
        "unzip -j {input} 'ncbi_dataset/data/*/*.faa' 'ncbi_dataset/data/*/*.fna' 'ncbi_dataset/data/*/*.gff' -d {path}/assemblies/{wildcards.strain}/{wildcards.assembly}; mv {path}/assemblies/{wildcards.strain}/{wildcards.assembly}/cds_from_genomic.fna {path}/assemblies/{wildcards.strain}/{wildcards.assembly}/genomic.cds; mv {path}/assemblies/{wildcards.strain}/{wildcards.assembly}/*_genomic.fna {path}/assemblies/{wildcards.strain}/{wildcards.assembly}/genomic.fna"


rule ip:
    input:
        path + "/assemblies/{strain}/{assembly}/protein.faa",
    output:
        temp(path + "/assemblies/{strain}/{assembly}/annotation.tsv"),
    threads: 2
    shell:
        "/home/tu/tu_tu/tu_bbpgo01/ip/interproscan-5.67-99.0/interproscan.sh -T $TMPDIR -goterms -dra --iprlookup --cpu {threads} -i {input} -o {output} -f TSV -appl Pfam # SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam"


rule convert_to_parquet:
    input:
        path + "/assemblies/{strain}/{assembly}/annotation.tsv",
    output:
        path + "/assemblies/{strain}/{assembly}/annotation.parquet",
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
