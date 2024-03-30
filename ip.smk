import pandas as pd

# TODO: this should be extended to the download and filter scripts

folder = "/home/gomez/gomez/assemblies_linkbert_500_filtered_5"
(
    S,
    F,
) = glob_wildcards(folder + "/{strain}/{f}.fna.gz")


rule final:
    input:
        expand(
            folder + "/{strain}/{f}.parquet",
            zip,
            f=F,
            strain=S,
        ),


rule unzip:
    input:
        folder + "/{strain}/{f}.fna.gz",
    output:
        folder + "/{strain}/{f}.fna",
    threads: 1
    shell:
        "gunzip -c {input} > {output}"


rule annotate:
    input:
        folder + "/{strain}/{f}.fna",
    output:
        fasta=temp(folder + "/{strain}/{f}.fasta"),
        gff=folder + "/{strain}/{f}.gff",
    threads: 1
    shell:
        "prodigal -f 'gff' -q -a {output.fasta} -i {input} -o {output.gff}"


rule replace_asterisks:
    input:
        folder + "/{strain}/{f}.fasta",
    output:
        folder + "/{strain}/{f}.faa",
    threads: 1
    shell:
        "sed 's/*//g' {input} > {output}"


rule ip:
    input:
        folder + "/{strain}/{f}.faa",
    output:
        temp(folder + "/{strain}/{f}.tsv"),
    threads: 5
    shell:
        "/home/gomez/interproscan-5.67-99.0/interproscan.sh -goterms --iprlookup -pa --cpu {threads} -i {input} -o {output} -f TSV -appl SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam"


rule convert_to_parquet:
    input:
        folder + "/{strain}/{f}.tsv",
    output:
        folder + "/{strain}/{f}.parquet",
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
