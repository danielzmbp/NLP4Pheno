import pandas as pd


configfile: "config.yaml"


data = config["dataset"]
max_assemblies = config["max_assemblies"]
min_samples = config["min_samples"]
word_size_limit = config["word_size_limit"]

path = f"/home/gomez/gomez/assemblies/{data}/{max_assemblies}_{min_samples}"

(
    S,
    F,
) = glob_wildcards(path + "/{strain}/{f}.fna.gz")


rule final:
    input:
        expand(
            folder + "/{strain}/{f}.parquet",
            zip,
            f=F,
            strain=S,
        ),
        expand(
            folder + "/{strain}/{f}.cds",
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
        "/home/gomez/interproscan-5.67-99.0/interproscan.sh -goterms -dra --iprlookup --cpu {threads} -i {input} -o {output} -f TSV -appl Pfam # SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam"


rule fix_gff:
    input:
        temp(folder + "/{strain}/{f}.gff"),
    output:
        folder + "/{strain}/{f}.gff3",
    threads: 1
    shell:
        """
            awk '{
                if ($0 ~ /^[^#]/ && $3 == "CDS") {
                    # Extract the full ID at the beginning of the line
                    full_id=$1;
                    # Replace the number before "_" in the ID with the full ID, ensuring the part after "_" is preserved
                    sub(/ID=[^;_]+_/, "ID=" full_id "_", $0);
                }
                print $0;
            }' {input} > {output}
        """


rule get_cds:
    input:
        fna=folder + "/{strain}/{f}.fna",
        gff=folder + "/{strain}/{f}.gff3",
    output:
        folder + "/{strain}/{f}.cds",
    threads: 1
    shell:
        "gffread -x {output} -g {input.fna} {input.gff}"


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
