import pandas as pd


configfile: "config.yaml"


data = config["dataset"]

path = f"/home/tu/tu_tu/tu_kmpaj01/link/seqfiles_{data}"
(R,) = glob_wildcards(path + "/{rel}/seq.faa")


rule final:
    input:
        expand(
            path + "/{rel}/seq.json",
            rel=R,
        ),


rule align:
    input:
        path + "/{rel}/seq.faa",
    output:
        path + "/{rel}/seq.aln",
    threads: 5
    shell:
        "mafft --auto --thread 5 {input} > {output}"


rule fasttree:
    input:
        path + "/{rel}/seq.aln",
    output:
        path + "/{rel}/seq.tree",
    shell:
        "fasttree -nosupport {input} > {output}"


rule codonaln:
    input:
        pro_align=path + "/{rel}/seq.aln",
        nucl_seq=path + "/{rel}/seq.fna",
    output:
        alignment=path + "/{rel}/seq.aln.codon",
    shell:
        "pal2nal.pl {input.pro_align} {input.nucl_seq} -output fasta > {output.alignment}"


rule remove_dups:
    input:
        aln_codon=path + "/{rel}/seq.aln.codon",
        tree=path + "/{rel}/seq.tree",
    output:
        path + "/{rel}/seq.nxh",
    shell:
        "hyphy /home/gomez/hyphy-analyses/remove-duplicates/remove-duplicates.bf --msa {input.aln_codon} --tree {input.tree} --output {output}"


rule busted:
    input:
        path + "/{rel}/seq.nxh",
    output:
        json=path + "/{rel}/seq.json",
        log=path + "/{rel}/seq.log",
    threads: 20
    shell:
        """
        ENV=TOLERATE_NUMERICAL_ERRORS=1
        CPU=20
        hyphy busted --alignment {input} --output {output.json} > {output.log}
        """
