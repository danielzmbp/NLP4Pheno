import pandas as pd


configfile: "config.yaml"


data = config["dataset"]

path = f"/home/tu/tu_tu/tu_kmpaj01/link/seqfiles_{data}"
(R,) = glob_wildcards(path + "/{rel}/seq.faa")

localrules: align, codonaln, remove_dups, final


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
    threads: 10
    resources:
        mem_mb=16 * 1024,
        slurm_partition="single",
        runtime=2000,
    shell:
        "fasttree -nosupport {input} > {output}"


rule codonaln:
    input:
        pro_align=path + "/{rel}/seq.aln",
        nucl_seq=path + "/{rel}/seq.fna",
    output:
        alignment=path + "/{rel}/seq.aln.codon",
    shell:
        "pal4nal.pl {input.pro_align} {input.nucl_seq} -output fasta -o {output.alignment}"


rule remove_dups:
    input:
        aln_codon=path + "/{rel}/seq.aln.codon",
        tree=path + "/{rel}/seq.tree",
    output:
        path + "/{rel}/seq.nxh",
    shell:
        "hyphy /home/tu/tu_tu/tu_kmpaj01/hyphy-analyses/remove-duplicates/remove-duplicates.bf --msa {input.aln_codon} --tree {input.tree} --output {output}"


rule busted:
    input:
        path + "/{rel}/seq.nxh",
    output:
        json=path + "/{rel}/seq.json",
        log=path + "/{rel}/seq.log",
    threads: 32
    resources:
        mem_mb=64 * 1024,
        slurm_partition="single",
        runtime=4320,
    shell:
        """
        ENV=TOLERATE_NUMERICAL_ERRORS=1
        CPU=32
        hyphy busted --alignment {input} --output {output.json} > {output.log}
        """
