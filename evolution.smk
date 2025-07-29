import pandas as pd

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
    threads: 1
    resources:
        slurm_partition="cpu",
        runtime=600,
        mem_mb=4096
    shell:
        "mafft --auto --thread {threads} {input} > {output}"


rule fasttree:
    input:
        path + "/{rel}/seq.aln",
    output:
        path + "/{rel}/seq.tree",
    threads: 16
    resources:
        **COMMON_RESOURCES,
    shell:
        "fasttree -nosupport {input} > {output}"


rule codonaln:
    input:
        pro_align=path + "/{rel}/seq.aln",
        nucl_seq=path + "/{rel}/seq.fna",
    output:
        alignment=path + "/{rel}/seq.aln.codon",
    resources:
        slurm_partition="cpu",
        runtime=600,
        mem_mb=4096
    shell:
        "pal4nal.pl {input.pro_align} {input.nucl_seq} -output fasta -o {output.alignment}"


rule remove_dups:
    input:
        aln_codon=path + "/{rel}/seq.aln.codon",
        tree=path + "/{rel}/seq.tree",
    output:
        path + "/{rel}/seq.nxh",
    resources:
        slurm_partition="cpu",
        runtime=1200,
        mem_mb=8192
    shell:
        "hyphy /home/tu/tu_tu/tu_kmpaj01/hyphy-analyses/remove-duplicates/remove-duplicates.bf --msa {input.aln_codon} --tree {input.tree} --output {output}"


rule busted:
    input:
        path + "/{rel}/seq.nxh",
    output:
        json=path + "/{rel}/seq.json",
        log=path + "/{rel}/seq.log",
    threads: 48
    resources:
        **HEAVY_RESOURCES,
    shell:
        """
        ENV=TOLERATE_NUMERICAL_ERRORS=1
        CPU={threads}
        hyphy busted --alignment {input} --output {output.json} > {output.log}
        """
