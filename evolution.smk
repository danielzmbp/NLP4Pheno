import pandas as pd

max_assemblies = 5
data = 810
min_samples = 21

folder = f"/home/gomez/gomez/seqfiles_linkbert_{data}_{max_assemblies}_{min_samples}"
(R,) = glob_wildcards(folder + "/{rel}/seq.faa")


rule final:
    input:
        expand(
            folder + "/{rel}/seq.json",
            rel=R,
        ),


rule align:
    input:
        folder + "/{rel}/seq.faa",
    output:
        folder + "/{rel}/seq.aln",
    threads: 5
    shell:
        "mafft --auto --thread 5 {input} > {output}"


rule fasttree:
    input:
        folder + "/{rel}/seq.aln",
    output:
        folder + "/{rel}/seq.tree",
    shell:
        "fasttree -nosupport {input} > {output}"


rule codonaln:
    input:
        pro_align=folder + "/{rel}/seq.aln",
        nucl_seq=folder + "/{rel}/seq.fna",
    output:
        alignment=folder + "/{rel}/seq.aln.codon",
    shell:
        "pal2nal.pl {input.pro_align} {input.nucl_seq} -output fasta > {output.alignment}"


rule remove_dups:
    input:
        aln_codon=folder + "/{rel}/seq.aln.codon",
        tree=folder + "/{rel}/seq.tree",
    output:
        folder + "/{rel}/seq.nxh",
    shell:
        "hyphy /home/gomez/hyphy-analyses/remove-duplicates/remove-duplicates.bf --msa {input.aln_codon} --tree {input.tree} --output {output}"


rule busted:
    input:
        folder + "/{rel}/seq.nxh",
    output:
        json=folder + "/{rel}/seq.json",
        log=folder + "/{rel}/seq.log",
    threads: 20
    shell:
        """
        ENV=TOLERATE_NUMERICAL_ERRORS=1
        CPU=20
        hyphy busted --alignment {input} --output {output.json} > {output.log}
        """
