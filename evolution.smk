import pandas as pd

max_assemblies = 5
data = 810
folder = f"/home/gomez/gomez/seqfiles_linkbert_{data}_{max_assemblies}"
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
    threads: 1
    shell:
        "mafft --auto --thread 1 {input} > {output}"


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


rule busted:
    input:
        aln_codon=folder + "/{rel}/seq.aln.codon",
        tree=folder + "/{rel}/seq.tree",
    output:
        json=folder + "/{rel}/seq.json",
        log=folder + "/{rel}/seq.log",
    threads: 1
    shell:
        "hyphy busted --alignment {input.aln_codon} --tree {input.tree} --output {output.json} > {output.log}"
