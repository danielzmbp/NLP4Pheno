sbatch -p single -n 20 -t 72:00:00 --mem=90000 snakemake -s ip.smk --cores 40 -k --rerun-incomplete
