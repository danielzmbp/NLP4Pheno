sbatch -p single -n 40 -t 2-00:00:00 --mem=180000 snakemake -s ip.smk --cores 80 -k
