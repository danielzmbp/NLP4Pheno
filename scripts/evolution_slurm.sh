#!/bin/bash
#SBATCH --job-name=evolution_dataset
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=72:00:00
#SBATCH --mem=170G
#SBATCH --output=evolution_dataset_%j.log
#SBATCH --error=evolution_dataset_%j.err

srun python scripts/create_evolution_dataset.py
