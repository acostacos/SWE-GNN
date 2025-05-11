#!/bin/sh
#SBATCH --job-name=swe_gnn_data
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=64000

. ../venv/bin/activate

srun python create_dataset_from_hecras.py