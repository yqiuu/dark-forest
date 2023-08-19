#!/bin/bash

#SBATCH --job-name=L210_N4320

#SBATCH --time=5:30:00

#SBATCH --mail-type=ALL --mail-user=balu.srdhr@gmail.com

#SBATCH --nodes=1

#SBATCH --tasks-per-node=1

#SBATCH --mem-per-cpu=20G

#SBATCH --array=2-63

time python 5_add_props_special.py params.yml $SLURM_ARRAY_TASK_ID
