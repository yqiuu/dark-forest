#!/bin/bash

#SBATCH --job-name=L210N1536

#SBATCH --time=0:35:00


#SBATCH --nodes=1

#SBATCH --mail-type=ALL --mail-user=bsreedhar@student.unimelb.edu.au

#SBATCH --tasks-per-node=1

#SBATCH --mem-per-cpu=20G

#SBATCH --array=0

time python3 1_create_dforest.py
#time python 4_add_forward_fix.py params.yml $SLURM_ARRAY_TASK_ID
#time python 5_add_props_special.py params.yml $SLURM_ARRAY_TASK_ID
#time python 6_save.py params.yml
