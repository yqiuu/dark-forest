#!/bin/bash

#SBATCH --job-name=L210_N4320

#SBATCH --time=1:20:00

#SBATCH --mail-type=ALL --mail-user=balu.srdhr@gmail.com

#SBATCH --nodes=1

#SBATCH --tasks-per-node=32

#SBATCH --mem-per-cpu=5G

#SBATCH --array=0-63

time mpiexec -n 32 python3 3_grow.py params.yml $SLURM_ARRAY_TASK_ID
