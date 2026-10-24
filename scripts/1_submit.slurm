#!/bin/bash

#SBATCH --job-name=L210_N4320

#SBATCH --time=2:05:00

#SBATCH --mail-type=ALL --mail-user=balu.srdhr@gmail.com

#SBATCH --nodes=1

#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G

time python3 1_create_dforest.py
