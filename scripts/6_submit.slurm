#!/bin/bash

#SBATCH --job-name=L210N1536

#SBATCH --time=40:00:00


#SBATCH --nodes=1

#SBATCH --mail-type=ALL --mail-user=balu.srdhr@gmail.com

#SBATCH --tasks-per-node=32

#SBATCH --mem=250G

time python 6_save.py params.yml
