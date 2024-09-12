#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 8            # number of cores 
#SBATCH -t 0-18:00:00   # time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest

# Using python, so source activate an appropriate environment
source activate cse507_practice_1_env

python /home/rawhad/CSE507/practice_1/preprocess_chestxray.py

