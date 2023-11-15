#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name="MLP_118bus_N3_16neurons"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=Education-EEMCS-MSc-SET

module load 2023r1
module load openmpi
module load python
module load py-mpi4py

srun python MLP_118bus_N3_16neurons.py > MLP_118bus_N3_16neurons_results.log