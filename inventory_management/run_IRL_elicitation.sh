#!/bin/bash
#SBATCH --job-name=E0_IM_IRL
#SBATCH --output=/scratch/work/%u/E0_IM_IRL_%A_%a.out
#SBATCH --array=1-20
#SBATCH --time=60:00
#SBATCH --mem=1000

module load julia
srun julia ElicitationExperiment.jl IRL "inventory_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}.jld" inventory_experiment/E0_modeled.jld ${SLURM_ARRAY_TASK_ID}
