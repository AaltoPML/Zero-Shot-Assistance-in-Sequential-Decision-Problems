#!/bin/bash
#SBATCH --job-name=E0_IRL
#SBATCH --output=/scratch/work/%u/E0_IRL_%A_%a.out
#SBATCH --array=1-50
#SBATCH --time=8:00:00
#SBATCH --mem=2000

module load julia
srun julia ElicitationExperiment.jl IRL "basic_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}.jld" basic_experiment/E0_basic_experiment.jld ${SLURM_ARRAY_TASK_ID}
