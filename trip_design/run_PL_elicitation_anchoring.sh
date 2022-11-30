#!/bin/bash
#SBATCH --job-name=E0_PL
#SBATCH --output=/scratch/work/%u/E0_PL_%A_%a.out
#SBATCH --array=1-75
#SBATCH --time=3:00:00
#SBATCH --mem=2000

module load julia
srun julia ElicitationExperiment.jl PREFERENCE "anchoring_experiment/E0_PL_${SLURM_ARRAY_TASK_ID}.jld" anchoring_experiment/E0_anchoring_experiment_not_modeled.jld ${SLURM_ARRAY_TASK_ID}
