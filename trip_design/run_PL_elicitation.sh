#!/bin/bash
#SBATCH --job-name=E0_PL
#SBATCH --output=/scratch/work/%u/E0_PL_%A_%a.out
#SBATCH --array=1-50
#SBATCH --time=3:00:00
#SBATCH --mem=2000

module load julia
srun julia ElicitationExperiment.jl PREFERENCE "basic_experiment/E0_PL_${SLURM_ARRAY_TASK_ID}.jld" basic_experiment/E0_basic_experiment.jld ${SLURM_ARRAY_TASK_ID}
