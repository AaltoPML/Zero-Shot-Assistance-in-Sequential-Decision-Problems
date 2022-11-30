#!/bin/bash
# SBATCH --job-name=E0_AIAD_automate
# SBATCH --output=/scratch/work/%u/E0_AIAD_automate_%A_%a.out
# SBATCH --array=1-75
# SBATCH --time=20:00:00
# SBATCH --mem=5000

module load julia
srun julia AIADExperiment.jl "anchoring_experiment/E0_AIAD_automate_${SLURM_ARRAY_TASK_ID}.jld" anchoring_experiment/E0_anchoring_experiment_modeled.jld ${SLURM_ARRAY_TASK_ID} AUTOMATE