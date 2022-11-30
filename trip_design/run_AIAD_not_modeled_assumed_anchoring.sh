#!/bin/bash
# SBATCH --job-name=E0_AIAD_NMA
# SBATCH --output=/scratch/work/%u/E0_AIAD_NMA_%A_%a.out
# SBATCH --array=1-75
# SBATCH --time=15:00:00
# SBATCH --mem=5000

module load julia
srun julia AIADExperiment.jl "anchoring_experiment/E0_AIAD_NMA_${SLURM_ARRAY_TASK_ID}.jld" anchoring_experiment/E0_foviation_experiment_not_modeled_assumed.jld ${SLURM_ARRAY_TASK_ID}