#!/bin/bash
#SBATCH --job-name=E0_PL_OPT_30
#SBATCH --output=/scratch/work/%u/E0_PL_OPT_%A_%a.out
#SBATCH --array=1-75
#SBATCH --time=3:30:00
#SBATCH --mem=2000
STEP=30

module load julia

srun julia OptimizeTrip.jl "anchoring_experiment/E0_PL_${SLURM_ARRAY_TASK_ID}_OPT_${STEP}_RESTART.jld" anchoring_experiment/E0_anchoring_experiment_not_modeled.jld "anchoring_experiment/E0_PL_${SLURM_ARRAY_TASK_ID}.jld" ${STEP} RESTART
