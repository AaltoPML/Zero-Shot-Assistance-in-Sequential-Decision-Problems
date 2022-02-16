#!/bin/bash
#SBATCH --job-name=E0_IRL_OPT
#SBATCH --array=1-75
#SBATCH --time=3:00:00
#SBATCH --mem=2000
STEP=30

module load julia

srun julia OptimizeTrip.jl "anchoring_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}_OPT_${STEP}_RESTART.jld" anchoring_experiment/E0_anchoring_experiment_modeled.jld "anchoring_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}.jld" ${STEP} RESTART
