#!/bin/bash
#SBATCH --job-name=E0_PL_OPT_15
#SBATCH --output=/scratch/work/%u/E0_PL_OPT_%A_%a.out
#SBATCH --array=1-50
#SBATCH --time=3:30:00
#SBATCH --mem=2000
STEP=15

module load julia

srun julia OptimizeTrip.jl "basic_experiment/E0_PL_${SLURM_ARRAY_TASK_ID}_OPT_${STEP}_RESTART.jld" basic_experiment/E0_basic_experiment.jld "basic_experiment/E0_PL_${SLURM_ARRAY_TASK_ID}.jld" ${STEP} RESTART
