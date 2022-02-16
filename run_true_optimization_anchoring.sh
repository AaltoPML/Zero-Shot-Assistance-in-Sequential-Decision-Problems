#!/bin/bash
#SBATCH --job-name=E0_TRUE
#SBATCH --array=1-75
#SBATCH --time=3:00:00
#SBATCH --mem=2000

module load julia
srun julia OptimizeTrueTrip.jl "anchoring_experiment/E0_TRUE_${SLURM_ARRAY_TASK_ID}_OPT.jld" anchoring_experiment/E0_anchoring_experiment_modeled.jld ${SLURM_ARRAY_TASK_ID}
