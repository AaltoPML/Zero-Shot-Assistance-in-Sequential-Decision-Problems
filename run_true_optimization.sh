#!/bin/bash
#SBATCH --job-name=E0_TRUE
#SBATCH --array=1-50
#SBATCH --time=3:00:00
#SBATCH --mem=2000

module load julia
srun julia OptimizeTrueTrip.jl "basic_experiment/E0_TRUE_${SLURM_ARRAY_TASK_ID}_OPT.jld" basic_experiment/E0_basic_experiment.jld ${SLURM_ARRAY_TASK_ID}
