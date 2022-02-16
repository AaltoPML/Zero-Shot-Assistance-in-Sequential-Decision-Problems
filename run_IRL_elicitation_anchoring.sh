#!/bin/bash
#SBATCH --job-name=E0_IRL
#SBATCH --array=1-75
#SBATCH --time=10:00:00
#SBATCH --mem=2000

module load julia
srun julia ElicitationExperiment.jl IRL "anchoring_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}.jld" anchoring_experiment/E0_anchoring_experiment_modeled.jld ${SLURM_ARRAY_TASK_ID}
