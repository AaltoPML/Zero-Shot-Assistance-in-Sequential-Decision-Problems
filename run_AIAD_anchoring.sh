#!/bin/bash
#SBATCH --job-name=E0_AIAD
#SBATCH --array=1-75
#SBATCH --time=14:00:00
#SBATCH --mem=5000

module load julia
srun julia AIADExperiment.jl "anchoring_experiment/E0_AIAD_${SLURM_ARRAY_TASK_ID}.jld" anchoring_experiment/E0_anchoring_experiment_modeled.jld ${SLURM_ARRAY_TASK_ID}
