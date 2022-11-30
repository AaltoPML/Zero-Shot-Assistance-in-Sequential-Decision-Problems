#!/bin/bash
#SBATCH --array=1-75
#SBATCH --time=14:00:00
#SBATCH --mem=5000

module load julia
srun julia PartialAutomationExperiment.jl "anchoring_experiment/E0_automation_only_${SLURM_ARRAY_TASK_ID}.jld" anchoring_experiment/E0_anchoring_experiment_modeled.jld ${SLURM_ARRAY_TASK_ID}
