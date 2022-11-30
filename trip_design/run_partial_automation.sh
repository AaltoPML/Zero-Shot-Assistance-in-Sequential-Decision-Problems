#!/bin/bash
#SBATCH --array=1-50
#SBATCH --time=15:00:00
#SBATCH --mem=4000

module load julia
srun julia PartialAutomationExperiment.jl "basic_experiment/E0_automation_only_${SLURM_ARRAY_TASK_ID}.jld" basic_experiment/E0_basic_experiment.jld ${SLURM_ARRAY_TASK_ID}
