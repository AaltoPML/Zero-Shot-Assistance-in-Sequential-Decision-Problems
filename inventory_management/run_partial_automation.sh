#!/bin/bash
#SBATCH --job-name=E0_IM_AIAD_AO
#SBATCH --output=/scratch/work/%u/E0_IM_AIAD_autom_only_%A_%a.out
#SBATCH --array=1-20
#SBATCH --time=8:00:00
#SBATCH --mem=6000

module load julia
srun julia PartialAutomationExperiment.jl "inventory_experiment/E0_automation_only_${SLURM_ARRAY_TASK_ID}.jld" inventory_experiment/E0_modeled.jld ${SLURM_ARRAY_TASK_ID}
