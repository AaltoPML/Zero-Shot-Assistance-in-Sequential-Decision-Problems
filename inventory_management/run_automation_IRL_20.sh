#!/bin/bash
#SBATCH --job-name=E0_IRL_IM_OPT_20
#SBATCH --output=/scratch/work/%u/E0_IRL_IM_OPT_20_%A_%a.out
#SBATCH --array=1-20
#SBATCH --time=4:00:00
#SBATCH --mem=1500
STEP=20

module load julia

srun julia Automate.jl "inventory_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}_OPT_${STEP}.jld" inventory_experiment/E0_modeled.jld "inventory_experiment/E0_IRL_${SLURM_ARRAY_TASK_ID}.jld" ${STEP}
