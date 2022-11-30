#!/bin/bash
#SBATCH --job-name=E0_IM_ORACLE
#SBATCH --output=/scratch/work/%u/E0_IM_ORACLE_%A_%a.out
#SBATCH --array=1-20
#SBATCH --time=60:00
#SBATCH --mem=3000

module load julia

srun julia AutomateOracle.jl "inventory_experiment/E0_ORACLE_${SLURM_ARRAY_TASK_ID}.jld" inventory_experiment/E0_modeled.jld ${SLURM_ARRAY_TASK_ID}
