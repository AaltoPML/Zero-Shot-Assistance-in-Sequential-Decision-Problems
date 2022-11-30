#!/bin/bash
#SBATCH --job-name=E0_IM_AIAD
#SBATCH --output=/scratch/work/%u/E0_IM_AIAD_%A_%a.out
#SBATCH --array=1-20
#SBATCH --time=5:00:00
#SBATCH --mem=6000

module load julia
srun julia AIADExperiment.jl "inventory_experiment/E0_AIAD_${SLURM_ARRAY_TASK_ID}.jld" inventory_experiment/E0_modeled.jld ${SLURM_ARRAY_TASK_ID}
