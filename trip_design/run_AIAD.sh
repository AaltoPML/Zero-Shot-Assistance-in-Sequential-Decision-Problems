#!/bin/bash
#SBATCH --job-name=E0_AIAD
#SBATCH --output=/scratch/work/%u/E0_AIAD_%A_%a.out
#SBATCH --array=1-50
#SBATCH --time=15:00:00
#SBATCH --mem=4000

module load julia
srun julia AIADExperiment.jl "basic_experiment/E0_AIAD_${SLURM_ARRAY_TASK_ID}.jld" basic_experiment/E0_basic_experiment.jld ${SLURM_ARRAY_TASK_ID}
