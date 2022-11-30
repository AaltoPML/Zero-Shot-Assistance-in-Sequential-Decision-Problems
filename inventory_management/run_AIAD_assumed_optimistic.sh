#!/bin/bash                                                                                                                                                                                   
#SBATCH --job-name=E0_IM_AIAD_optim                                                                                                                                                           
#SBATCH --output=/scratch/work/%u/E0_IM_AIAD_optim_%A_%a.out                                                                                                                                  
#SBATCH --array=1-20                                                                                                                                                                     
#SBATCH --time=8:00:00                                                                                                                                                                        
#SBATCH --mem=6000                                                                                                                                                                            

module load julia
srun julia AIADExperiment.jl "inventory_experiment/E0_AIAD_assumed_optimistic_${SLURM_ARRAY_TASK_ID}.jld" inventory_experiment/E0_not_modeled_assumed_optimistic.jld ${SLURM_ARRAY_TASK_ID}
