# Code for "Zero-Shot Assistance in Novel Decision Problems"

This repository contains the code implementing the methods and experiments in

Sebastiaan De Peuter, Samuel Kaski (2022)  
**Zero-Shot Assistance in Novel Decision Problems**,  
([arXiv preprint](https://arxiv.org/abs/2202.07364))

## Files Overview
The experiments are implemented in Julia 1.7, using multiple package from the POMDPs.jl ecosystem and JuMP.jl.

The source files are:
- LibTravelPlanner.jl: Implements the trip planning problem (trip scheduling problem in the code), the agent model (user model in the code) and the assistant's decision problem.
- LibBFSSolver.jl: Implements the BFS solver used in the agent model
- LibRootSamplingMCTS.jl: Implements GHPMCP.
- AIADExperiment.jl: Implements the main loop for running AIAD.
- ElicitationExperiment.jl: Implements the main loop for running preference learning and inverse reinforcement learning.
- OptimizeTrip.jl: Implements the main loop for automating based on a posterior obtained from ElicitationExperiment.jl.
- OptimizeTrueTrip.jl: Implements the main loop for automating based on an oracle which tells us the agent's true problem.
- basic_experiment/makeExperimentSpecification.jl: Code to generate an experiment specification for running AIAD on trip planning with unbiased agents.
- anchoring_experiment/makeExperimentSpecification.jl: Code to generate an experiment specification for running AIAD on trip planning with potentially biased agents.

The folder "experiment_data" contain data from the experiment runs we report on in the paper:
- experiment_data/E0_basic_experiment.jld contains the experiment specification we used for the experiment with unbiased agents
- experiment_data/E0_anchoring_experiment_modeled.jld and experiment_data/E0_anchoring_experiment_not_modeled.jld contains the experiment specification we used for the experiment with potentially biased agents
- experiment_data/experiment_data_basic_experiment.jld contains aggregate data from our experiment with unbiased agents. This includes objective value achieved and posterior entropy at different time steps for all methods. This data was used to generate the plots and figures in the paper.
- experiment_data/experiment_data_anchoring_experiment.jld contains aggregate data from our experiment with potentially biased agents. This includes objective value achieved and posterior entropy at different time steps for all methods. This data was used to generate the plots and figures in the paper.

## How to run

First ensure that all packages in Project.toml are installed correctly. Start Julia in the repository root and type
```
using Pkg; Pkg.activate("."); Pkg.instantiate()

```

### To run AIAD for day trip design with unbiased agents
Slurm files for running the experiments have been provided. First run `julia makeExperimentSpecification.jl` from the "basic_experiment" folder. This will generate a file called "E0_basic_experiment.jld" which contains the specification of every experiment that will be ran. Now run the following commands from the repository root:

```
sbatch run_AIAD.sh
sbatch run_IRL_elicitation.sh
sbatch run_PL_elicitation.sh
sbatch run_true_optimization.sh
```
As the names suggest, these commands run AIAD, Inverse Reinforcement Learning, Preference Learning, and automation based on an oracle. The results will be stored in the "basic_experiment" folder as the following files:
- basic_experiment/E0_AIAD_I.jld stores run I of AIAD
- basic_experiment/E0_IRL_I.jld stores run I of Inverse Reinforcement Learning
- basic_experiment/E0_PL_I.jld stores run I of Preference Learning
- basic_experiment/E0_TRUE_I_OPT.jld stores run I of automation based on an oracle

Once all jobs have finished, run the automation experiments:
```
sbatch run_optimization_IRL.sh
sbatch run_optimization_PL.sh
```
The results will be stored in the "basic_experiment". Both run_optimization_IRL.sh and run_optimization_PL.sh contain a parameter called STEP which allows you to set from which time step automation should start. We ran these for STEP values [0,5,10,15,20,25,30]. The results will be stored in:
- basic_experiment/E0_IRL_I_OPT_J.jld stores run I of automation based on an Inverse Reinforcement Learning posterior starting at time step J
- basic_experiment/E0_PL_I_OPT_J.jld stores run I of automation based on a Preference Learning posterior starting at time step J

Finally, run `julia plot_data_basic.jl` to aggregate the data into a file called "experiment_data_basic_experiment" and to plot the statistics of interest.
### To run AIAD for day trip design with agents with potential anchoring biases
Slurm files for running the experiments have been provided. First run `julia makeExperimentSpecification.jl` from the "anchoring_experiment" folder. This will generate two files called "E0_anchoring_experiment_modeled.jld" and "E0_anchoring_experiment_not_modeled.jld". These contains the specification of every experiment that will be ran. Now run the following commands from the repository root:

```
sbatch run_AIAD_anchoring.sh
sbatch run_AIAD_not_modeled_anchoring.sh
sbatch run_IRL_elicitation_anchoring.sh
sbatch run_PL_elicitation_anchoring.sh
sbatch run_true_optimization_anchoring.sh
```
As the names suggest, these commands run AIAD, AIAD without modelling the anchoring bias, Inverse Reinforcement Learning, Preference Learning, and automation based on an oracle. The results will be stored in the "anchoring_experiment" folder as the following files:
- anchoring_experiment/E0_AIAD_I.jld stores run I of AIAD
- anchoring_experiment/E0_AIAD_NM_I.jld stores run I of AIAD without modelling the bias
- anchoring_experiment/E0_IRL_I.jld stores run I of Inverse Reinforcement Learning
- anchoring_experiment/E0_PL_I.jld stores run I of Preference Learning
- anchoring_experiment/E0_TRUE_I_OPT.jld stores run I of automation based on an oracle

Once all jobs have finished, run the automation experiments:
```
sbatch run_optimization_IRL.sh
sbatch run_optimization_PL.sh
```
The results will also be stored in the "anchoring_experiment" folder. Both run_optimization_IRL.sh and run_optimization_PL.sh contain a parameter called STEP which allows you to set from which time step automation should start. We ran these for STEP values [0,5,10,15,20,25,30]. The results will be stored in:
- anchoring_experiment/E0_IRL_I_OPT_J.jld stores run I of automation based on an Inverse Reinforcement Learning posterior starting at time step J
- anchoring_experiment/E0_PL_I_OPT_J.jld stores run I of automation based on a Preference Learning posterior starting at time step J

Finally, run `julia plot_data_anchoring.jl` to aggregate the data into a file called "experiment_data_anchoring_experiment.jld" and to plot the statistics of interest.

## Contact

 * Sebastiaan De Peuter, sebastiaan.depeuter@aalto.fi
 * Samuel Kaski, samuel.kaski@aalto.fi
