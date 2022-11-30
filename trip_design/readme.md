# AI-Advised Decision Making for Day Trip Design

## Files Overview
- experiment_data/E0_basic_experiment.jld contains the experiment specification we used for the experiment with unbiased agents
- experiment_data/E0_anchoring_experiment_modeled.jld, experiment_data/E0_anchoring_experiment_not_modeled.jld and experiment_data/E0_anchoring_experiment_not_modeled_assumed.jld contains the experiment specification we used for the experiment with potentially biased agents
- basic_experiment/experiment_data_basic_experiment.jld contains aggregate data from our experiment with unbiased agents. This includes objective value achieved and posterior entropy at different time steps for all methods. This data was used to generate the plots and figures in the paper.
- anchoring_experiment/experiment_data_anchoring_experiment.jld contains aggregate data from our experiment with potentially biased agents. This includes objective value achieved and posterior entropy at different time steps for all methods. This data was used to generate the plots and figures in the paper.

## How to run

First ensure that all packages in Project.toml are installed correctly. Start Julia in this folder and type:
```
using Pkg; Pkg.activate("../."); Pkg.instantiate()
```

### AIAD for day trip design with potentially biased agents

Slurm files for running the experiments have been provided. First run `julia makeExperimentSpecification.jl` from the "anchoring_experiment" folder. This will generate two files called "E0_anchoring_experiment_modeled.jld", "E0_anchoring_experiment_not_modeled.jld" and "E0_anchoring_experiment_not_modeled_assumed.jld". These contains the specification of every experiment that will be ran. 

To run the first part of the main experiments run the following commands from this folder:
```
sbatch run_AIAD_anchoring.sh
sbatch run_AIAD_automation_anchoring.sh
sbatch run_IRL_elicitation_anchoring.sh
sbatch run_PL_elicitation_anchoring.sh
sbatch run_true_optimization_anchoring.sh
sbatch run_partial_automation_anchoring.sh
```

Once all jobs have finished, run the automation experiments:
```
sbatch run_optimization_IRL_anchoring_0.sh
sbatch run_optimization_IRL_anchoring_5.sh
sbatch run_optimization_IRL_anchoring_10.sh
sbatch run_optimization_IRL_anchoring_15.sh
sbatch run_optimization_IRL_anchoring_20.sh
sbatch run_optimization_IRL_anchoring_25.sh
sbatch run_optimization_IRL_anchoring_30.sh
sbatch run_optimization_PL_anchoring_5.sh
sbatch run_optimization_PL_anchoring_10.sh
sbatch run_optimization_PL_anchoring_15.sh
sbatch run_optimization_PL_anchoring_20.sh
sbatch run_optimization_PL_anchoring_25.sh
sbatch run_optimization_PL_anchoring_30.sh
```

Next, run the ablation study experiments:
```
sbatch run_AIAD_not_modeled_anchoring.sh
sbatch run_AIAD_not_modeled_assumed_anchoring.sh
```
Finally, run `julia plot_data_anchoring.jl` to aggregate the data into a file called "experiment_data_anchoring_experiment.jld" and to plot the statistics of interest.

### AIAD for day trip design with unbiased agents
Slurm files for running the experiments have been provided. First run `julia makeExperimentSpecification.jl` from the "basic_experiment" folder. This will generate a file called "E0_basic_experiment.jld" which contains the specification of every experiment that will be ran. Now run the following commands from this folder:

```
sbatch run_AIAD.sh
sbatch run_AIAD_automation.sh
sbatch run_IRL_elicitation.sh
sbatch run_PL_elicitation.sh
sbatch run_true_optimization.sh
sbatch run_partial_automation.sh
```

Once all jobs have finished, run the automation experiments:
```
sbatch run_optimization_IRL_0.sh
sbatch run_optimization_IRL_5.sh
sbatch run_optimization_IRL_10.sh
sbatch run_optimization_IRL_15.sh
sbatch run_optimization_IRL_20.sh
sbatch run_optimization_IRL_25.sh
sbatch run_optimization_IRL_30.sh
sbatch run_optimization_PL_5.sh
sbatch run_optimization_PL_10.sh
sbatch run_optimization_PL_15.sh
sbatch run_optimization_PL_20.sh
sbatch run_optimization_PL_25.sh
sbatch run_optimization_PL_30.sh
```

Finally, run `julia plot_data_basic.jl` to aggregate the data into a file called "experiment_data_basic_experiment" and to plot the statistics of interest.

