# This file implements the optimization of the day trip based on the true reward function of the agent. This is known in the paper as Oracle + automation.
#
#

using Pkg
Pkg.activate("../.")

include("LibTravelPlanner.jl")
using .DaytripDesignAssistance
include("../LibRootSamplingMCTS.jl")
using .RootSamplingMCTS

using POMDPs
using MCTS
using ProgressMeter
using Distributions
using JLD2
using CPUTime
using LRUCache

function find_optimal_trip(home, POIs, true_um_spec, N; value_estimator = 0.0)
    mdp = MachineDaytripScheduleMDP(home, POIs, true_um_spec.first, 0.99)
    solver = DPWSolver(n_iterations=10_000_000, depth=2, exploration_constant=0.1, estimate_value=value_estimator, check_repeat_state = true, enable_action_pw = false, show_progress = false, tree_in_info=false)
    planner = DPWPlanner(solver, mdp);

    s = initialstate(mdp).val

    state_history = [s]
    utility_history = [utility(mdp, s)]
    println("========= 0 (AI) ==========")
    println("starting utility: " * string(utility(mdp, s)))
    for i in 1:N
        println("========= $i (AI) ==========")
        a = actions(mdp, s)[1]
        if (i >= 2) && (state_history[end] == state_history[end-1])
            # default to NOOP action once planner can't find ways of improving the design anymore
        elseif (i >= 4) && (state_history[end] == state_history[end-2]) && (state_history[end-1] == state_history[end-3])
            # default to NOOP action if planner starts to cycle through the same two states
        else
            a = action(planner, s)
        end
        s = @gen(:sp)(mdp, s, a)

        @show a
        @show duration(s), utility(mdp, s)

        push!(state_history, s)
        push!(utility_history, utility(mdp, s))
    end
    
    return state_history, utility_history
end

################################################################################
#                            EXPERIMENT RUN CODE                               #
################################################################################

function show_help()
    println("Finds the optimal daytrip for the true user model of a given experiment")
    println("Usage:    julia ElicitationExperiment.jl OUTFILE SPECFILE DATAFILE STARTPOINT CONTINUE")
    println("    OUTFILE         Specifies the name of the output file. All the results will be written to this file.")
    println("    SPECFILE        Specifies the name of the experiment specification file to be used.")
    println("    SPEC_IDX        Integer. Specifies the index of the experiment scenario within SPECFILE to run.")
    exit(1)
end

function main()
    N_AI = 40

    if length(Base.ARGS) != 3
        @warn "incorrect number of arguements given"
        show_help()
    end

    @load Base.ARGS[2] POIs home true_user_model
    experiment_spec_name = Base.ARGS[3]
    experiment_index = -1
    try
        experiment_index = parse(Int64, Base.ARGS[3])
    catch e
        @show e
        show_help()
    end
    true_user_model_spec = true_user_model[experiment_index]
    POIs = POIs[experiment_index]
    home = home[experiment_index]
    
    state_history, utility_history = find_optimal_trip(home, POIs, true_user_model_spec, N_AI, value_estimator = 0.0)
    @save Base.ARGS[1] experiment_spec_name experiment_index state_history utility_history N_AI
end

main()
