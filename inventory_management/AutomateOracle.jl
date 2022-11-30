# This file implements automation based on the true reward function of the agent.
#
#

using Pkg
Pkg.activate("../.")

include("LibInventoryManager.jl")

include("../LibRootSamplingMCTS.jl")
using .RootSamplingMCTS

using POMDPs
using MCTS
using ProgressMeter
using Distributions
using JLD2
using CPUTime

function entropy(x)
    nonzero = x .> 0.0
    return -sum(log2.(x[nonzero]) .* x[nonzero])
end

function discounted_reward(rewards::Array{Float64,1}, discounting::Float64 = 1.0)
    r_tot = 0
    for (i,r) in enumerate(rewards)
        r_tot += r * discounting^i
    end
    return r_tot
end

import Base.hash
hash(s::InventoryState{Int64}) = UInt(stateindex(InventoryPlanningMDP, s))

function automate(true_user_model::UserSpecification, problem_spec::ProblemSpecification, N::Int64)
    mdp = InventoryPlanningMDP(problem_spec, true_user_model)
    solver = DPWSolver(n_iterations=1_000_000,
                       depth=4,
                       exploration_constant=10.0,
                       estimate_value=0.0,
                       check_repeat_state = true,
                       enable_action_pw = false,
                       show_progress = false,
                       max_time = 15*60.0)
    planner = DPWPlanner(solver, mdp)

    s = initialstate(mdp)
    state_history = []
    reward_history = Float64[]
    for i in 1:N
        println("========= $i (AI) ==========")
        @time a = action(planner, s)
        s,r = @gen(:sp, :r)(mdp, s, a)
        @show r

        push!(state_history, s)
        push!(reward_history, r)
    end
    
    return state_history, reward_history
end

################################################################################
#                            EXPERIMENT RUN CODE                               #
################################################################################

function show_help()
    println("Automates inventory management according to the true reward function and with no user involvement.")
    println("Usage:    julia ElicitationExperiment.jl OUTFILE SPECFILE DATAFILE STARTPOINT CONTINUE")
    println("    OUTFILE         Specifies the name of the output file. All the results will be written to this file.")
    println("    SPECFILE        Specifies the name of the experiment specification file to be used.")
    println("    SPEC_IDX      Specifies the index of the experiment scenario within SPECFILE to run.")
    exit(1)
end

function main()
    N = 50

    if length(Base.ARGS) == 3
        @load Base.ARGS[2] true_user_model problem_spec
        experiment_index = parse(Int64, Base.ARGS[3])
        problem_spec = problem_spec[experiment_index]
        true_user_model = true_user_model[experiment_index]
        experiment_spec_name = Base.ARGS[2]

        state_history, reward_history = automate(true_user_model, problem_spec, N)
        @save Base.ARGS[1] experiment_spec_name experiment_index state_history reward_history
    else
        @warn "incorrect number of arguments given"
        show_help()
    end
end

main()
