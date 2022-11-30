# This file implements the automation part of the IRL + automation baseline. You can start automating from any point in the trajectory collected by ElicitationExperiment.jl
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

mutable struct RMDSMDP <: MDP{InventoryState{Int64}, ProduceProductsAction}
    underlying_MDP::InventoryPlanningMDP
    underlying_MDP_idx::Int64

    problem_spec::ProblemSpecification
    user_models::Array{UserSpecification,1}
    weights::Array{Float64,1} # unnormalized log-probabilities for elements of user_models

    function RMDSMDP(user_models::Array{UserSpecification,1}, weights::Array{Float64,1}, problem_spec::ProblemSpecification)
        @assert size(user_models) == size(weights)
        idx = argmax(weights .+ rand(Gumbel(), length(user_models)))
        ret = new(InventoryPlanningMDP(problem_spec, user_models[idx]), idx, problem_spec, user_models, weights)
        return ret
    end
end

POMDPs.initialstate(m::RMDSMDP) = POMDPs.initialstate(m.underlying_MDP)

function POMDPs.gen(m::RMDSMDP, s::InventoryState{Int64}, a::ProduceProductsAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    idx = argmax(m.weights .+ rand(Gumbel(), length(m.user_models)))
    m.underlying_MDP = InventoryPlanningMDP(m.problem_spec, m.user_models[idx])
    m.underlying_MDP_idx = idx

    return POMDPs.gen(m.underlying_MDP, s, a, rng)
end

POMDPs.actions(m::RMDSMDP, s::InventoryState{Int64}) = POMDPs.actions(m.underlying_MDP, s)
POMDPs.actions(m::RMDSMDP) = POMDPs.actions(m.underlying_MDP)
POMDPs.discount(m::RMDSMDP) = POMDPs.discount(m.underlying_MDP)

import Base.hash
hash(s::InventoryState{Int64}) = UInt(stateindex(InventoryPlanningMDP, s))

function automate(true_user_model::UserSpecification, user_models::Array{UserSpecification,1}, weights::Array{Float64,1}, problem_spec::ProblemSpecification, s::InventoryState{Int64}, N::Int64)
    mdp = RMDSMDP(user_models, weights, problem_spec)
    solver = DPWSolver(n_iterations=1_000_000,
                       depth=4,
                       exploration_constant=10.0,
                       estimate_value=0.0,
                       check_repeat_state = true,
                       enable_action_pw = false,
                       show_progress = false,
                       max_time = 15*60.0)
    true_objective_mdp = InventoryPlanningMDP(problem_spec, true_user_model)
    planner = DPWPlanner(solver, mdp)

    state_history = []
    reward_history = Float64[]
    for i in 1:N
        println("========= $i (AI) ==========")
        @time a = action(planner, s)
        s,r = @gen(:sp, :r)(true_objective_mdp, s, a)
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
    println("Automates inventory management from any point in a prior AIAD or Elicitation Experiment.")
    println("Usage:    julia ElicitationExperiment.jl OUTFILE SPECFILE DATAFILE STARTPOINT [RESTART N_AI]")
    println("    OUTFILE         Specifies the name of the output file. All the results will be written to this file.")
    println("    SPECFILE        Specifies the name of the experiment specification file to be used.")
    println("    DATAFILE        Specifies the name of of the datafile produced by either AIADExperiment or ElicitationExperiment to be used for weights and starting state.")
    println("    STARTPOINT      Integer. Specifies after how many interactions in DATAFILE the optimizer should start It will continue for the length of the episode, unless RESTART is specified.")
    println("    \"RESTART\"     If this is present, automation will restart from 0.")
    println("    N_AI            Integer. Only used if RESTART is present. Indicates for how many steps to automate.")
    exit(1)
end

function main()
    if length(Base.ARGS) in [4,6]

        restart = (length(Base.ARGS) == 6) && (Base.ARGS[5] == "RESTART")

        if restart
            @load Base.ARGS[3] state_history posterior_history experiment_index experiment_spec_name
        else
            @load Base.ARGS[3] state_history reward_history posterior_history experiment_index experiment_spec_name
        end

        @load Base.ARGS[2] user_models true_user_model problem_spec
        user_models = user_models[experiment_index]
        problem_spec = problem_spec[experiment_index]
        true_user_model = true_user_model[experiment_index]
        datafile_name = Base.ARGS[3]

        if experiment_spec_name != Base.ARGS[2]
            @warn "specfile name does not correspond to datafile, are you sure this is correct?"
        end

        N_h = 0
        try
            N_h = parse(Int64, Base.ARGS[4])
        catch e
            @show e
            show_help()
        end

        if restart
            s = state_history[1]
            N_AI = 50
            try
                N_AI = parse(Int64, Base.ARGS[6])
            catch e
                @show e
                show_help()
            end
        else
            s = state_history[N_h + 1]
            N_AI = length(reward_history) - N_h
        end

        AI_state_history, AI_reward_history = automate(true_user_model, user_models, posterior_history[N_h + 1,:], problem_spec, s, N_AI)
        if restart
            state_history = AI_state_history
            reward_history = AI_reward_history
        else
            state_history = vcat(state_history[1:N_h+1], AI_state_history)
            reward_history = vcat(reward_history[1:N_h], AI_reward_history)
        end
        @save Base.ARGS[1] experiment_spec_name datafile_name experiment_index state_history reward_history N_h N_AI
    else
        @warn "incorrect number of arguments given"
        show_help()
    end
end

main()
