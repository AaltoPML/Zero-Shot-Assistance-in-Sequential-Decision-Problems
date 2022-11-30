# This file implements the automation part of the IRL + automation and PL + automation baseline. You can start automating from any point in the trajectories collected by ElicitationExperiment.jl
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
using Random

import Base.hash
import POMDPs.stateindex

function entropy(x)
    nonzero = x .> 0.0
    return -sum(log2.(x[nonzero]) .* x[nonzero])
end


mutable struct RMDSMDP{UMP <: Pair{WM,UM} where WM <:WorldModelSpecification where UM <:MCTSUserModelSpecification} <: MDP{Daytrip,DaytripEdit}
    underlying_MDP::MachineDaytripScheduleMDP
    underlying_MDP_idx::Int64

    user_models::Array{UMP,1}
    weights::Array{Float64,1} # unnormalized log-probabilities for elements of user_models

    home::POI
    POIs::Array{POI,1}

    discounting::Float64
    optimizer::Symbol

    function RMDSMDP(user_models::Array{Pair{WM,UM},1}, weights::Array{Float64,1}, home::POI, POIs::Array{POI,1}, discounting::Float64) where WM <:WorldModelSpecification where UM <:MCTSUserModelSpecification
        @assert size(user_models) == size(weights)
        idx = argmax(weights .+ rand(Gumbel(), length(user_models)))
        ret = new{Pair{WM,UM}}(MachineDaytripScheduleMDP(home, POIs, user_models[idx].first, discounting, optimizer = :CW), idx, user_models, weights, home, POIs, discounting, :CW)
        return ret
    end
end

RMDSMDP(user_models::Array{Pair{WM,UM},1}, home::POI, POIs::Array{POI,1}, discounting::Float64) where WM <:WorldModelSpecification where UM <:MCTSUserModelSpecification = RMDSMDP(user_models, zeros(length(user_models)), home, POIs, discounting)

POMDPs.initialstate(m::RMDSMDP) = POMDPs.initialstate(m.underlying_MDP)
POMDPs.actions(m::RMDSMDP, s::Daytrip) = POMDPs.actions(m.underlying_MDP, s)
POMDPs.actions(m::RMDSMDP) = POMDPs.actions(m.underlying_MDP)
POMDPs.discount(m::RMDSMDP) = POMDPs.discount(m.underlying_MDP)
POMDPs.stateindex(m::RMDSMDP, s::Daytrip)::BigInt = POMDPs.stateindex(m.underlying_MDP, s)

function POMDPs.gen(m::RMDSMDP, s::Daytrip, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG)
    idx = argmax(m.weights .+ rand(Gumbel(), length(m.user_models)))
    m.underlying_MDP = MachineDaytripScheduleMDP(m.home, m.POIs, m.user_models[idx].first, m.discounting, optimizer = m.optimizer)
    m.underlying_MDP_idx = idx

    return POMDPs.gen(m.underlying_MDP, s, a, rng)
end

function find_optimal_trip(home, POIs, um, user_models, log_weights, N, starting_state = missing; value_estimator = 0.0)
    mdp = RMDSMDP(user_models, log_weights, home, POIs, 0.99)
    solver = DPWSolver(n_iterations=1_000_000, 
                       depth=3, 
                       exploration_constant=0.1, 
                       estimate_value=value_estimator, 
                       check_repeat_state = true, 
                       enable_action_pw = false, 
                       show_progress = false, 
                       tree_in_info=false)
    true_objective_mdp = MachineDaytripScheduleMDP(um.world_model, 0.99)
    planner = DPWPlanner(solver, mdp);

    hash(s::Daytrip) = UInt(POMDPs.stateindex(mdp, s))

    s = ismissing(starting_state) ? initialstate(mdp).val : starting_state
    if typeof(s) == DaytripAndInfo
        s = s.trip
    end

    state_history = [s]
    utility_history = [utility(true_objective_mdp, s)]
    println("========= 0 (AI) ==========")
    println("starting utility: " * string(utility(true_objective_mdp, s)))
    a_prev = missing
    for i in 1:N
        println("========= $i (AI) ==========")
        a = actions(mdp, s)[1]
        if (i >= 2) && (state_history[end] == state_history[end-1])
            # default to NOOP action once planner can't find ways of improving the design anymore
        elseif (i >= 4) && (state_history[end] == state_history[end-2]) && (state_history[end-1] == state_history[end-3])
            # default to NOOP action if planner starts to cycle through the same two states
        else
            @time a = action(planner, s)
            @show length(planner.tree.s_labels)
        end
        s = @gen(:sp)(mdp, s, a)

        @show a
        @show duration(s), utility(true_objective_mdp, s)

        push!(state_history, s)
        push!(utility_history, utility(true_objective_mdp, s))
	a_prev = a
    end
    
    return state_history, utility_history
end

################################################################################
#                            EXPERIMENT RUN CODE                               #
################################################################################

function show_help()
    println("Optimizes a daytrip starting from any point in a prior AIAD or Elicitation Experiment.")
    println("Usage:    julia ElicitationExperiment.jl OUTFILE SPECFILE DATAFILE STARTPOINT CONTINUE")
    println("    OUTFILE         Specifies the name of the output file. All the results will be written to this file.")
    println("    SPECFILE        Specifies the name of the experiment specification file to be used.")
    println("    DATAFILE        Specifies the name of of the datafile produced by either AIADExperiment or ElicitationExperiment to be used for weights and starting state.")
    println("    STARTPOINT      Integer. Specifies after how many interactions in DATAFILE the optimizer should start.")
    println("    CONTINUE        \"CONTINUE\" or \"RESTART\". Specifies whether the optimizer should continue from the state recorded in DATAFILE at STARTPOIT or should restart.")
    exit(1)
end

function main()
    N_AI = 40

    if length(Base.ARGS) == 5
        @load Base.ARGS[3] state_history posterior_history experiment_index experiment_spec_name
        @load Base.ARGS[2] POIs home user_models true_user_model
        user_models = user_models[experiment_index]
        POIs = POIs[experiment_index]
        home = home[experiment_index]
        true_user_model = instantiate(true_user_model[experiment_index], home, POIs)
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

        s = missing
        if Base.ARGS[5] == "CONTINUE"
            s = state_history[N_h + 1]
        elseif Base.ARGS[5] == "RESTART"
            s = missing
        else
            @warn "unrecognized value: $Base.ARGS[4]"
            show_help()
        end

        state_history, utility_history = find_optimal_trip(home, POIs, true_user_model, user_models, posterior_history[N_h+1,:], N_AI, s, value_estimator = 0.0)
        @save Base.ARGS[1] experiment_spec_name datafile_name experiment_index state_history utility_history N_h N_AI
    else
        @warn "incorrect number of arguements given"
        show_help()
    end
end

main()
