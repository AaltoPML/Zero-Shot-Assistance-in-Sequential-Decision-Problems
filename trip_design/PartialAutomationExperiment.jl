# This file implements the partial automation baseline for the day trip design experiment. Some output files call this "automation only" because this is technically an assistant that can only automate, and can't advise.
#
#

using Pkg
Pkg.activate("../.")

include("LibTravelPlanner.jl")
using .DaytripDesignAssistance

include("../LibRootSamplingMCTS.jl")
using .RootSamplingMCTS
import .DaytripDesignAssistance.utility

using LRUCache
using Distributions
using Random
using POMDPs
using MCTS
using JLD2
using POMDPModelTools


function entropy(x)
    nonzero = x .> 0.0
    return -sum(log2.(x[nonzero]) .* x[nonzero])
end


mutable struct DaytripScheduleAutomationMDP <: MDP{DaytripAndInfo, Union{DaytripEdit,Nothing}}
    user_model::MCTSUserModel
    objective_world_model::MachineDaytripScheduleMDP

    # these are technically already present in user_model but they're stored here too
    POIs::Array{POI,1}
    home::POI

    discounting::Float64

    optimizer::Symbol

    DaytripScheduleAutomationMDP(m::MCTSUserModel, discounting::Float64; optimizer = :CW) = new(m, MachineDaytripScheduleMDP(m.world_model, discounting, optimizer = optimizer), m.world_model.POIs, m.world_model.home, discounting, optimizer)
end

POMDPs.discount(m::DaytripScheduleAutomationMDP) = m.discounting
POMDPs.initialstate(m::DaytripScheduleAutomationMDP) = Deterministic(DaytripAndInfo([m.home]))
POMDPs.isterminal(::DaytripScheduleAutomationMDP, ::DaytripAndInfo) = false

utility(m::DaytripScheduleAutomationMDP, s::DaytripAndInfo) = utility(m.objective_world_model, s.trip)

"""
Helper function for gen. Implements the daytrip editing logic.
"""
apply_trip_edit(m::DaytripScheduleAutomationMDP, s::Daytrip, a::DaytripEdit) = @gen(:sp)(m.objective_world_model, s, a)

function POMDPs.gen(m::DaytripScheduleAutomationMDP, s::DaytripAndInfo, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false)
    sp = DaytripAndInfo(apply_trip_edit(m, s.trip, a))
    return (sp=sp, r=POMDPs.reward(m, s, a, sp), info=missing)
end

function POMDPs.gen(m::DaytripScheduleAutomationMDP, s::DaytripAndInfo, a::Nothing, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false)
    a_sup = act(m.user_model, s, missing; verbose = verbose)
    sp = DaytripAndInfo(apply_trip_edit(m, s.trip, a_sup), a_sup)
    return (sp=sp, r=POMDPs.reward(m, s, a_sup, sp), info=missing)
end

POMDPs.reward(m::DaytripScheduleAutomationMDP, s::DaytripAndInfo, ::Any, sp::DaytripAndInfo) = POMDPs.discount(m) * utility(m, sp) - utility(m, s)

function POMDPs.actions(m::DaytripScheduleAutomationMDP, s::DaytripAndInfo)
    ret = Array{Union{DaytripEdit,Nothing},1}()
    push!(ret, nothing)
    push!(ret, actions(m.objective_world_model, s.trip)...)
    return ret
end

"""
WARNING action indices do not necessarily match with indices within the list returned by action()
"""
POMDPs.actionindex(m::DaytripScheduleAutomationMDP, a::EditRecommendation)::Int64 = POMDPs.actionindex(m.objective_world_model, a.edit)
POMDPs.actionindex(m::DaytripScheduleAutomationMDP, a::DaytripEdit)::Int64 = POMDPs.actionindex(m.objective_world_model, a.edit) + length(m.POIs)



mutable struct RDSAMDP{UMP <: Pair{WM,UM} where WM <: WorldModelSpecification where UM <: MCTSUserModelSpecification} <: RandomMDP{DaytripAndInfo, Union{DaytripEdit,Nothing}}
    underlying_MDP::DaytripScheduleAutomationMDP
    underlying_MDP_idx::Int64

    user_models::Array{UMP,1}
    weights::Array{Float64,1} # unnormalized log-probabilities for elements of user_models
    
    subsample_size::Int64
    subsample_idxs::Array{Int64,1} # this will be used to weight down any samples not in the subsample

    home::POI
    POIs::Array{POI,1}

    discounting::Float64
    optimizer::Symbol

    qvalue_cache::LRU{Tuple{Int64,BigInt},Array{Float64,1}}

    function RDSAMDP(user_models::Array{Pair{WM,UM},1}, weights::Array{Float64,1}, home::POI, POIs::Array{POI,1}, discounting::Float64, subsample_size::Union{Int64,Missing} = missing; cache_maxsize = 100000, enable_direct_edits = false) where WM <: WorldModelSpecification where UM <:MCTSUserModelSpecification
        subsample_size = ismissing(subsample_size) ? length(user_models) : subsample_size
        @assert size(user_models) == size(weights)
        @assert subsample_size <= length(user_models)
        idx = argmax(weights .+ rand(Gumbel(), length(user_models)))
        ret = new{Pair{WM,UM}}(DaytripScheduleAutomationMDP(instantiate(user_models[idx], home, POIs), discounting), idx, user_models, weights, subsample_size, zeros(Float64, length(user_models)), home, POIs, discounting, :CW, LRU{Tuple{Int64,BigInt},Array{Float64,1}}(maxsize = cache_maxsize))
        ret.underlying_MDP = DaytripScheduleAutomationMDP(instantiate(user_models[idx], home, POIs, ret), discounting)
        
        # set subsample and resample underlying MDP
        resample_subsample!(ret)
        rand!(ret)
        
        return ret
    end
end

RDSAMDP(user_models::Array{Pair{WM,UM},1}, home::POI, POIs::Array{POI,1}, discounting::Float64, subsample_size::Union{Int64,Missing} = missing; cache_maxsize = 100000, enable_direct_edits = false) where WM <: WorldModelSpecification where UM <: MCTSUserModelSpecification = RDSAMDP(user_models, zeros(length(user_models)), home, POIs, discounting, subsample_size; cache_maxsize = cache_maxsize, enable_direct_edits = enable_direct_edits)

function resample_subsample!(m::RDSAMDP)
    m.subsample_idxs = sortperm(m.weights .+ rand(Gumbel(), length(m.weights)))[end-m.subsample_size+1:end]
    return m
end

import Base: setindex!, get
function setindex!(m::RDSAMDP, v::Array{Float64,1}, k::BigInt)
    m.qvalue_cache[(m.underlying_MDP_idx,k)] = v
end
get(m::RDSAMDP, k::BigInt, default) = get(m.qvalue_cache, (m.underlying_MDP_idx,k), default)

function Random.rand!(m::RDSAMDP)
    subidx = argmax(m.weights[m.subsample_idxs] .+ rand(Gumbel(), m.subsample_size))
    idx = m.subsample_idxs[subidx]
    m.underlying_MDP = DaytripScheduleAutomationMDP(instantiate(m.user_models[idx], m.home, m.POIs, m), m.discounting)
    m.underlying_MDP_idx = idx
    return m
end

function update_weights!(m::RDSAMDP, weights::Array{Float64,1})
    m.weights = weights
    return m
end

function run_rootsampled_planning(um, user_models; estimate_value = 0.0, N = 10, n_iterations = 1000, planning_depth = 10, planning_subset_size = 100, max_time::Float64 = Inf)
    N_um = length(user_models)

     # This is the assistance problem with the true user model
    true_assMDP = DaytripScheduleAutomationMDP(um, 1.0)

    RassMDP = RDSAMDP(user_models,
                      um.world_model.home,
                      um.world_model.POIs,
                      0.95,
                      planning_subset_size;
                      cache_maxsize = 1_000_000)
    solver = DPWSolver(n_iterations=n_iterations,
                       depth=planning_depth,
                       exploration_constant=0.1,
                       estimate_value = estimate_value,
                       check_repeat_state = false,
                       enable_action_pw = false,
                       show_progress = false,
                       max_time = max_time)
    planner = RootSamplingDPWPlanner(solver, RassMDP)
    
    s = initialstate(true_assMDP).val
    
    state_history = [s]
    utility_history = vcat(utility(true_assMDP, s), zeros(N))
    posterior_history = zeros(N+1, N_um)
    posterior_history[1,:] .= RassMDP.weights
    aH_history = []
    aAI_history = []
    
    println("starting utility: ", utility(true_assMDP, s))
    last_planning_point = 0
    aH = missing
    aAI = missing
    for i in 1:N
        println("=============== $(i) ===============")
        δ = i == 1 ? Inf : sum(abs.((exp.(posterior_history[i,:]) ./ sum(exp.(posterior_history[i,:]))) .- (exp.(posterior_history[last_planning_point,:]) ./ sum(exp.(posterior_history[last_planning_point,:])))))
        @show δ
        if δ < 0.01 && state_history[end] == state_history[end-1]
            println("Not enough change in posterior ($δ), not re-planning...")
         else
            resample_subsample!(planner.mdp)
            @time aAI = action(planner, s)
            last_planning_point = i
        end
        sp, r, _ = gen(true_assMDP, s, aAI, verbose = true)
        aH = sp.UserEdit
        
        # update beliefs if user was involved
        new_weights = RassMDP.weights
        if typeof(aAI) == Nothing
            nonzero = new_weights .!= -Inf
            @time new_weights[nonzero] .+= map(θ -> log(likelihood(instantiate(θ, um.world_model.home, um.world_model.POIs), s, missing, sp)), RassMDP.user_models[nonzero])
            update_weights!(RassMDP, new_weights);
        end

        println("AI: ", string(aAI), " H: ", string(aH), " -- ", utility(true_assMDP, sp), " -- ", duration(sp), " -- ", entropy(exp.(new_weights) ./ sum(exp.(new_weights))))
        
        utility_history[i+1] = utility(true_assMDP, sp)
        posterior_history[i+1,:] .= new_weights
        push!(aAI_history, aAI)
        push!(aH_history, aH)

        # state change
        s = sp
        push!(state_history, s)
    end
    
    return state_history, utility_history, aAI_history, aH_history, posterior_history
end

function main()
    N = 30
    n_iterations = 750_000
    planning_depth = 2
    planning_subset_size = 100
    max_time = 30*60.0 # max time for planning per iteration, in seconds

    if length(Base.ARGS) in [3]
        @load Base.ARGS[2] user_models true_user_model POIs home
        experiment_spec_name = Base.ARGS[2]
        experiment_index = parse(Int64, Base.ARGS[3])
        user_models = user_models[experiment_index]
        @show N
        POIs = POIs[experiment_index]
        home = home[experiment_index]
        true_user_model = instantiate(true_user_model[experiment_index], home, POIs)
        planning_subset_size = planning_subset_size > length(user_models) ? length(user_models) : planning_subset_size

        state_history, utility_history, aAI_history, aH_history, posterior_history = run_rootsampled_planning(true_user_model,
                                                                                                              user_models,
                                                                                                              estimate_value = 0.0,
                                                                                                              N = N,
                                                                                                              n_iterations = n_iterations,
                                                                                                              planning_depth = planning_depth,
                                                                                                              planning_subset_size = planning_subset_size,
                                                                                                              max_time = max_time)
        @save Base.ARGS[1] experiment_spec_name experiment_index state_history utility_history aAI_history aH_history posterior_history n_iterations planning_depth planning_subset_size enable_direct_edits
    else
        println("This implements the partial automation expriment.")
        println("Usage:    julia PartialAutomationExperiment.jl OUTFILE SPECFILE SPEC_IDX")
        println("    OUTFILE       Specifies the name of the output file. All the results will be written to this file.")
        println("    SPECFILE      Specifies the name of the experiment specification file to be used.")
        println("    SPEC_IDX      Specifies the index of the experiment scenario within SPECFILE to run.")
        exit(1)
    end
end

main()
