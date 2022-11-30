# This file implements the partial automation baseline for the inventory management experiment. Some output files call this "automation only" because this is technically an assistant that can only automate, and can't advise.
#

using Pkg
Pkg.activate("./..")

include("LibInventoryManager.jl")

include("../LibRootSamplingMCTS.jl")
using .RootSamplingMCTS

using LRUCache
using Distributions
using Random
using POMDPs
using MCTS
using JLD2


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


mutable struct InventoryPlanningAutomationMDP <: MDP{InventoryState{Int64}, Union{ProduceProductsAction,Nothing}}
    user_model::BoltzmannUserModel
    world_model::InventoryPlanningMDP

    InventoryPlanningAutomationMDP(um::BoltzmannUserModel) = new(um, InventoryPlanningMDP(um.world_model))
end

POMDPs.discount(m::InventoryPlanningAutomationMDP) = m.world_model.discounting
POMDPs.initialstate(m::InventoryPlanningAutomationMDP) = initialstate(m.world_model)

function POMDPs.gen(m::InventoryPlanningAutomationMDP, s::InventoryState{Int64}, a::ProduceProductsAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    return gen(m.world_model, s, a, rng)
end

function POMDPs.gen(m::InventoryPlanningAutomationMDP, s::InventoryState{Int64}, ::Nothing, rng::AbstractRNG = Random.GLOBAL_RNG)
    a_sup = act(m.user_model, s, missing)
    sp, r = @gen(:sp, :r)(m.world_model, s, a_sup, rng)
    return (sp=sp, r=r, info = Dict("user action" => a_sup))
end

function POMDPs.actions(m::InventoryPlanningAutomationMDP, s::InventoryState{Int64})
    ret = Union{ProduceProductsAction,Nothing}[a for a in actions(m.world_model, s)]
    push!(ret, nothing)
    return ret
end


mutable struct RDSAMDP <: RandomMDP{InventoryState{Int64}, Union{ProduceProductsAction,Nothing}}
    underlying_MDP::InventoryPlanningAutomationMDP
    underlying_MDP_idx::Int64

    problem_spec::ProblemSpecification
    user_models::Array{UserSpecification,1}
    weights::Array{Float64,1} # unnormalized log-probabilities for elements of user_models
    
    subsample_size::Int64
    subsample_idxs::Array{Int64,1} # this will be used to weight down any samples not in the subsample

    discounting::Float64

    qvalue_cache::LRU{Tuple{Int64,BigInt},Array{Float64,1}}

    function RDSAMDP(user_models::Array{UserSpecification,1}, weights::Array{Float64,1}, problem_spec::ProblemSpecification, subsample_size::Union{Int64,Missing} = missing; cache_maxsize = 100000)
        subsample_size = ismissing(subsample_size) ? length(user_models) : subsample_size
        @assert size(user_models) == size(weights)
        @assert subsample_size <= length(user_models)
        idx = argmax(weights .+ rand(Gumbel(), length(user_models)))
        ret = new(InventoryPlanningAutomationMDP(BoltzmannUserModel(problem_spec,user_models[idx])), idx, problem_spec, user_models, weights, subsample_size, zeros(Float64, length(user_models)), problem_spec.discounting, LRU{Tuple{Int64,BigInt},Array{Float64,1}}(maxsize = cache_maxsize))
        ret.underlying_MDP = InventoryPlanningAutomationMDP(BoltzmannUserModel(problem_spec,user_models[idx], ret))

        # set subsample and resample underlying MDP
        resample_subsample!(ret)
        rand!(ret)
        
        return ret
    end
end

RDSAMDP(user_models::Array{UserSpecification,1}, problem_spec::ProblemSpecification, subsample_size::Union{Int64,Missing} = missing; cache_maxsize = 100000) = RDSAMDP(user_models, zeros(length(user_models)), problem_spec, subsample_size; cache_maxsize = cache_maxsize)

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
    m.underlying_MDP = InventoryPlanningAutomationMDP(BoltzmannUserModel(m.problem_spec, m.user_models[idx], m))
    m.underlying_MDP_idx = idx
    return m
end

function update_weights!(m::RDSAMDP, weights::Array{Float64,1})
    m.weights = weights
    return m
end

function run_rootsampled_planning(true_user_model::UserSpecification, user_models::Array{UserSpecification,1}, problem_spec::ProblemSpecification; estimate_value = 0.0, N::Int64 = 10, n_iterations::Int64 = 1000, planning_depth::Int64 = 10, planning_subset_size::Int64 = 100, max_time::Float64 = Inf)
    N_um = length(user_models)

     # This is the assistance problem with the true user model
    true_assMDP = InventoryPlanningAutomationMDP(BoltzmannUserModel(problem_spec, true_user_model))

    RassMDP = RDSAMDP(user_models,
                      problem_spec,
                      planning_subset_size;
                      cache_maxsize = 1_000_000)
    solver = DPWSolver(n_iterations=n_iterations,
                       depth=planning_depth,
                       exploration_constant=10.0,
                       estimate_value = estimate_value,
                       check_repeat_state = false,
                       enable_action_pw = false,
                       show_progress = false,
                       max_time = max_time)
    planner = RootSamplingDPWPlanner(solver, RassMDP)
    
    s = initialstate(true_assMDP)
    
    state_history = [s]
    reward_history = zeros(N)
    posterior_history = zeros(N+1, N_um)
    posterior_history[1,:] .= RassMDP.weights
    aH_history = []
    aAI_history = []
    
    for i in 1:N
        println("=============== $(i) ===============")
        resample_subsample!(planner.mdp)
        @time aAI = action(planner, s)

        sp, r, info = gen(true_assMDP, s, aAI)
        aH = "user action" in keys(info) ? info["user action"] : missing
        
        # update beliefs if user was involved
        new_weights = RassMDP.weights
        if !ismissing(aAI) && isa(aAI, ProduceProductsRecommendation)
            @time new_weights .+= map(θ -> log(likelihood(BoltzmannUserModel(problem_spec, θ), s, aH, aAI)), RassMDP.user_models)
            update_weights!(RassMDP, new_weights);
        end
        
        reward_history[i] = r
        posterior_history[i+1,:] .= new_weights
        push!(aAI_history, aAI)
        push!(aH_history, aH)

        println("AI: ", string(aAI), " H: ", ismissing(aH) ? "" : string(aH), " -- R = ", discounted_reward(reward_history[1:i], problem_spec.discounting), " -- ", entropy(exp.(new_weights) ./ sum(exp.(new_weights))))

        # state change
        s = sp
        push!(state_history, s)
    end
    
    return state_history, reward_history, aAI_history, aH_history, posterior_history
end

function main()
    N = 50
    n_iterations = 500_000
    planning_depth = 4
    planning_subset_size = 200
    max_time = 15*60.0 # max time for planning per iteration, in seconds

    if length(Base.ARGS) in [3,4]
        @load Base.ARGS[2] problem_spec user_models true_user_model
        experiment_spec_name = Base.ARGS[2]
        experiment_index = parse(Int64, Base.ARGS[3])
        user_models = user_models[experiment_index]
        true_user_model = true_user_model[experiment_index]
        problem_spec = problem_spec[experiment_index]
        planning_subset_size = planning_subset_size > length(user_models) ? length(user_models) : planning_subset_size

        state_history, reward_history, aAI_history, aH_history, posterior_history = run_rootsampled_planning(true_user_model,
                                                                                                             user_models,
                                                                                                             problem_spec,
                                                                                                             estimate_value = 0.0,
                                                                                                             N = N,
                                                                                                             n_iterations = n_iterations,
                                                                                                             planning_depth = planning_depth,
                                                                                                             planning_subset_size = planning_subset_size,
                                                                                                             max_time = max_time)
        @save Base.ARGS[1] experiment_spec_name experiment_index state_history reward_history aAI_history aH_history posterior_history n_iterations planning_depth planning_subset_size
    else
        println("Main AI-assisted design experiment")
        println("Usage:    julia AIADExperiment.jl OUTFILE SPECFILE SPEC_IDX")
        println("    OUTFILE       Specifies the name of the output file. All the results will be written to this file.")
        println("    SPECFILE      Specifies the name of the experiment specification file to be used.")
        println("    SPEC_IDX      Specifies the index of the experiment scenario within SPECFILE to run.")
        exit(1)
    end
end

main()
