# This library implements the common parts of the inventory management experiment. You will find here the basic problem implementation, the agent model implementation and the implementation of AIAD.

include("../LibBFSSolver.jl")

using Distributions
using Random
using POMDPs
using POMDPModelTools: SparseCat

import Base: ==, string
import Distributions: std

export DemandPrediction, InventoryState, ProduceProductsAction, ProduceProductsRecommendation, ProblemSpecification, UserSpecification
export InventoryPlanningMDP, InventoryHeuristicPlanningMDP, InventoryPlanningAssistanceMDP, expectedReward
export BoltzmannUserModel, act, likelihood, get_policy

const EPISODE_LIMIT = 10_000
const INVENTORY_LIMIT = 1_000


DemandPrediction = SparseCat{Vector{Int64},Vector{Float64}}
std(d::SparseCat{Vector{K},Vector{V}}) where {K,V} = sqrt(sum(d.probs .* ((d.vals .- mean(d)) .^ 2)))


struct InventoryState{T}
    product_quantities::Array{T,1}
    timestep::Int64

    InventoryState{T}(product_quantities::Array{T,1}, timestep::Int64) where T = new{T}(product_quantities, timestep)
    InventoryState{Float64}(s::InventoryState{Int64}) = new{Float64}(convert(Array{Float64,1}, s.product_quantities), s.timestep)
end

==(x::InventoryState{T}, y::InventoryState{T}) where T = (x.timestep == y.timestep) && (x.product_quantities == y.product_quantities)


struct ProduceProductsAction
    production_amount::Array{Int64,1}
end

==(x::ProduceProductsAction, y::ProduceProductsAction) = x.production_amount == y.production_amount

string(a::ProduceProductsAction) = return join(["P$(i)=$(q)" for (i,q) in enumerate(a.production_amount)], ", ")


function generate_action_space(N_PRODUCTS, production_capacity)
    ret = ProduceProductsAction[]
    for i in 0:2:production_capacity
        push!(ret, ProduceProductsAction([i]))
    end

    for _ in 2:N_PRODUCTS
        temp_ret = ProduceProductsAction[]
        for a in ret
            for i in 0:2:(production_capacity - sum(a.production_amount))
                push!(temp_ret, ProduceProductsAction(vcat(a.production_amount, i)))
            end
        end
        ret = temp_ret
    end
    return ret
end


"""
Defines all user-independent aspects of an Inventory Planning Problem
"""
struct ProblemSpecification
    demand_predictions::Array{DemandPrediction,2}
    production_capacity::Int64
    discounting::Float64
end


"""
Defines all user-specific aspects of an Inventory Planning Problem
"""
struct UserSpecification
    product_profit::Array{Float64,1}
    inventory_cost::Array{Float64,1}
    lost_business_cost::Array{Float64,1}

    optimism::Float64

    # planning parameters
    multi_choice_optimality::Float64
    comparison_optimality::Float64
    planning_depth::Int64
    planning_iterations::Int64

    function UserSpecification(product_profit::Array{Float64,1}, inventory_cost::Array{Float64,1}, lost_business_cost::Array{Float64,1}, optimism::Float64 = 0.0, multi_choice_optimality::Float64 = 1.0, comparison_optimality::Float64 = 10.0, planning_depth::Int64 = 3, planning_iterations::Int64 = 500)
        return new(product_profit, inventory_cost, lost_business_cost, optimism, multi_choice_optimality, comparison_optimality, planning_depth, planning_iterations)
    end

    function UserSpecification(product_profit::Array{Float64,1}, inventory_cost::Float64, lost_business_cost::Float64, optimism::Float64 = 0.0, multi_choice_optimality::Float64 = 1.0, comparison_optimality::Float64 = 10.0, planning_depth::Int64 = 3, planning_iterations::Int64 = 500)
        return UserSpecification(product_profit, ones(Float64, length(product_profit)) .* inventory_cost, ones(Float64, length(product_profit)) .* lost_business_cost, optimism, multi_choice_optimality, comparison_optimality, planning_depth, planning_iterations)
    end

    function UserSpecification(product_profit::Array{Float64,1}, inventory_cost::Float64, lost_business_cost::Float64, multi_choice_optimality::Float64 = 1.0, comparison_optimality::Float64 = 10.0, planning_depth::Int64 = 3, planning_iterations::Int64 = 500)
        return UserSpecification(product_profit, ones(Float64, length(product_profit)) .* inventory_cost, ones(Float64, length(product_profit)) .* lost_business_cost, 0.0, multi_choice_optimality, comparison_optimality, planning_depth, planning_iterations)
    end
end


mutable struct InventoryPlanningMDP <: MDP{InventoryState{Int64},ProduceProductsAction}
    # future demand predictions in a HORIZON X N_PRODUCTS matrix
    demand_predictions::Array{DemandPrediction,2}
    production_capacity::Int64

    N_PRODUCTS::Int64
    
    # utility definition
    product_profit::Array{Float64,1}
    inventory_cost::Array{Float64,1}
    lost_business_cost::Array{Float64,1}
    discounting::Float64

    # action accounting
    action_space::Array{ProduceProductsAction,1}

    function InventoryPlanningMDP(demand_predictions::Array{DemandPrediction,2}, production_capacity::Int64, product_profit::Array{Float64,1}, inventory_cost::Array{Float64,1}, lost_business_cost::Array{Float64,1}, discounting::Float64 = 0.99)
        @assert size(demand_predictions,2) == length(product_profit)
        @assert all(product_profit .>= 0.0)
        @assert all(inventory_cost .>= 0.0)
        @assert all(lost_business_cost .>= 0.0)

        action_space = generate_action_space(size(demand_predictions,2), production_capacity)

        new(demand_predictions,
            production_capacity,
            size(demand_predictions,2),
            product_profit,
            inventory_cost,
            lost_business_cost,
            discounting,
            action_space)
    end

    function InventoryPlanningMDP(demand_predictions::Array{DemandPrediction,2}, production_capacity::Int64, product_profit::Array{Float64,1}, inventory_cost::Float64, lost_business_cost::Float64, discounting::Float64 = 0.99)
        InventoryPlanningMDP(demand_predictions,
                             production_capacity,
                             size(demand_predictions,2),
                             product_profit,
                             ones(Float64, size(demand_predictions,2)) .* inventory_cost,
                             ones(Float64, size(demand_predictions,2)) .* lost_business_cost,
                             discounting)
    end

    function InventoryPlanningMDP(ps::ProblemSpecification, us::UserSpecification)
        InventoryPlanningMDP(ps.demand_predictions,
                             ps.production_capacity,
                             us.product_profit,
                             us.inventory_cost,
                             us.lost_business_cost,
                             ps.discounting)
    end
end

POMDPs.initialstate(m::InventoryPlanningMDP) = InventoryState{Int64}(zeros(Int64, m.N_PRODUCTS), 1)

function POMDPs.gen(m::InventoryPlanningMDP, s::InventoryState{Int64}, a::ProduceProductsAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    if s.timestep > size(m.demand_predictions,1) # episode has ended, continue gracefully
        return (sp = s, r = 0.0, info = Dict("error" => "stepped beyond episode length"))
    end

    new_product_quantities = Array{Int64,1}()
    r = 0.0
    for (D, Ii, P, g, c, l) in zip(m.demand_predictions[s.timestep,:], s.product_quantities, a.production_amount, m.product_profit, m.inventory_cost, m.lost_business_cost)
        Di = rand(D)
        newI = max(Ii + P - Di, 0)
        r += min(Ii + P, Di) * g - newI * c - min(Ii + P - Di, 0) * l
        push!(new_product_quantities, newI)
    end

    return (sp = InventoryState{Int64}(new_product_quantities, s.timestep + 1), r = r, info = Dict())
end

function expectedReward(m::InventoryPlanningMDP, s::InventoryState{Int64}, a::ProduceProductsAction)
    if s.timestep > size(m.demand_predictions,1) # episode has ended, continue gracefully
        return 0.0
    end

    r = 0.0
    for (D, I, P, g, c, l) in zip(m.demand_predictions[s.timestep,:], s.product_quantities, a.production_amount, m.product_profit, m.inventory_cost, m.lost_business_cost)
        for (Di, Dp) in D
            if Dp == 0.0
                continue
            end
            newI = max(I + P - Di, 0)
            r_case = min(I + P, Di) * g - newI * c - min(I + P - Di, 0) * l
            r += r_case * Dp
        end
    end

    return r
end


mutable struct InventoryHeuristicPlanningMDP <: MDP{InventoryState{Float64},ProduceProductsAction}
    # future demand predictions in a HORIZON X N_PRODUCTS matrix
    demand_predictions::Array{DemandPrediction,2}
    production_capacity::Int64

    N_PRODUCTS::Int64
    
    # utility definition
    product_profit::Array{Float64,1}
    inventory_cost::Array{Float64,1}
    lost_business_cost::Array{Float64,1}

    optimism::Float64

    discounting::Float64

    # action accounting
    action_space::Array{ProduceProductsAction,1}

    function InventoryHeuristicPlanningMDP(demand_predictions::Array{DemandPrediction,2}, production_capacity::Int64, product_profit::Array{Float64,1}, inventory_cost::Array{Float64,1}, lost_business_cost::Array{Float64,1}, optimism::Float64 = 0.0, discounting::Float64 = 0.99)
        @assert size(demand_predictions, 2) == length(product_profit)
        @assert size(demand_predictions, 1) < EPISODE_LIMIT "due to technical limits on state indexing, episode lengths are limited to $(EPISODE_LIMIT) steps"
        @assert all(product_profit .>= 0.0)
        @assert all(inventory_cost .>= 0.0)
        @assert all(lost_business_cost .>= 0.0)

        action_space = generate_action_space(size(demand_predictions,2), production_capacity)

        new(demand_predictions,
            production_capacity,
            size(demand_predictions,2),
            product_profit,
            inventory_cost,
            lost_business_cost,
            optimism,
            discounting,
            action_space)
    end

    function InventoryHeuristicPlanningMDP(demand_predictions::Array{DemandPrediction,2}, production_capacity::Int64, product_profit::Array{Float64,1}, inventory_cost::Float64, lost_business_cost::Float64, optimism::Float64 = 0.0, discounting::Float64 = 0.99)
        InventoryHeuristicPlanningMDP(demand_predictions,
                                      production_capacity,
                                      size(demand_predictions,2),
                                      product_profit,
                                      ones(Float64, size(demand_predictions,2)) .* inventory_cost,
                                      ones(Float64, size(demand_predictions,2)) .* lost_business_cost,
                                      optimism,
                                      discounting)
    end

    function InventoryHeuristicPlanningMDP(ps::ProblemSpecification, us::UserSpecification)
        InventoryHeuristicPlanningMDP(ps.demand_predictions,
                                      ps.production_capacity,
                                      us.product_profit,
                                      us.inventory_cost,
                                      us.lost_business_cost,
                                      us.optimism,
                                      ps.discounting)
    end
end

POMDPs.initialstate(m::InventoryHeuristicPlanningMDP) = InventoryState{Float64}(zeros(Float64, m.N_PRODUCTS), 1)

function POMDPs.gen(m::InventoryHeuristicPlanningMDP, s::InventoryState{Float64}, a::ProduceProductsAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    if s.timestep > size(m.demand_predictions,1) # episode has ended, continue gracefully
        return (sp = s, r = 0.0, info = Dict("error" => "stepped beyond episode length"))
    end

    new_product_quantities = Array{Float64,1}()
    r = 0.0
    for (D, Ii, P, g, c, l) in zip(m.demand_predictions[s.timestep,:], s.product_quantities, a.production_amount, m.product_profit, m.inventory_cost, m.lost_business_cost)
        Di = max(mean(D) + m.optimism * std(D), 0.0)
        newI = max(Ii + Float64(P) - Di, 0)
        if newI >= INVENTORY_LIMIT
            # inventory is artificially limited to INVENTORY_LIMIT. This limit should never be reached given normal values for demand and inventory cost.
            newI = INVENTORY_LIMIT - 1
            @warn "inventory has exceeded $(INVENTORY_LIMIT) and has been truncated"
        end
        r += min(Ii + Float64(P), Di) * g - newI * c - min(Ii + Float64(P) - Di, 0) * l
        push!(new_product_quantities, newI)
    end

    return (sp = InventoryState{Float64}(new_product_quantities, s.timestep + 1), r = r, info = Dict())
end


function POMDPs.actions(m::Union{InventoryPlanningMDP,InventoryHeuristicPlanningMDP})
    return m.action_space
end

function POMDPs.actions(m::Union{InventoryPlanningMDP,InventoryHeuristicPlanningMDP}, s::InventoryState{T}) where T
    if s.timestep > size(m.demand_predictions, 1)
        return ProduceProductsAction[ProduceProductsAction(zeros(Int64, m.N_PRODUCTS))]
    end
    return POMDPs.actions(m)
end

function POMDPs.actionindex(m::Union{InventoryPlanningMDP,InventoryHeuristicPlanningMDP}, a::ProduceProductsAction)::Int64
    idx = findfirst(isequal(a), m.action_space)
    if ismissing(idx)
        @error "unknown action encoutered in actionindex"
    end
    return idx
end

function POMDPs.stateindex(::Type{InventoryPlanningMDP}, s::InventoryState{Int64})::BigInt
    IDX = BigInt(s.timestep)
    for (i,Ii) in enumerate(s.product_quantities)
        IDX += BigInt(Ii) * EPISODE_LIMIT * BigInt(INVENTORY_LIMIT ^ i)
    end
    return IDX
end

POMDPs.stateindex(::InventoryPlanningMDP, s::InventoryState{Int64})::BigInt = stateindex(InventoryPlanningMDP, s)

POMDPs.discount(m::Union{InventoryPlanningMDP,InventoryHeuristicPlanningMDP}) = m.discounting


InventoryPlanningMDP(m::InventoryHeuristicPlanningMDP) = InventoryPlanningMDP(m.demand_predictions, m.production_capacity, m.product_profit, m.inventory_cost, m.lost_business_cost, m.discounting)

#--------------------------------------------------------------------------------
# USER MODEL AND INTERACTION
#--------------------------------------------------------------------------------

struct ProduceProductsRecommendation
    val::ProduceProductsAction
end

string(a::ProduceProductsRecommendation) = return "try " * string(a.val)

"""
User model built on Boltzmann rational reasoning

cache is a cache to hold q_value calculations for re-use. It should map a state index (Int) to a vector (Array{Float64,1}) good options are LRU{Int,Array{Float64,1}} and Dict{Int,Array{Float64,1}}
"""
struct BoltzmannUserModel{CacheType}
    world_model::InventoryHeuristicPlanningMDP
    value_estimator::BFSSolver
    multi_choice_optimality::Float64 # optimality when choosing action
    comparison_optimality::Float64 # optimality when comparing actions
    cache::CacheType

    function BoltzmannUserModel{CacheType}(m::InventoryHeuristicPlanningMDP,
                                                          planning_depth::Int64,
                                                          multi_choice_optimality::Float64,
                                                          comparison_optimality::Float64,
                                                          cache::CacheType;
                                                          planning_iterations::Int64 = 500) where CacheType
        @assert multi_choice_optimality >= 0.0
        @assert comparison_optimality >= 0.0

        return new(m,
                   BFSSolver(m, planning_depth, planning_iterations),
                   multi_choice_optimality,
                   comparison_optimality,
                   cache)
    end

    function BoltzmannUserModel(ps::ProblemSpecification, us::UserSpecification, cache::CacheType = nothing) where CacheType
        @assert us.multi_choice_optimality >= 0.0
        @assert us.comparison_optimality >= 0.0
        @assert us.planning_depth > 0
        @assert us.planning_iterations > 0

        m = InventoryHeuristicPlanningMDP(ps, us)
        return new{CacheType}(m, BFSSolver(m, us.planning_depth, us.planning_iterations), us.multi_choice_optimality, us.comparison_optimality, cache)
    end
end

"""
    Returns the policy of a user model given a recommendation as a vector of probabilies.
"""
function get_policy(um::BoltzmannUserModel, s::InventoryState{Int64}, a_recommended::Union{ProduceProductsRecommendation,Missing})
    s_aug = InventoryState{Float64}(s)

    q_values = isnothing(um.cache) ? missing : get(um.cache, stateindex(InventoryPlanningMDP, s), missing)

    if ismissing(q_values)
        q_values = fill(-Inf, length(actions(um.world_model)))

        for (a,q) in find_q_values(um.value_estimator, s_aug)
            q_values[actionindex(um.world_model, a)] = q
        end

        if !isnothing(um.cache)
            setindex!(um.cache, q_values, stateindex(InventoryPlanningMDP, s))
        end
    end

    a_recommended_idx = ismissing(a_recommended) ? missing : actionindex(um.world_model, a_recommended.val)

    # prior selection probabilities
    probs = exp.(um.multi_choice_optimality .* q_values)
    probs ./= sum(probs)

    # Boltzmann-rational switch to recommended action
    # utility of switching is increase in Q-value from switching, utility of not switching is 0.
    if !ismissing(a_recommended)
        switch_prob = exp.(um.comparison_optimality .* (q_values[a_recommended_idx] .- q_values)) ./ (exp.(um.comparison_optimality .* (q_values[a_recommended_idx] .- q_values)) .+ 1.0)
        switch_prob[isnan.(switch_prob)] .= 1.0 # (actions with -Inf value will always cause a switch)
        switched_probs = zeros(length(q_values))
        switched_probs[a_recommended_idx] = sum(probs .* switch_prob)
        probs = (1.0 .- switch_prob) .* probs + switched_probs
    end

    return SparseCat(actions(um.world_model, s_aug), probs)
end

"""
    Simulate the user model on design s with recommendation a_recommended. Set a_recommended = missing if you there is no recommendation
"""
act(um::BoltzmannUserModel, s::InventoryState{Int64}, a_recommended::Union{ProduceProductsRecommendation,Missing} = missing; verbose = false) = rand(get_policy(um, s, a_recommended))

"""
Likelihood of action by user
"""
likelihood(um::BoltzmannUserModel, s::InventoryState{Int64}, a::ProduceProductsAction, a_recommended::Union{Missing,ProduceProductsRecommendation}) = pdf(get_policy(um, s, a_recommended), a)

#--------------------------------------------------------------------------------
# ASSISTANCE PROBLEM
#--------------------------------------------------------------------------------


mutable struct InventoryPlanningAssistanceMDP <: MDP{InventoryState{Int64}, Union{ProduceProductsAction,ProduceProductsRecommendation}}
    user_model::BoltzmannUserModel
    world_model::InventoryPlanningMDP

    enable_direct_intervention::Bool

    InventoryPlanningAssistanceMDP(um::BoltzmannUserModel, enable_direct_intervention::Bool = false) = new(um, InventoryPlanningMDP(um.world_model), enable_direct_intervention)
end

POMDPs.discount(m::InventoryPlanningAssistanceMDP) = m.world_model.discounting
POMDPs.initialstate(m::InventoryPlanningAssistanceMDP) = initialstate(m.world_model)

function POMDPs.gen(m::InventoryPlanningAssistanceMDP, s::InventoryState{Int64}, a::ProduceProductsAction, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false)
    @assert m.enable_direct_intervention
    return gen(m.world_model, s, a, rng)
end

function POMDPs.gen(m::InventoryPlanningAssistanceMDP, s::InventoryState{Int64}, a::ProduceProductsRecommendation, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false)
    a_sup = act(m.user_model, s, a; verbose = verbose)
    sp, r = @gen(:sp, :r)(m.world_model, s, a_sup, rng)
    return (sp=sp, r=r, info = Dict("user action" => a_sup))
end

function POMDPs.actions(m::InventoryPlanningAssistanceMDP, s::InventoryState{Int64})
    ret = Union{ProduceProductsAction,ProduceProductsRecommendation}[ProduceProductsRecommendation(a) for a in actions(m.world_model, s)]
    if m.enable_direct_intervention
        push!(ret, actions(m.world_model, s)...)
    end
    return ret
end
