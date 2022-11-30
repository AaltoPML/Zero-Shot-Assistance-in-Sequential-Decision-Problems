# This library contains all the code common to the day trip design experiments. It implements the basic problem, the agent models and the decision-theoretic formalization of AIAD for this specific problem.
#
#

module DaytripDesignAssistance

include("../LibBFSSolver.jl")

using Distributions
using Random
using POMDPs
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using MCTS
using LinearAlgebra
import Base: string, ==

export FINEARTS, POPCULTURE, SIGHTSEEING, HISTORICHOUSE, SHOPPING, HOME, POI, Daytrip, distance, travel_dist, travel_time, duration, cost, optimize_trip_cw
export DaytripEditType, DaytripEdit, EditRecommendation, DaytripAndInfo
export featurize
export MachineDaytripScheduleMDP, BasicHumanWorldModelMDP, act, ask, likelihood, get_edit_policy, get_answer_policy, utility
export MCTSUserModel, CachedBoltzmannUserModel, BoltzmannUserModel, MCTSUserModelSpecification, WorldModelSpecification, BoltzmannUserModelSpecification,  BasicHumanWorldModelSpecification, FoviatedHumanWorldModelSpecification, instantiate
export DaytripScheduleAssistanceMDP, RecommendOptimal


################################################################################
# DAYTRIP AND UTILITY DEFINITION
################################################################################

const global MAX_DAY_LENGTH = 60.0*12.0

struct POI
    coord_x::Float64 # x coordinate in km from center
    coord_y::Float64 # y coordinate in km from center
    topics::BitVector # Which (abstract!) topics does the POI belong to?
    visit_time::Float64 # visit time in minutes
    cost::Float64 # cost in eurodollars
    
    POI(coord_x::Float64, coord_y::Float64, topics::BitVector, visit_time::Float64 = 0.0, cost::Float64 = 0.0) = new(coord_x, coord_y, topics, visit_time, cost)
end

"""
Daytrips are represented by an array of POIs. POIs are visited in the order in which they appear in the array.
"""
Daytrip = Array{POI,1}

"""
measures the distance between POIs x and y
"""
distance(x::POI, y::POI) = sqrt((x.coord_x - y.coord_x)^2 + (x.coord_y - y.coord_y)^2)

"""
measures the included angle between AB and BC
"""
function included_angle(A::POI, B::POI, C::POI)
    BA = [A.coord_x - B.coord_x, A.coord_y - B.coord_y]
    BC = [C.coord_x - B.coord_x, C.coord_y - B.coord_y]
    acos(min(max(sum(BA .* BC) / (sqrt(sum(BA.^2)) * sqrt(sum(BC.^2))), -1.0), 1.0))
end

"""
Calculates the travel distance for a trip (in km)

parameters:
    trip::Daytrip:
        The daytrip
"""
travel_dist(trip::Daytrip)::Float64 = sum(Array{Float64,1}([distance(trip[i], trip[i%length(trip)+1]) for i in 1:length(trip)]))

"""
Calculates how long it will take to travel between all POIs in a trip (in minutes)

parameters:
    trip::Daytrip:
        The daytrip
    movement_speed::Float64:
        Speed at which distance is covered in km/h.
        default: 5.0
"""
travel_time(trip::Daytrip; movement_speed::Float64 = 5.0)::Float64 = travel_dist(trip) / movement_speed * 60.0

"""
Calculates how long it will take to visit all POIs in a trip (in minutes)

parameters:
    trip::Daytrip:
        The daytrip
    movement_speed::Float64:
        Speed at which distance is covered in km/h.
        default: 5.0
"""
function duration(trip::Daytrip; movement_speed::Float64 = 5.0)::Float64
    if length(trip) == 0
        return 0.0
    end
    move_time = travel_time(trip, movement_speed = movement_speed)
    visit_time = sum(poi.visit_time for poi in trip)
    return move_time + visit_time
end

"""
Calculates the cost to visit all POIs in a trip (in eurodollars)

parameters:
    trip::Daytrip:
        The daytrip
"""
cost(trip::Daytrip)::Float64 = sum(Array{Float64,1}([poi.cost for poi in trip]))

# implements Clarke-Wright savings heuristic for TSP problems for finding the optimal order for s
# https://www.cs.ubc.ca/~hutter/previous-earg/EmpAlgReadingGroup/TSP-JohMcg97.pdf
function optimize_trip_cw(s)
    if length(s) == 1
        return s
    end
    D = [distance(s[i],s[j]) for i in 1:length(s), j in 1:length(s)]
    loops = [[j] for j in 2:length(s)]
    while length(loops) > 1
        savings = [D[1,i[end]] + D[1,j[1]] - D[i[end],j[1]] for i in loops, j in loops]
        best = argmax(savings - (maximum(savings) * I))
        idx1, idx2 = best.I
        select = trues(length(loops))
        select[[idx1,idx2]] .= false
        loops = vcat(loops[select], [vcat(loops[idx1], loops[idx2])])
    end
    return s[vcat(1, loops[1])]
end


################################################################################
# DAYTRIP DESIGN AS AN MDP
################################################################################

@enum DaytripEditType SWITCH NOOP

# SWITCH adds or removes POI at -index- in POI list to the trip
# NOOP does nothing
struct DaytripEdit
    edit_type::DaytripEditType
    index::Int32
end

function string(e::DaytripEdit)
    if e.edit_type == NOOP
        return "noop"
    elseif e.edit_type == SWITCH
        return "switch " * string(e.index)
    else
        @warn "encountered unknown type in string conversion"
        return ""
    end
end

abstract type DaytripScheduleMDP <: MDP{Daytrip, DaytripEdit} end

"""
This function allows you to transform a Daytrip into the statespace used by the user's world model. It is used right before MCTS needs to be run on the world model for a given Daytrip state.
"""
translate_state(m::DaytripScheduleMDP, s::Daytrip) = s

POMDPs.discount(::DaytripScheduleMDP) = 0.99
POMDPs.initialstate(m::DaytripScheduleMDP) = Deterministic([m.home])
POMDPs.isterminal(::DaytripScheduleMDP, ::Daytrip) = false

function POMDPs.actions(m::DaytripScheduleMDP)
    ret = Array{DaytripEdit,1}()
    
    push!(ret, DaytripEdit(NOOP::DaytripEditType, 0))   
    for i in 1:length(m.POIs)
        push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
    end
    
    return ret
end

function POMDPs.actions(m::DaytripScheduleMDP, s::Daytrip)
    ret = Array{DaytripEdit,1}()
    
    # NOOP action
    push!(ret, DaytripEdit(NOOP::DaytripEditType, 0))
       
    # don't add actions that would lengthen the trip if it is full
    if duration(s) > MAX_DAY_LENGTH
        for i in 1:length(m.POIs)
            if m.POIs[i] in s
                push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
            end
        end
    else
        for i in 1:length(m.POIs)
            push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
        end
    end
    
    return ret
end

POMDPs.reward(m::DaytripScheduleMDP, s::Daytrip, ::DaytripEdit, sp::Daytrip) = utility(m, sp) - utility(m, s)

function POMDPs.actionindex(m::DaytripScheduleMDP, a::DaytripEdit)::Int64
    if a.edit_type == DaytripDesignAssistance.NOOP
        return 1
    elseif a.edit_type == DaytripDesignAssistance.SWITCH
        return 1 + a.index
    else
        @error "unknown action encoutered in actionindex"
    end
end

"""
    Returns a multiple-hot encoded representation of state s suitable for ML.
"""
function featurize(m::DaytripScheduleMDP, s::Daytrip)::Array{Float32,1}
    sf = zeros(Float32, length(m.POIs))
    for (i,p) in enumerate(m.POIs)
        if p in s
            sf[i] = 1.0f0
        end
    end
    return sf
end

POMDPs.stateindex(m::DaytripScheduleMDP, s::Daytrip)::BigInt = sum(BigInt(v == 1)*convert(BigInt, 2)^(i-1) for (i,v) in enumerate(featurize(m, s))) + 1

"""
Calculate the value of a trip (the utility of a design)

params:
    m<:DaytripScheduleMDP
        The human model for which we want  to evaluate the utility
    trip::Daytrip:
        The daytrip
"""
function utility(m::DaytripScheduleMDP, trip::Daytrip)
    cost_factor = 1 - cdf(truncated(Normal(m.cost_pref_mean, m.cost_pref_std), 0.0, Inf), cost(trip))
    
    # calculate average fun per minute
    fun_factor = 0.0
    for poi in trip
        n_topics = sum(poi.topics)
        fun_factor += poi.visit_time * sum(poi.topics .* m.topic_interests) / (n_topics == 0 ? 1 : n_topics)
    end
    fun_factor += travel_time(trip) * m.travel_dislike
    fun_factor /= MAX_DAY_LENGTH
    
    return fun_factor * cost_factor
end

"""
    Returns the relevant paramters for a <:DaytripScheduleMDP in format suitable for ML.

    NOTE: this implements some normalization which may not be suitable for your prior.
"""
featurize(mdp::DaytripScheduleMDP)::Array{Float32,1} = convert(Array{Float32,1}, vcat(mdp.topic_interests, mdp.travel_dislike, mdp.cost_pref_std / 15.0, mdp.cost_pref_mean / 100.0))


################################################################################
# WORLD MODELS
################################################################################

# This is a human conception of the daytrip design task. It differs from the real task in that the TSP problem that is part of trip planning is solved heuristically.
mutable struct BasicHumanWorldModelMDP <: DaytripScheduleMDP
    # state space definition
    POIs::Array{POI,1}
    home::POI
    
    # utility definition
    topic_interests::BitVector
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64

    # action foviation
    consideration_distance::Float64

    function BasicHumanWorldModelMDP(
        POIs::Array{POI,1},
        home::POI,
        topic_interests::BitVector,
        travel_dislike::Float64,
        cost_pref_mean::Float64,
        cost_pref_std::Float64,
        consideration_distance::Float64 = Inf)

        @assert (travel_dislike <= 1.0) && (travel_dislike >= 0.0) "Travel dislike must be in [0.0, 1.0]!"
        @assert (cost_pref_mean >= 0.0) "Cost preference mean parameter must be positive"
        @assert (cost_pref_std > 0.0) "Cost preference std parameter must be strictly positive"
        @assert (consideration_distance > 0.0) "Consideration distance must be strictly positive"
        new(POIs, home, topic_interests, travel_dislike, cost_pref_mean, cost_pref_std, consideration_distance)
    end
end

# Humans can't solve the TSP at every planning step so here it is solved (iteratively) using a visual heuristic. If a POI is removed the order remains unchanged. If a POI is added it is inserted its location is chosen using the maximum angle heuristic.
function POMDPs.gen(m::BasicHumanWorldModelMDP, s::Daytrip, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = copy(s)
    if a.edit_type == NOOP::DaytripEditType
        # noop does nothing
    elseif a.edit_type == SWITCH::DaytripEditType
        idx = findfirst(x -> x == m.POIs[a.index], sp)
        
        if isnothing(idx)
            if length(sp) == 1
                push!(sp, m.POIs[a.index])
            else # maximum angle heuristic
                loc = argmax([included_angle(sp[i], m.POIs[a.index], sp[i%length(sp)+1]) for i in 1:length(sp)])
                insert!(sp, loc+1, m.POIs[a.index])
            end
        else
            deleteat!(sp, idx)
        end
    else
        @error "unknown action encountered in gen"
    end
    
    return (sp=sp, r=POMDPs.reward(m, s, a, sp), info=missing)
end

"""
Returns whether P is within distance δ of the line segment between P1 and P2
"""
function is_close_to_path(P::POI, P1::POI, P2::POI, δ::Float64)
    @assert δ >= 0.0
    # is P within δ of either P1 or P2?
    if (distance(P1, P) < δ) || (distance(P2, P) < δ)
        return true
    end

    # line formula: ax - y + c = 0
    a = (P2.coord_y - P1.coord_y) / (P2.coord_x - P1.coord_x)
    c = -P1.coord_x * (P2.coord_y - P1.coord_y) / (P2.coord_x - P1.coord_x) + P1.coord_y

    # where does P project onto the line through P1,P2?
    Zx = ((P.coord_x + a*P.coord_y) - a*c) / (a^2 + 1.0)
    Zy = (a*(P.coord_x + a*P.coord_y) + c) / (a^2 + 1.0)

    # is P within δ of z, and if so does Z lie on the line segment between P1 and P2?
    if (sqrt((Zx - P.coord_x)^2 + (Zy - P.coord_y)^2) < δ) && (((Zx - P2.coord_x) / (P1.coord_x - P2.coord_x) <= 1.0) && ((Zx - P2.coord_x) / (P1.coord_x - P2.coord_x) >= 0.0))
        return true
    end
    return false
end

function POMDPs.actions(m::DaytripScheduleMDP, s::Daytrip)
    ret = Array{DaytripEdit,1}()
    
    # NOOP action
    push!(ret, DaytripEdit(NOOP::DaytripEditType, 0))
       
    # don't add actions that would lengthen the trip if it is full
    if duration(s) > MAX_DAY_LENGTH
        for i in 1:length(m.POIs)
            if m.POIs[i] in s
                push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
            end
        end
    else
        for i in 1:length(m.POIs)
            if any(is_close_to_path(m.POIs[i], s[j], s[j == (length(s)) ? 1 : j+1], m.consideration_distance) for j in 1:length(s))
                push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
            end
        end
    end
    
    return ret
end

################################################################################
# RECOMMENDATIONS, QUESTIONS, AND ANSWERS
################################################################################

"""
Recommendation for a specific edit
"""
struct EditRecommendation
    edit::DaytripEdit
end

string(a::EditRecommendation) = "try " * string(a.edit)

"""
This is an extended state representation for the problem which captures both the
state of the design problem (daytrip) and any additional info coming from the
user (mostly answers to questions).
"""
struct DaytripAndInfo
    trip::Daytrip
    UserEdit::Union{Missing,DaytripEdit}

    DaytripAndInfo(trip::Daytrip, UserEdit::DaytripEdit) = new(trip, UserEdit)
    DaytripAndInfo(trip::Daytrip) = new(trip, missing)
end

function ==(a::DaytripAndInfo, b::DaytripAndInfo)
    if a.trip != b.trip
        return false
    end
    if ismissing(a.UserEdit) ⊻ ismissing(b.UserEdit)
        return false
    end
    return (ismissing(a.UserEdit) && ismissing(b.UserEdit)) || (a.UserEdit == b.UserEdit)
end

travel_dist(s::DaytripAndInfo) = travel_dist(s.trip)
travel_time(s::DaytripAndInfo; movement_speed::Float64 = 5.0) = travel_time(s.trip; movement_speed = movement_speed)
duration(s::DaytripAndInfo; movement_speed::Float64 = 5.0) = duration(s.trip; movement_speed = movement_speed)
cost(s::DaytripAndInfo) = cost(s.trip)

has_useredit(s::DaytripAndInfo) = !ismissing(s.UserEdit)


################################################################################
# USER MODELS
################################################################################

#
# Boltzmann Rational User model
#

"""
    Abstract type for user models based on an MCTS value estimator. Instances must have fields:
        world_model::BasicHumanWorldModelMDP
        value_estimator::BFSSolver
"""
abstract type MCTSUserModel end

"""
User model built on Boltzmann rational reasoning

cache is a cache to hold q_value calculations for re-use. It should map a state index (Int) to a vector (Array{Float64,1}) good options are LRU{Int,Array{Float64,1}} and Dict{Int,Array{Float64,1}}
"""
struct CachedBoltzmannUserModel{CacheType,WorldModel} <: MCTSUserModel where CacheType where WorldModel
    world_model::WorldModel
    value_estimator::BFSSolver
    cache::CacheType
    multi_choice_optimality::Float64 # optimality when choosing edits
    comparison_optimality::Float64 # optimality when comparing edits
    abstract_comparison_optimality::Float64 # optimality when comparing abstract objects
    
    function CachedBoltzmannUserModel{CacheType,WorldModel}(m::WorldModel,
                                                            planning_depth::Int64,
                                                            multi_choice_optimality::Float64,
                                                            comparison_optimality::Float64,
                                                            cache::CacheType;
                                                            planning_iterations::Int64 = 500) where CacheType where WorldModel
        @assert multi_choice_optimality >= 0.0
        @assert comparison_optimality >= 0.0
        return new{CacheType,WorldModel}(m,
                                         BFSSolver(m, planning_depth, planning_iterations),
                                         cache,
                                         multi_choice_optimality,
                                         comparison_optimality)
    end

    function CachedBoltzmannUserModel{Nothing,WorldModel}(m::WorldModel,
                                                          planning_depth::Int64,
                                                          multi_choice_optimality::Float64,
                                                          comparison_optimality::Float64;
                                                          planning_iterations::Int64 = 500) where WorldModel
        @assert multi_choice_optimality >= 0.0
        @assert comparison_optimality >= 0.0
        return new{Nothing,WorldModel}(m,
                                       BFSSolver(m, planning_depth, planning_iterations),
                                       nothing,
                                       multi_choice_optimality,
                                       comparison_optimality)
    end
end

BoltzmannUserModel{WorldModel} = CachedBoltzmannUserModel{Nothing,WorldModel}

function Base.show(io::IO, um::CachedBoltzmannUserModel{CacheType,WorldModel}) where CacheType where WorldModel
    println(io, "  HOME  : (", um.world_model.home.coord_x, ", ", um.world_model.home.coord_y, ")")
    println(io, "  #POIs : ", length(um.world_model.POIs))
    println(io, "-------------- PLANNING ------------")
    println(io, "  depth                          : ", um.value_estimator.planning_depth)
    println(io, "  n_iterations                   : ", um.value_estimator.n_iterations)
    s = string(um.multi_choice_optimality)
    println(io, "  multi_choice_optimality        : ", s[1:min(6,length(s))])
    s = string(um.comparison_optimality)
    println(io, "  comparison_optimality          : ", s[1:min(6,length(s))])
    println(io, "-------------- UTILITY --------------")
    println(io, "  topic_interests : ", um.world_model.topic_interests)
    s = string(um.world_model.travel_dislike)
    println(io, "  travel_dislike       : ", s[1:min(6,length(s))])
    s = string(um.world_model.cost_pref_mean)
    println(io, "  cost_pref_mean       : ", s[1:min(6,length(s))])
    s = string(um.world_model.cost_pref_std)
    println(io, "  cost_pref_std        : ", s[1:min(6,length(s))])
end

"""
    Returns the relevant paramters for a BoltzmannUserModel in format suitable for ML.
"""
featurize(um::CachedBoltzmannUserModel{CacheType,WorldModel}) where CacheType where WorldModel = convert(Array{Float32,1}, vcat(um.value_estimator.solver.depth, um.multi_choice_optimality, um.comparison_optimality,  featurize(um.world_model)))

#
# Editing policy
#

"""
    Returns the policy of a user model given a recommendation as a vector of probabilies.
    
    action a's probability is located at index actionindex(HumanDaytripMDP, a)
"""
function get_edit_policy(um::CachedBoltzmannUserModel{CacheType}, s::Daytrip, a_recommended::Union{EditRecommendation,Missing}) where CacheType
    q_values = isnothing(um.cache) ? missing : get(um.cache, stateindex(um.world_model, s), missing)

    if ismissing(q_values)
        s_aug = translate_state(um.world_model, s)
        q_values = fill(-Inf, length(actions(um.world_model)))

        for (a,q) in find_q_values(um.value_estimator, s_aug)
            q_values[actionindex(um.world_model, a)] = q
        end

        if !isnothing(um.cache)
            setindex!(um.cache, q_values, stateindex(um.world_model, s))
        end
    end

    noop_idx = actionindex(um.world_model, DaytripEdit(NOOP, 1))
    a_recommended_idx = ismissing(a_recommended) ? missing : actionindex(um.world_model, a_recommended.edit)

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

    # Boltzmann-rational switch to NOOP
    switch_prob = exp.(um.comparison_optimality .* (q_values[noop_idx] .- q_values)) ./ (exp.(um.comparison_optimality .* (q_values[noop_idx] .- q_values)) .+ 1.0)
    switch_prob[isnan.(switch_prob)] .= 1.0
    switched_probs = zeros(length(q_values))
    switched_probs[noop_idx] = sum(probs .* switch_prob)
    probs = (1.0 .- switch_prob) .* probs + switched_probs

    avail_actions = .!isinf.(q_values)
    return SparseCat(actions(um.world_model)[avail_actions], probs[avail_actions])
end

get_edit_policy(um::CachedBoltzmannUserModel{CacheType,WorldModel}, s::DaytripAndInfo, a_recommended::Union{EditRecommendation,Missing}) where CacheType where WorldModel = get_edit_policy(um, s.trip, a_recommended)

"""
    Simulate the user model on design s with recommendation a_recommended. Set a_recommended = missing if you there is no recommendation
"""
act(um::CachedBoltzmannUserModel{CacheType,WorldModel}, s::Daytrip, a_recommended::Union{EditRecommendation,Missing} = missing; verbose = false) where CacheType where WorldModel = rand(get_edit_policy(um, s, a_recommended))
act(um::CachedBoltzmannUserModel{CacheType,WorldModel}, s::DaytripAndInfo, a_recommended::Union{EditRecommendation,Missing} = missing; verbose = false) where CacheType where WorldModel = act(um, s.trip, a_recommended; verbose = verbose)

"""
Likelihood of edit by user
"""
likelihood(um::CachedBoltzmannUserModel{CacheType,WorldModel}, s::Daytrip, a::DaytripEdit, a_recommended::Union{Missing,EditRecommendation}) where CacheType where WorldModel = pdf(get_edit_policy(um, s, a_recommended), a)

"""
General definition of likelihood that is uniform across all interactions
"""
likelihood(um::CachedBoltzmannUserModel{cacheType,WorldModel}, s::DaytripAndInfo, a_AI::Union{Missing,EditRecommendation}, sp::DaytripAndInfo) where cacheType where WorldModel = likelihood(um, s.trip, sp.UserEdit, a_AI)

################################################################################
# USER MODEL SPECIFICATIONS
################################################################################
"""
Abstract type for user model specifications. These only capture the parameters of the user model.
"""
abstract type MCTSUserModelSpecification end

"""
Abstract type for world model specifications. These only capture the parameters of the world model (primarily utility).
"""
abstract type WorldModelSpecification end

struct BasicHumanWorldModelSpecification <: WorldModelSpecification
    # utility:
    topic_interests::BitVector
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64
end

struct FoviatedHumanWorldModelSpecification <: WorldModelSpecification
    # utility:
    topic_interests::BitVector
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64
    consideration_distance::Float64
end

struct BoltzmannUserModelSpecification <: MCTSUserModelSpecification
    # user model:
    planning_depth::Int64
    n_iterations::Int64
    multi_choice_optimality::Float64
    comparison_optimality::Float64
end

"""
    Instantiate a user model based on the given spec.
"""
function instantiate(wm_spec::BasicHumanWorldModelSpecification, um_spec::BoltzmannUserModelSpecification, home::POI, POIs::Array{POI, 1}, cache::CacheType) where CacheType
    return CachedBoltzmannUserModel{CacheType,BasicHumanWorldModelMDP}(BasicHumanWorldModelMDP(POIs,
                                               home,
                                               wm_spec.topic_interests,
                                               wm_spec.travel_dislike,
                                               wm_spec.cost_pref_mean,
                                               wm_spec.cost_pref_std),
    um_spec.planning_depth, um_spec.multi_choice_optimality, um_spec.comparison_optimality, cache, planning_iterations = um_spec.n_iterations)
end

function instantiate(wm_spec::FoviatedHumanWorldModelSpecification, um_spec::BoltzmannUserModelSpecification, home::POI, POIs::Array{POI, 1}, cache::CacheType) where CacheType
    return CachedBoltzmannUserModel{CacheType,BasicHumanWorldModelMDP}(BasicHumanWorldModelMDP(POIs,
                                               home,
                                               wm_spec.topic_interests,
                                               wm_spec.travel_dislike,
                                               wm_spec.cost_pref_mean,
                                               wm_spec.cost_pref_std,
                                               wm_spec.consideration_distance),
    um_spec.planning_depth, um_spec.multi_choice_optimality, um_spec.comparison_optimality, cache, planning_iterations = um_spec.n_iterations)
end

instantiate(wm_spec::WM where WM, um_spec::BoltzmannUserModelSpecification, home::POI, POIs::Array{POI, 1}) = instantiate(wm_spec, um_spec, home, POIs, nothing)

instantiate(um_pair::Pair{WM,BoltzmannUserModelSpecification} where WM, home::POI, POIs::Array{POI, 1}) = instantiate(um_pair.first, um_pair.second, home, POIs)

instantiate(um_pair::Pair{WM,BoltzmannUserModelSpecification} where WM, home::POI, POIs::Array{POI, 1}, cache::CacheType) where CacheType = instantiate(um_pair.first, um_pair.second, home, POIs, cache)

function Base.show(io::IO, um_spec::BoltzmannUserModelSpecification)
    println(io, "BoltzmannUserModel Specification:")
    println(io, "  depth                          : ", um_spec.planning_depth)
    println(io, "  n_iterations                   : ", um_spec.n_iterations)
    s = string(um_spec.multi_choice_optimality)
    println(io, "  multi_choice_optimality        : ", s[1:min(6,length(s))])
    s = string(um_spec.comparison_optimality)
    println(io, "  comparison_optimality          : ", s[1:min(6,length(s))])
end
                                            
function Base.show(io::IO, um_spec::BasicHumanWorldModelSpecification)
    println(io, "BasicHumanWorldModel Specification:")
    println(io, "  topic_interests         : ", um_spec.topic_interests)
    s = string(um_spec.travel_dislike)
    println(io, "  travel_dislike          : ", s[1:min(6,length(s))])
    s = string(um_spec.cost_pref_mean)
    println(io, "  cost_pref_mean          : ", s[1:min(6,length(s))])
    s = string(um_spec.cost_pref_std)
    println(io, "  cost_pref_std           : ", s[1:min(6,length(s))])
end
                                                
################################################################################
# OBJECTIVE WORLD MODEL
################################################################################

mutable struct MachineDaytripScheduleMDP <: DaytripScheduleMDP
    # state space definition
    POIs::Array{POI,1}
    home::POI
    
    # utility definition
    topic_interests::BitVector
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64

    # hyperparameters
    discounting::Float64
    optimizer::Symbol
      
    function MachineDaytripScheduleMDP(
        POIs::Array{POI,1},
        home::POI,
        topic_interests::BitVector,
        travel_dislike::Float64,
        cost_pref_mean::Float64,
        cost_pref_std::Float64,
        discounting::Float64;
        optimizer::Symbol = :CW)

        @assert (travel_dislike <= 1.0) && (travel_dislike >= 0.0) "Travel dislike must be in [0.0, 1.0]!"
        @assert (cost_pref_mean >= 0.0) "Cost preference mean parameter must be positive"
        @assert (cost_pref_std > 0.0) "Cost preference std parameter must be strictly positive"
        new(POIs, home, topic_interests, travel_dislike, cost_pref_mean, cost_pref_std, discounting, optimizer)
    end

    function MachineDaytripScheduleMDP(
        wm::BasicHumanWorldModelMDP,
        discounting::Float64;
        optimizer::Symbol = :CW)
        
        new(wm.POIs, wm.home, wm.topic_interests, wm.travel_dislike, wm.cost_pref_mean, wm.cost_pref_std, discounting, optimizer)
    end
    
    function MachineDaytripScheduleMDP(
        home::POI,
        POIs::Array{POI,1},
        wm_spec::Union{BasicHumanWorldModelSpecification,FoviatedHumanWorldModelSpecification},
        discounting::Float64;
        optimizer::Symbol = :CW)
        
        new(POIs, home, wm_spec.topic_interests, wm_spec.travel_dislike, wm_spec.cost_pref_mean, wm_spec.cost_pref_std, discounting, optimizer)
    end
end

function POMDPs.gen(m::MachineDaytripScheduleMDP, s::Daytrip, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = copy(s)

    if a.edit_type == NOOP::DaytripEditType
        # noop does nothing
    elseif a.edit_type == SWITCH::DaytripEditType
        idx = findfirst(x -> x == m.POIs[a.index], sp)
        
        if isnothing(idx)
            push!(sp, m.POIs[a.index])
        else
            deleteat!(sp, idx)
        end
        if m.optimizer == :CW
            sp = optimize_trip_cw(sp)
        else
            @error "specified TSP solver for planning does not exist"
        end
    else
        @error "unknown action encountered in gen"
    end

    return (sp=sp, r=POMDPs.reward(m, s, a, sp), info=missing)
end
                                                                                            
function POMDPs.actions(m::MachineDaytripScheduleMDP, s::Daytrip)
    ret = Array{DaytripEdit,1}()
    
    # NOOP action
    push!(ret, DaytripEdit(NOOP::DaytripEditType, 0))
       
    # don't add actions that would lengthen the trip if it is full
    if duration(s) > MAX_DAY_LENGTH
        for i in 1:length(m.POIs)
            if m.POIs[i] in s
                push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
            end
        end
    else
        for i in 1:length(m.POIs)
            push!(ret, DaytripEdit(SWITCH::DaytripEditType, i))
        end
    end
    
    return ret
end


POMDPs.discount(m::MachineDaytripScheduleMDP) = m.discounting

################################################################################
# ASSISTANCE MDP
################################################################################

mutable struct DaytripScheduleAssistanceMDP <: MDP{DaytripAndInfo, Union{DaytripEdit,EditRecommendation}}
    user_model::MCTSUserModel
    objective_world_model::MachineDaytripScheduleMDP

    # these are technically already present in user_model but they're stored here too
    POIs::Array{POI,1}
    home::POI

    enable_direct_edits::Bool

    discounting::Float64

    optimizer::Symbol

    DaytripScheduleAssistanceMDP(m::MCTSUserModel, discounting::Float64; enable_direct_edits::Bool = false, optimizer = :CW) = new(m, MachineDaytripScheduleMDP(m.world_model, discounting, optimizer = optimizer), m.world_model.POIs, m.world_model.home, enable_direct_edits, discounting, optimizer)
end

POMDPs.discount(m::DaytripScheduleAssistanceMDP) = m.discounting
POMDPs.initialstate(m::DaytripScheduleAssistanceMDP) = Deterministic(DaytripAndInfo([m.home]))
POMDPs.isterminal(::DaytripScheduleAssistanceMDP, ::DaytripAndInfo) = false

utility(m::DaytripScheduleAssistanceMDP, s::DaytripAndInfo) = utility(m.objective_world_model, s.trip)

"""
Helper function for gen. Implements the daytrip editing logic.
"""
apply_trip_edit(m::DaytripScheduleAssistanceMDP, s::Daytrip, a::DaytripEdit) = @gen(:sp)(m.objective_world_model, s, a)

function POMDPs.gen(m::DaytripScheduleAssistanceMDP, s::DaytripAndInfo, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false)
    @assert m.enable_direct_edits
    sp = DaytripAndInfo(apply_trip_edit(m, s.trip, a))
    return (sp=sp, r=POMDPs.reward(m, s, a, sp), info=missing)
end

function POMDPs.gen(m::DaytripScheduleAssistanceMDP, s::DaytripAndInfo, a::Union{Missing,EditRecommendation}, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false)
    a_sup = act(m.user_model, s, a; verbose = verbose)
    sp = DaytripAndInfo(apply_trip_edit(m, s.trip, a_sup), a_sup)
    return (sp=sp, r=POMDPs.reward(m, s, a_sup, sp), info=missing)
end

POMDPs.reward(m::DaytripScheduleAssistanceMDP, s::DaytripAndInfo, ::Any, sp::DaytripAndInfo) = POMDPs.discount(m) * utility(m, sp) - utility(m, s)

function POMDPs.actions(m::DaytripScheduleAssistanceMDP, s::DaytripAndInfo)
    ret = Array{Union{DaytripEdit,EditRecommendation},1}()
    push!(ret, EditRecommendation(DaytripEdit(NOOP::DaytripEditType, 0)))
    if m.enable_direct_edits
        push!(ret, actions(m.objective_world_model, s.trip)...)
    end
    return ret
end

"""
WARNING action indices do not necessarily match with indices within the list returned by action()
"""
POMDPs.actionindex(m::DaytripScheduleAssistanceMDP, a::EditRecommendation)::Int64 = POMDPs.actionindex(m.objective_world_model, a.edit)
POMDPs.actionindex(m::DaytripScheduleAssistanceMDP, a::DaytripEdit)::Int64 = POMDPs.actionindex(m.objective_world_model, a.edit) + length(m.POIs)

#
# Rollout Policies
#

"""
Under this rollout policy the AI will simply recommend the action with highest Q-value under the user's world model.
"""
mutable struct RecommendOptimal <: Policy
    user_model::MCTSUserModel
end

function POMDPs.action(p::RecommendOptimal, s::DaytripAndInfo)
    return action(p.user_model.value_estimator, s.trip)
end

end # DaytripDesignAssistance
