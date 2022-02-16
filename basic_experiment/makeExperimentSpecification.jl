using Pkg
Pkg.activate("../.")

include("../LibTravelPlanner.jl")
using .DaytripDesignAssistance
using JLD2
using Distributions

N_EXPERIMENTS = 100 # Number of experiments you'd like to run. All experiments are run on the same city but with different simulated agents and particle sets.
N_PARTICLES = 1024 # Number of particles over which beliefs are kept.
N_TOPICS = 20
REWARD_SCALING = 10.0 # scaling to bring rewards into roughtly a [-1, 1] interval
P_TOPIC = 0.10
P_INTEREST = 0.3

function generate_city(N_POIs::Int64 = 100; x_limits::Float64 = 5.0, y_limits::Float64 = 5.0)
    location_distribution = MvNormal(2, 1.15)

    cityPOIs = Array{POI,1}()
    for _ in 1:N_POIs
        c_x = Inf
        c_y = Inf
        while (c_x > x_limits) || (c_x < -x_limits) || (c_y > y_limits) || (c_y < -y_limits)
            c_x, c_y = rand(location_distribution, 1)
        end

        c_cost = rand(truncated(Normal(10.0, 3.0), 0.0, Inf))
        c_visit_time = rand(truncated(Normal(30.0, 20.0), 0.0, 100.0))
        c_topics = BitVector([rand(Bernoulli(P_TOPIC)) for _ in 1:N_TOPICS])
        p = POI(c_x, c_y, c_topics, c_visit_time, c_cost)
        push!(cityPOIs, p)
    end
    return cityPOIs
end

function sample_spec_from_prior(::Type{BoltzmannUserModel}; n_iterations::Int64 = 500)
    planning_depth = 3
    multi_choice_optimality = rand(Uniform(1.0, 4.0)) * REWARD_SCALING
    comparison_optimality = 10.0 * multi_choice_optimality

    return BoltzmannUserModelSpecification(planning_depth, n_iterations, multi_choice_optimality, comparison_optimality)
end

function sample_spec_from_belief_prior(::Type{BoltzmannUserModel}; n_iterations::Int64 = 500)
    spec = sample_spec_from_prior(BoltzmannUserModel, n_iterations = n_iterations)

    return BoltzmannUserModelSpecification(spec.planning_depth, spec.n_iterations, 2.0 * REWARD_SCALING, 20.0 * REWARD_SCALING)
end

function sample_spec_from_prior(::Type{BasicHumanWorldModelMDP})
    topic_interests = BitVector([rand(Bernoulli(P_INTEREST)) for _ in 1:N_TOPICS])

    travel_dislike = 0.0
    cost_pref_std = 10.0
    cost_pref_mean = rand(Normal(100.0, 25.0)) + 4 * cost_pref_std

    return BasicHumanWorldModelSpecification(topic_interests, travel_dislike, cost_pref_mean, cost_pref_std)
end

user_models = Array{Array{Pair{BasicHumanWorldModelSpecification,BoltzmannUserModelSpecification},1},1}()
true_user_model = Array{Pair{BasicHumanWorldModelSpecification,BoltzmannUserModelSpecification},1}()
POIs = Array{Array{POI,1},1}()
home = Array{POI,1}()
for _ in 1:N_EXPERIMENTS
    push!(user_models, [Pair(sample_spec_from_prior(BasicHumanWorldModelMDP), sample_spec_from_belief_prior(BoltzmannUserModel)) for i = 1:N_PARTICLES])
    push!(true_user_model, Pair(sample_spec_from_prior(BasicHumanWorldModelMDP), sample_spec_from_prior(BoltzmannUserModel)))
    push!(POIs, generate_city(100))
    push!(home, POI(0.0, 0.0, falses(N_TOPICS)))
end

@save "E0_basic_experiment.jld" home POIs user_models true_user_model
