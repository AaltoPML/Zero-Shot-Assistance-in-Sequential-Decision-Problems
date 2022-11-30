#  This file is used to generate the specification files which allow us to run our experiments.
#
#

using Pkg
Pkg.activate("../../.")

include("../LibInventoryManager.jl")
using JLD2
using Distributions

MAX_EPISODE_LENGTH = 100
N_EXPERIMENTS = 100 # Number of experiments you'd like to run. All experiments are run on the same city but with different simulated agents and particle sets.
N_PARTICLES = 2048 # Number of particles over which beliefs are kept.
N_PRODUCTS = 3
OPTIMISM_STDDEV = 2.0

function create_demand_distribution(μ, σ, min_bin_size = 0.01)
    d = TruncatedNormal(μ, σ, 0.0, Inf64)

    i = 0
    bins = Int64[]
    probs = Float64[]
    while cdf(d, Float64(i)+0.5) < (1.0 - min_bin_size)
        push!(probs, cdf(d, Float64(i)+0.5) - sum(probs))
        push!(bins, i)
        i += 1
    end
    push!(probs, 1 - cdf(d, Float64(i)-0.5))
    push!(bins, i)
    return DemandPrediction(bins, probs)
end

function generate_demand()
    product_demand_forecasts = Array{Array{DemandPrediction,1},1}()
    for _ in 1:N_PRODUCTS
        means = rand(TruncatedNormal(2.0, 0.75, 0.0, 5.0), MAX_EPISODE_LENGTH)
        stds = rand(Chisq(1.25), MAX_EPISODE_LENGTH)
        push!(product_demand_forecasts, DemandPrediction[create_demand_distribution(μ, σ) for (μ, σ) in zip(means, stds)])
    end
    return hcat(product_demand_forecasts...)
end

function sample_user_spec()
    product_profits = rand(Uniform(0.0, 1.0), N_PRODUCTS)
    product_profits[rand(1:N_PRODUCTS)] = 1.0
    
    inventory_cost = rand(Beta(2.5,8))
    lost_business_cost = rand(Beta(3,3))
    optimism = rand(TruncatedNormal(0.0, 1.5, -3.0, 3.0))
    # alternatively we could sample the degree of optimism from rand([])

    # ps = ProblemSpecification(d, 12, 0.99)
    # us = UserSpecification([1.0, 0.5, 0.75], 0.15, 0.5, 0.0, 1.0, 10.0, 3, 500)

    return UserSpecification(product_profits, inventory_cost, lost_business_cost, optimism, 1.0, 10.0, 2, 300)
end

function copy_US_with_optimism(us::UserSpecification, optimism::Float64)
    return UserSpecification(us.product_profit,
                             us.inventory_cost, 
                             us.lost_business_cost, 
                             optimism, 
                             us.multi_choice_optimality, 
                             us.comparison_optimality, 
                             us.planning_depth, 
                             us.planning_iterations)
end


user_models_modeled = Array{Array{UserSpecification,1},1}()
user_models_not_modeled = Array{Array{UserSpecification,1},1}()
user_models_not_modeled_assumed_optimistic = Array{Array{UserSpecification,1},1}()
user_models_not_modeled_assumed_pessimistic = Array{Array{UserSpecification,1},1}()
true_user_model = Array{UserSpecification,1}()
problem_spec = Array{ProblemSpecification,1}()

for _ in 1:N_EXPERIMENTS
    push!(problem_spec, ProblemSpecification(generate_demand(), 12, 0.99))
    push!(true_user_model, sample_user_spec())
    particle_set = [sample_user_spec() for _ = 1:N_PARTICLES]
    push!(user_models_modeled, particle_set)
    push!(user_models_not_modeled, [copy_US_with_optimism(us, 0.0) for us in particle_set])
    push!(user_models_not_modeled_assumed_optimistic, [copy_US_with_optimism(us, 1.0) for us in particle_set])
    push!(user_models_not_modeled_assumed_pessimistic, [copy_US_with_optimism(us, -1.0) for us in particle_set])
end

user_models = user_models_modeled
@save "E0_modeled.jld" problem_spec user_models true_user_model
user_models = user_models_not_modeled
@save "E0_not_modeled.jld" problem_spec user_models true_user_model
user_models = user_models_not_modeled_assumed_optimistic
@save "E0_not_modeled_assumed_optimistic.jld" problem_spec user_models true_user_model
user_models = user_models_not_modeled_assumed_pessimistic
@save "E0_not_modeled_assumed_pessimistic.jld" problem_spec user_models true_user_model
