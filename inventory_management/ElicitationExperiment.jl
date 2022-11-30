#
# This file implements the elicitation of a reward function using inverse reinforcement learning for the inventory management problem. Automation is take care of separately by Automate.jl
#

using Pkg
Pkg.activate("../.")

include("LibInventoryManager.jl")

using POMDPs
using MCTS
using Distributions
using JLD2
using CPUTime

import Base.hash

function entropy(x)
    nonzero = x .> 0.0
    return -sum(log2.(x[nonzero]) .* x[nonzero])
end

function hash(a::ProduceProductsAction)::UInt
    idx::UInt = 0
    for (i,q) in enumerate(a.production_amount)
        idx += (q+1) * 100^(i-1)
    end
    return idx
end

################################################################################
#                              DATA GENERATION                                 #
################################################################################

function run_IRL(true_user_model::UserSpecification, user_models::Array{UserSpecification,1}, problem_spec::ProblemSpecification, N = 50)
    um = BoltzmannUserModel(problem_spec, true_user_model)
    m = InventoryPlanningMDP(problem_spec, true_user_model)
    s = initialstate(m)

    um_logprobs = log.(ones(length(user_models)) ./ length(user_models))
    
    state_history = [s]
    reward_history = Float64[]
    posterior_history = zeros(N+1, length(user_models))
    posterior_history[1,:] .= um_logprobs
    
    for i in 1:N
        println("========= $i (human) ==========")
        # update posterior
        aH = act(um, s, missing)
        @time um_logprobs .+= map(θ -> log(likelihood(BoltzmannUserModel(problem_spec, θ), s, aH, missing)), user_models)
        s, r = @gen(:sp, :r)(m, s, aH)

        @show aH, entropy(exp.(um_logprobs) / sum(exp.(um_logprobs)))
        
        push!(reward_history, r)
        push!(state_history, s)
        posterior_history[i+1,:] .= um_logprobs
    end
   
    return state_history, reward_history, posterior_history
end

################################################################################
#                            EXPERIMENT RUN CODE                               #
################################################################################

function main()
    N = 50

    SHOW_HELP = false

    if length(Base.ARGS) == 4
        @load Base.ARGS[3] problem_spec user_models true_user_model
        experiment_spec_name = Base.ARGS[3]
        experiment_index = parse(Int64, Base.ARGS[4])
        user_models = user_models[experiment_index]
        true_user_model = true_user_model[experiment_index]
        problem_spec = problem_spec[experiment_index]

		if Base.ARGS[1] == "IRL"
            state_history, reward_history, posterior_history = run_IRL(true_user_model, user_models, problem_spec, N)
            @save Base.ARGS[2] experiment_spec_name experiment_index state_history reward_history posterior_history
        else
            SHOW_HELP = true
        end
    else
        SHOW_HELP = true
    end

    if SHOW_HELP
        println("Elicitation in AI-assisted design experiment")
        println("Usage:    julia ElicitationExperiment.jl ELICIT_TYPE OUTFILE [SPECFILE SPEC_IDX]")
        println("    ELICIT_TYPE   Can only be IRL, as only IRL is implemented.")
        println("    OUTFILE       Specifies the name of the output file. All the results will be written to this file.")
        println("    SPECFILE      Specifies the name of the experiment specification file to be used.")
        println("    SPEC_IDX      Specifies the index of the experiment scenario within SPECFILE to run.")
        exit(1)
    end
end

main()
