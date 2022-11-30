# This file implements the main loop for running preference learning and inverse reinforcement learning. It only implements the reward learning part of the IRL + automation and PL + automation baselines. OptimizeTrip.jl handles the automation part.
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
using Distributions
using JLD2
using CPUTime

function entropy(x)
    nonzero = x .> 0.0
    return -sum(log2.(x[nonzero]) .* x[nonzero])
end

################################################################################
#                              DATA GENERATION                                 #
###############################################################################

function find_max_EIG_pair(ums, um_probs::Array{Float64,1}, home::POI, POIs::Array{POI,1}; n_iters::Int64 = 10000, max_time_s::Float64 = 600.0)
    trips = Array{Array{POI,1},1}()
    trips_utilities = Array{Array{Float64,1},1}()

    best_pair = missing
    max_EIG = -Inf
    start_time_us = CPUtime_us()
    for iii in 1:n_iters
        if CPUtime_us() - start_time_us >= max_time_s * 1e6
            @warn "stopped early at $iii"
            break
        end
        res = [home]
        for p in POIs
            if rand() < 0.125
                push!(res, p)
            end
        end
        trip = optimize_trip_cw(res)

        if duration(trip) > DaytripDesignAssistance.MAX_DAY_LENGTH
            continue
        end

        trip_utilities = Array{Float64,1}()
        util = missing
        for (i,um_spec) in enumerate(ums)
            um = instantiate(um_spec, home, [home])
            util = utility(um.world_model, trip)
            push!(trip_utilities, util)
        end

        for (last_trip, last_trip_utilties) in zip(trips, trips_utilities)
            p_left_ums = Array{Float64,1}()
            p_right_ums = Array{Float64,1}()
            for (i,um_spec) in enumerate(ums)
                # Account for the difference in scale between q-values (for which comparison_optimality is usually used) and complete designs.
                τ = um_spec.second.comparison_optimality * 5.0
                right = exp(τ * util) / (exp(τ * util) + exp(τ * last_trip_utilties[i]))
                left = 1.0 - right
                push!(p_left_ums, left)
                push!(p_right_ums, right)
            end

            denom_left = sum(p_left_ums .* um_probs)
            denom_right = sum(p_right_ums .* um_probs)

            EIG = sum(p_left_ums .* um_probs .* (log2.(p_left_ums) .- log2(denom_left))) + sum(p_right_ums .* um_probs .* (log2.(p_right_ums) .- log2(denom_right)))

            if EIG > max_EIG
                best_pair = (last_trip, trip)
                max_EIG = EIG
            end
        end

        push!(trips, trip)
        push!(trips_utilities, trip_utilities)
    end
    println("best pair found had expected information gain of $max_EIG")
    return best_pair
end

function run_preference_learning(um, user_models, N = 50; max_time_s = 5.0*60.0)
    home = um.world_model.home
    POIs = um.world_model.POIs
    
    um_probs = ones(length(user_models)) ./ length(user_models)
    
    posterior_history = zeros(N+1, length(user_models))
    posterior_history[1,:] .= log.(um_probs)
    for i in 1:N
        println("========= $i (elicitation) ==========")
        question_pair = optimize_trip_cw.(find_max_EIG_pair(user_models, um_probs, home, POIs; max_time_s = max_time_s))
        
        # simuate user choice
        τ = min(um.comparison_optimality, 50.0) * 4.0
        util_left = util = utility(um.world_model, question_pair[1])
        util_right = util = utility(um.world_model, question_pair[2])
        chose_right = rand() < exp(τ * util_right) / (exp(τ * util_right) + exp(τ * util_left))

        # update posterior
        prob_updates = []
        @time for um_spec in user_models
            um_i = instantiate(um_spec, home, [home])
            util_left = utility(um_i.world_model, question_pair[1])
            util_right = utility(um_i.world_model, question_pair[2])
            τ = um_spec.second.comparison_optimality * 5.0
            if chose_right
                push!(prob_updates, exp(τ * util_right) / (exp(τ * util_right) + exp(τ * util_left)))
            else
                push!(prob_updates, exp(τ * util_left) / (exp(τ * util_right) + exp(τ * util_left)))

            end
        end
        um_probs = um_probs .* prob_updates ./ sum(um_probs .* prob_updates);
        
        @show entropy(um_probs)
        
        posterior_history[i+1,:] .= log.(um_probs)
    end
   
    return posterior_history
end

function run_IRL(um, user_models, N = 50)
    home = um.world_model.home
    POIs = um.world_model.POIs
    objective_world_model = MachineDaytripScheduleMDP(um.world_model, 1.0)

    s = initialstate(objective_world_model).val
    um_logprobs = log.(ones(length(user_models)) ./ length(user_models))
    
    state_history = [s]
    utility_history = [utility(objective_world_model, s)]
    posterior_history = zeros(N+1, length(user_models))
    posterior_history[1,:] .= um_logprobs
    
    s = initialstate(objective_world_model).val
    for i in 1:N
        println("========= $i (human) ==========")
        # update posterior
        a = act(um, s, missing)
        @time um_logprobs .+= map(θ -> log(likelihood(instantiate(θ, home, POIs), s, a, missing)), user_models)
        s = @gen(:sp)(objective_world_model, s, a)

        @show a, entropy(exp.(um_logprobs) / sum(exp.(um_logprobs))), duration(s), utility(objective_world_model, s)
        
        push!(utility_history, utility(objective_world_model, s))
        push!(state_history, s)
        posterior_history[i+1,:] .= um_logprobs
    end
   
    return state_history, utility_history, posterior_history
end


################################################################################
#                            EXPERIMENT RUN CODE                               #
################################################################################

function main()
    N = 30

    # specific to preference learning
    planning_time = 5.0*60.0

    SHOW_HELP = false

    if length(Base.ARGS) == 4
        @load Base.ARGS[3] user_models true_user_model POIs home
        experiment_spec_name = Base.ARGS[3]
        experiment_index = parse(Int64, Base.ARGS[4])
        user_models = user_models[experiment_index]
        POIs = POIs[experiment_index]
        home = home[experiment_index]
        true_user_model = instantiate(true_user_model[experiment_index], home, POIs)

        if Base.ARGS[1] == "PREFERENCE"
            posterior_history = run_preference_learning(true_user_model, user_models, N, max_time_s = planning_time)
            state_history = missing
            utility_history = missing
            @save Base.ARGS[2] experiment_spec_name experiment_index state_history utility_history posterior_history
        elseif Base.ARGS[1] == "IRL"
            state_history, utility_history, posterior_history = run_IRL(true_user_model, user_models, N)
            @save Base.ARGS[2] experiment_spec_name experiment_index state_history utility_history posterior_history
        else
            SHOW_HELP = true
        end
    else
        SHOW_HELP = true
    end

    if SHOW_HELP
        println("Elicitation in AI-assisted design experiment")
        println("Usage:    julia ElicitationExperiment.jl ELICIT_TYPE OUTFILE [SPECFILE SPEC_IDX]")
        println("    ELICIT_TYPE   One of either IRL or PREFERENCE, depending on whether you want to run IRL or preference learning respectively.")
        println("    OUTFILE       Specifies the name of the output file. All the results will be written to this file.")
        println("    SPECFILE      Specifies the name of the experiment specification file to be used.")
        println("    SPEC_IDX      Specifies the index of the experiment scenario within SPECFILE to run.")
        exit(1)
    end
end

main()
