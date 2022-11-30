# This file collects the data from all the runs of the day trip design experiment with unbiased agents, post-processes it, and generates the plots used in the paper.
#
#

using Pkg
Pkg.activate("../.")

include("../LibTravelPlanner.jl")
using .DaytripDesignAssistance

using JLD2
using FileIO
using Statistics
using HypothesisTests
using ProgressMeter
using POMDPs
using Plots

function entropy2(x)
    x_p = exp.(x) ./ sum(exp.(x))
    nonzero = x_p .> 0.0
    return -sum(log2.(x_p[nonzero]) .* x_p[nonzero])
end

DATADIR = "basic_experiment"
EXP_NAME = "E0"
EXP_IDXS = 1:50

AIAD_utilities = Array{Array{Float64,1},1}()
AIAD_posterior_entropies = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_AIAD_$(i).jld" utility_history posterior_history
    posterior_entropy_history = [entropy2(posterior_history[i,:]) for i in 1:size(posterior_history,1)]

    push!(AIAD_utilities, utility_history)
    push!(AIAD_posterior_entropies, posterior_entropy_history)
end

AIAD_automation_utilities = Array{Array{Float64,1},1}()
AIAD_automation_tot_num_interactions = Array{Float64,1}()
AIAD_automation_posterior_entropies = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_AIAD_automation_$(i).jld" utility_history posterior_history aAI_history
    posterior_entropy_history = [entropy2(posterior_history[i,:]) for i in 1:size(posterior_history,1)]
    
    utility_history_filtered = Float64[]
    posterior_history_filtered = Float64[posterior_entropy_history[1]]
    if typeof(aAI_history[1]) == EditRecommendation
        push!(utility_history_filtered, utility_history[1])
    end
    for j in 1:length(aAI_history)-1
        if typeof(aAI_history[j+1]) == EditRecommendation
            push!(utility_history_filtered, utility_history[j])
        end
    end
    for _ in 1:length(utility_history)-length(utility_history_filtered)
        push!(utility_history_filtered, utility_history[end])
    end
    
    for j in 1:length(aAI_history)-1
        if typeof(aAI_history[j]) == EditRecommendation
            push!(posterior_history_filtered, posterior_entropy_history[j+1])
        end
    end
    for _ in 1:length(posterior_entropy_history)-length(posterior_history_filtered)
        push!(posterior_history_filtered, posterior_entropy_history[end])
    end

    push!(AIAD_automation_utilities, utility_history_filtered[1:31])
    push!(AIAD_automation_posterior_entropies, posterior_history_filtered[1:31])
    push!(AIAD_automation_tot_num_interactions, sum(typeof.(aAI_history[1:60]) .== EditRecommendation))
end

automation_only_utilities = Array{Array{Float64,1},1}()
automation_only_tot_num_interactions = Array{Float64,1}()
automation_only_posterior_entropies = Array{Array{Float64,1},1}()

for i in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_automation_only_$(i).jld" utility_history posterior_history aAI_history
    posterior_entropy_history = [entropy2(posterior_history[i,:]) for i in 1:size(posterior_history,1)]
    
    utility_history_filtered = Float64[]
    posterior_history_filtered = Float64[posterior_entropy_history[1]]
    if typeof(aAI_history[1]) == EditRecommendation
        push!(utility_history_filtered, utility_history[1])
    end
    for j in 1:length(aAI_history)-1
        if typeof(aAI_history[j+1]) == EditRecommendation
            push!(utility_history_filtered, utility_history[j])
        end
    end
    for _ in 1:length(utility_history)-length(utility_history_filtered)
        push!(utility_history_filtered, utility_history[end])
    end
    
    for j in 1:length(aAI_history)-1
        if typeof(aAI_history[j]) == EditRecommendation
            push!(posterior_history_filtered, posterior_entropy_history[j+1])
        end
    end
    for _ in 1:length(posterior_entropy_history)-length(posterior_history_filtered)
        push!(posterior_history_filtered, posterior_entropy_history[end])
    end

    push!(automation_only_utilities, utility_history_filtered[1:31])
    push!(automation_only_posterior_entropies, posterior_history_filtered[1:31])
    push!(automation_only_tot_num_interactions, sum(typeof.(aAI_history[1:60]) .== EditRecommendation))
end

IRL_utilities = Array{Array{Float64,1},1}()
IRL_posterior_entropies = Array{Array{Float64,1},1}()
IRL_optimized_0_restart = Array{Float64,1}()
IRL_optimized_5_restart = Array{Float64,1}()
IRL_optimized_10_restart = Array{Float64,1}()
IRL_optimized_15_restart = Array{Float64,1}()
IRL_optimized_20_restart = Array{Float64,1}()
IRL_optimized_25_restart = Array{Float64,1}()
IRL_optimized_30_restart = Array{Float64,1}()
for i in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_0_RESTART.jld" utility_history
    optimized_utility_0_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_5_RESTART.jld" utility_history
    optimized_utility_5_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_10_RESTART.jld" utility_history
    optimized_utility_10_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_15_RESTART.jld" utility_history
    optimized_utility_15_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_20_RESTART.jld" utility_history
    optimized_utility_20_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_25_RESTART.jld" utility_history
    optimized_utility_25_restart = utility_history[end]

    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i)_OPT_30_RESTART.jld" utility_history
    optimized_utility_30_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(i).jld" utility_history posterior_history
    posterior_entropy_history = [entropy2(posterior_history[i,:]) for i in 1:size(posterior_history,1)]
    
    push!(IRL_utilities, utility_history)
    push!(IRL_posterior_entropies, posterior_entropy_history)
    push!(IRL_optimized_0_restart, optimized_utility_0_restart)
    push!(IRL_optimized_5_restart, optimized_utility_5_restart)
    push!(IRL_optimized_10_restart, optimized_utility_10_restart)
    push!(IRL_optimized_15_restart, optimized_utility_15_restart)
    push!(IRL_optimized_20_restart, optimized_utility_20_restart)
    push!(IRL_optimized_25_restart, optimized_utility_25_restart)
    push!(IRL_optimized_30_restart, optimized_utility_30_restart)
end

PL_posterior_entropies = Array{Array{Float64,1},1}()
PL_optimized_5_restart = Array{Float64,1}()
PL_optimized_10_restart = Array{Float64,1}()
PL_optimized_15_restart = Array{Float64,1}()
PL_optimized_20_restart = Array{Float64,1}()
PL_optimized_25_restart = Array{Float64,1}()
PL_optimized_30_restart = Array{Float64,1}()
for i in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i)_OPT_5_RESTART.jld" utility_history
    optimized_utility_5_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i)_OPT_10_RESTART.jld" utility_history
    optimized_utility_10_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i)_OPT_15_RESTART.jld" utility_history
    optimized_utility_15_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i)_OPT_20_RESTART.jld" utility_history
    optimized_utility_20_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i)_OPT_25_RESTART.jld" utility_history
    optimized_utility_25_restart = utility_history[end]

    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i)_OPT_30_RESTART.jld" utility_history
    optimized_utility_30_restart = utility_history[end]
    
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(i).jld" posterior_history
    posterior_entropy_history = [entropy2(posterior_history[i,:]) for i in 1:size(posterior_history,1)]
    
    push!(PL_posterior_entropies, posterior_entropy_history)
    push!(PL_optimized_5_restart, optimized_utility_5_restart)
    push!(PL_optimized_10_restart, optimized_utility_10_restart)
    push!(PL_optimized_15_restart, optimized_utility_15_restart)
    push!(PL_optimized_20_restart, optimized_utility_20_restart)
    push!(PL_optimized_25_restart, optimized_utility_25_restart)
    push!(PL_optimized_30_restart, optimized_utility_30_restart)
end

TRUE_optimized = []
for i in EXP_IDXS
    try
        @load "$(DATADIR)/$(EXP_NAME)_TRUE_$(i)_OPT.jld" utility_history
        push!(TRUE_optimized, utility_history[end])
    catch e
        push!(TRUE_optimized, missing)
    end  
end

@load "$(DATADIR)/$(EXP_NAME)_basic_experiment.jld" user_models true_user_model
IRL_expected_reward_error = []
for exp_IDX in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_IRL_$(exp_IDX).jld" state_history utility_history posterior_history
    
    E_distance = Float64[]
    for timestep in 1:length(state_history)
        probs = exp.(posterior_history[timestep,:])
        probs ./= sum(probs)

        E_d = 0.0
        for (i,um_spec) in enumerate(user_models[exp_IDX])
            d = sqrt(sum((true_user_model[exp_IDX].first.topic_interests .- um_spec.first.topic_interests).^2))
            E_d += d * probs[i]
        end
        push!(E_distance, E_d)
    end
    push!(IRL_expected_reward_error, E_distance)
end

AIAD_expected_reward_error = []
for exp_IDX in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_AIAD_$(exp_IDX).jld" state_history utility_history posterior_history

    E_distance = Float64[]
    for timestep in 1:length(state_history)
        probs = exp.(posterior_history[timestep,:])
        probs ./= sum(probs)

        E_d = 0.0
        for (i,um_spec) in enumerate(user_models[exp_IDX])
            d = sqrt(sum((true_user_model[exp_IDX].first.topic_interests .- um_spec.first.topic_interests).^2))
            E_d += d * probs[i]
        end
        push!(E_distance, E_d)
    end
    push!(AIAD_expected_reward_error, E_distance)
end

AIAD_automation_expected_reward_error = []
for exp_IDX in EXP_IDXS 
    @load "$(DATADIR)/$(EXP_NAME)_AIAD_automation_$(exp_IDX).jld" state_history utility_history posterior_history aAI_history

    posterior_history_filtered = zeros(61, size(posterior_history,2))
    posterior_history_filtered[1,:] .= posterior_history[1,:]
    i = 2
    for j in 1:length(aAI_history)-1
        if typeof(aAI_history[j]) == EditRecommendation
            posterior_history_filtered[i,:] .= posterior_history[j+1,:]
            i += 1
        end
    end
    for _ in 1:(61+1-i)
        posterior_history_filtered[i,:] .= posterior_history[end,:]
        i += 1
    end
    
    min_d = 0

    E_distance = Float64[]
    for timestep in 1:31
        probs = exp.(posterior_history_filtered[timestep,:])
        probs ./= sum(probs)

        E_d = 0.0
        for (i,um_spec) in enumerate(user_models[exp_IDX])
            d = sqrt(sum((true_user_model[exp_IDX].first.topic_interests .- um_spec.first.topic_interests).^2))
            E_d += (d - min_d) * probs[i]
        end
        push!(E_distance, E_d)
    end
    push!(AIAD_automation_expected_reward_error, E_distance)
end

PL_expected_reward_error = []
for exp_IDX in EXP_IDXS
    @load "$(DATADIR)/$(EXP_NAME)_PL_$(exp_IDX).jld" utility_history posterior_history

    E_distance = Float64[]
    for timestep in 1:31
        probs = exp.(posterior_history[timestep,:])
        probs ./= sum(probs)

        E_d = 0.0
        for (i,um_spec) in enumerate(user_models[exp_IDX])
            d = sqrt(sum((true_user_model[exp_IDX].first.topic_interests .- um_spec.first.topic_interests).^2))
            E_d += d * probs[i]
        end
        push!(E_distance, E_d)
    end
    push!(PL_expected_reward_error, E_distance)
end


#
# save data
#

@save "experiment_data_basic_experiment.jld" TRUE_optimized AIAD_utilities AIAD_posterior_entropies AIAD_expected_reward_error AIAD_automation_utilities AIAD_automation_posterior_entropies AIAD_automation_expected_reward_error AIAD_automation_tot_num_interactions IRL_utilities IRL_posterior_entropies IRL_expected_reward_error IRL_optimized_0_restart IRL_optimized_5_restart IRL_optimized_10_restart IRL_optimized_15_restart IRL_optimized_20_restart IRL_optimized_25_restart IRL_optimized_30_restart PL_posterior_entropies PL_expected_reward_error PL_optimized_5_restart PL_optimized_10_restart PL_optimized_15_restart PL_optimized_20_restart PL_optimized_25_restart PL_optimized_30_restart

RUN_LENGTH = length(AIAD_utilities[1])-1

agg(X, f) = reshape(f(hcat(X...), dims = 2), length(X[1]))
agg(X, f, n) = reshape(f(hcat(X...)[1:n+1,:], dims = 2), n+1)

#
# objective value plot
#

colorstyle = :seaborn_colorblind6
stds = 1.0
alpha = 0.1

p = plot(legend = :bottomright, legend_column = 1, xlabel = "number of agent interactions", ylabel = "objective value", grid = true, gridalpha = 0.07, ylims = (0.0, Inf), size = (450,270))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_utilities, mean, RUN_LENGTH), ribbon = agg(AIAD_utilities, std, RUN_LENGTH) ./ sqrt(length(AIAD_utilities)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automation_utilities, mean, RUN_LENGTH), ribbon = agg(AIAD_automation_utilities, std, RUN_LENGTH) ./ sqrt(length(AIAD_automation_utilities)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_utilities, mean, RUN_LENGTH), ribbon = agg(automation_only_utilities, std, RUN_LENGTH) ./ sqrt(length(automation_only_utilities)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], linestyle = :solid, label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_utilities, mean, RUN_LENGTH), ribbon = agg(IRL_utilities, std, RUN_LENGTH) ./ sqrt(length(IRL_utilities)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "unassisted agent")
plot!(p, [0, 5, 10, 15, 20, 25, 30], 
     [mean(IRL_optimized_0_restart), mean(IRL_optimized_5_restart), mean(IRL_optimized_10_restart), mean(IRL_optimized_15_restart), mean(IRL_optimized_20_restart), mean(IRL_optimized_25_restart), mean(IRL_optimized_30_restart)],
     yerr = [std(IRL_optimized_0_restart), std(IRL_optimized_5_restart), std(IRL_optimized_10_restart), std(IRL_optimized_15_restart), std(IRL_optimized_20_restart), std(IRL_optimized_25_restart), std(IRL_optimized_30_restart)] ./ sqrt(length(IRL_utilities)) .* stds,
     fillalpha = alpha, color = palette(colorstyle)[3], label = "IRL + automation", linestyle = :dash)
plot!(p, [0, 5, 10, 15, 20, 25, 30], 
     [mean(IRL_optimized_0_restart), mean(PL_optimized_5_restart), mean(PL_optimized_10_restart), mean(PL_optimized_15_restart), mean(PL_optimized_20_restart), mean(PL_optimized_25_restart), mean(PL_optimized_30_restart)],
     yerr = [std(IRL_optimized_0_restart), std(PL_optimized_5_restart), std(PL_optimized_10_restart), std(PL_optimized_15_restart), std(PL_optimized_20_restart), std(PL_optimized_25_restart), std(PL_optimized_30_restart)] ./ sqrt(length(IRL_utilities)) .* stds,
     fillalpha = alpha, color = palette(colorstyle)[4], label = "PL + automation", linestyle = :dash)
savefig("trip_design_basic_experiment_utility_per_interaction.pdf")

#
# posterior entropy plot
#

p = plot(legend = :topright, xlabel = "number of agent interactions", ylabel = "posterior entropy", grid = true, gridalpha = 0.07, ylims = (0.0, 12.5))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_posterior_entropies, mean, RUN_LENGTH), ribbon = agg(AIAD_posterior_entropies, std, RUN_LENGTH) ./ sqrt(length(AIAD_posterior_entropies)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automation_posterior_entropies, mean, RUN_LENGTH), ribbon = agg(AIAD_automation_posterior_entropies, std, RUN_LENGTH) ./ sqrt(length(AIAD_automation_posterior_entropies)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_posterior_entropies, mean, RUN_LENGTH), ribbon = agg(automation_only_posterior_entropies, std, RUN_LENGTH) ./ sqrt(length(automation_only_posterior_entropies)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], linestyle = :solid, label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_posterior_entropies, mean, RUN_LENGTH), ribbon = agg(IRL_posterior_entropies, std, RUN_LENGTH) ./ sqrt(length(IRL_posterior_entropies)) .* stds, fillalpha = alpha, color = palette(colorstyle)[3], label = "IRL")
plot!(p, collect(0:RUN_LENGTH), agg(PL_posterior_entropies, mean, RUN_LENGTH), ribbon = agg(PL_posterior_entropies, std, RUN_LENGTH) ./ sqrt(length(PL_posterior_entropies)) .* stds, fillalpha = alpha, color = palette(colorstyle)[4], label = "PL")
savefig(p, "trip_design_basic_experiment_posterior_entropy_per_interaction.pdf")

#
# inference error plot
#

colorstyle = :seaborn_colorblind6
stds = 1.0
alpha = 0.1

p = plot(legend = :topright, xlabel = "number of agent interactions", ylabel = "mean inference error in ω", grid = true, gridalpha = 0.07, ylims = (-Inf, 3.05))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_expected_reward_error, mean, RUN_LENGTH), ribbon = agg(AIAD_expected_reward_error, std, RUN_LENGTH) ./ sqrt(length(AIAD_expected_reward_error)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automation_expected_reward_error, mean, RUN_LENGTH), ribbon = agg(AIAD_automation_expected_reward_error, std, RUN_LENGTH) ./ sqrt(length(AIAD_automation_expected_reward_error)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_expected_reward_error, mean, RUN_LENGTH), ribbon = agg(automation_only_expected_reward_error, std, RUN_LENGTH) ./ sqrt(length(automation_only_expected_reward_error)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], linestyle = :solid, label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_expected_reward_error, mean, RUN_LENGTH), ribbon = agg(IRL_expected_reward_error, std, RUN_LENGTH) ./ sqrt(length(IRL_expected_reward_error)) .* stds, fillalpha = alpha, color = palette(colorstyle)[3], label = "IRL")
plot!(p, collect(0:RUN_LENGTH), agg(PL_expected_reward_error, mean, RUN_LENGTH), ribbon = agg(PL_expected_reward_error, std, RUN_LENGTH) ./ sqrt(length(PL_expected_reward_error)) .* stds, fillalpha = alpha, color = palette(colorstyle)[4], label = "PL")
savefig(p, "trip_design_basic_experiment_posterior_loss_per_interaction.pdf")

#
# cumulative reward plot
#

colorstyle = :seaborn_colorblind6
stds = 1.0
alpha = 0.1

p = plot(legend = :bottomright, legend_column = 1, xlabel = "number of actions", ylabel = "cumulative reward", grid = true, gridalpha = 0.07, ylims = (0.0, Inf))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_utilities, mean, RUN_LENGTH), ribbon = agg(AIAD_utilities, std, RUN_LENGTH) ./ sqrt(length(AIAD_utilities)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automation_utilities_unfiltered_trace, mean, RUN_LENGTH), ribbon = agg(AIAD_automation_utilities_unfiltered_trace, std, RUN_LENGTH) ./ sqrt(length(AIAD_automation_utilities_unfiltered_trace)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_utilities_unfiltered_trace, mean, RUN_LENGTH), ribbon = agg(automation_only_utilities_unfiltered_trace, std, RUN_LENGTH) ./ sqrt(length(automation_only_utilities_unfiltered_trace)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], linestyle = :solid, label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_utilities, mean, RUN_LENGTH), ribbon = agg(IRL_utilities, std, RUN_LENGTH) ./ sqrt(length(IRL_utilities)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "unassisted agent")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_optimized_30_restart_unfiltered_trace, mean, RUN_LENGTH), ribbon = agg(IRL_optimized_30_restart_unfiltered_trace, std, RUN_LENGTH) ./ sqrt(length(IRL_optimized_30_restart_unfiltered_trace)) .* stds, fillalpha = alpha, color = palette(colorstyle)[3], label = "automation (IRL)", linestyle = :solid)
plot!(p, collect(0:RUN_LENGTH), agg(PL_optimized_30_restart_unfiltered_trace, mean, RUN_LENGTH), ribbon = agg(PL_optimized_30_restart_unfiltered_trace, std, RUN_LENGTH) ./ sqrt(length(PL_optimized_30_restart_unfiltered_trace)) .* stds, fillalpha = alpha, color = palette(colorstyle)[4], label = "automation (PL)", linestyle = :solid)
plot!(p, collect(0:RUN_LENGTH), agg(TRUE_optimized_unfiltered_trace, mean, RUN_LENGTH), ribbon = agg(TRUE_optimized_unfiltered_trace, std, RUN_LENGTH) ./ sqrt(length(TRUE_optimized_unfiltered_trace)) .* stds, fillalpha = alpha, color = "black", label = "automation (oracle)", linestyle = :dash)
savefig(p, "trip_design_basic_experiment_cumulative_reward.pdf")