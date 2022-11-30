# This file contains all the data collection, postprocessing and plotting code needed to generate the plots used in he paper for the inventory management experiment.

using Pkg
Pkg.activate("../.")

include("LibInventoryManager.jl")

using Distributions
using Random
using POMDPs
using MCTS
using JLD2
using Plots

function discounted_reward(rewards::Array{Float64,1}, discounting::Float64 = 1.0)
    r_tot = 0
    for (i,r) in enumerate(rewards)
        r_tot += r * discounting^i
    end
    return r_tot
end

function parameters(user_spec)
    return hcat(user_spec.product_profit, user_spec.inventory_cost, user_spec.lost_business_cost)
end

function posterior_error(posterior_history::Array{Float64,2}, true_user_spec, user_specs)
    E_distance = Float64[]
    for timestep in 1:size(posterior_history,1)
        E_d = posterior_error(posterior_history[timestep,:], true_user_spec, user_specs)
        push!(E_distance, E_d)
    end
    return E_distance
end

function posterior_error(posterior::Array{Float64,1}, true_user_spec, user_specs)
    probs = exp.(posterior)
    probs ./= sum(probs)

    E_d = 0.0
    for (i,um_spec) in enumerate(user_specs)
        d = sqrt(sum((parameters(true_user_spec) .- parameters(um_spec)).^2))
        E_d += d * probs[i]
    end
    return E_d
end

function entropy(x)
    x_p = exp.(x) ./ sum(exp.(x))
    nonzero = x_p .> 0.0
    return -sum(log2.(x_p[nonzero]) .* x_p[nonzero])
end


EXP_IDXS = 1:20


@load "inventory_experiment/E0_modeled.jld" problem_spec true_user_model user_models

AIAD_cumrew = Array{Array{Float64,1},1}()
AIAD_posterior_entropy = Array{Array{Float64,1},1}()
AIAD_posterior_error = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_$(i).jld" reward_history posterior_history
    push!(AIAD_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
    push!(AIAD_posterior_entropy, [entropy(posterior_history[j,:]) for j in 1:size(posterior_history,1)])
    push!(AIAD_posterior_error, posterior_error(posterior_history, true_user_model[i], user_models[i]))
end

AIAD_automate_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_automate_$(i).jld" reward_history
    push!(AIAD_automate_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

AIAD_automate_cumrew_per_interaction = Array{Array{Float64,1},1}()
AIAD_automate_posterior_entropy_per_interaction = Array{Array{Float64,1},1}()
AIAD_automate_posterior_error_per_interaction = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_automate_$(i).jld" reward_history aAI_history posterior_history
    
    cumrew_history = vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)])

    cumrew_history_filtered = Float64[]
    posterior_entropy_filtered = Float64[]
    posterior_error_filtered = Float64[]
    if typeof(aAI_history[1]) == ProduceProductsRecommendation
        push!(cumrew_history_filtered, cumrew_history[1])
        push!(posterior_entropy_filtered, entropy(posterior_history[1,:]))
        push!(posterior_error_filtered, posterior_error(posterior_history[1,:], true_user_model[i], user_models[i]))
    end
    for j in 2:length(aAI_history)-1
        if typeof(aAI_history[j]) == ProduceProductsRecommendation
            push!(cumrew_history_filtered, cumrew_history[j])
            push!(posterior_entropy_filtered, entropy(posterior_history[j,:]))
            push!(posterior_error_filtered, posterior_error(posterior_history[j,:], true_user_model[i], user_models[i]))
        end
    end
    for _ in 1:length(cumrew_history)-length(cumrew_history_filtered)
        push!(cumrew_history_filtered, cumrew_history[end])
        push!(posterior_entropy_filtered, entropy(posterior_history[end,:]))
        push!(posterior_error_filtered, posterior_error(posterior_history[end,:], true_user_model[i], user_models[i]))
    end

    push!(AIAD_automate_cumrew_per_interaction, cumrew_history_filtered)
    push!(AIAD_automate_posterior_entropy_per_interaction, posterior_entropy_filtered)
    push!(AIAD_automate_posterior_error_per_interaction, posterior_error_filtered)
end

automation_only_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_automation_only_$(i).jld" reward_history
    push!(automation_only_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

automation_only_cumrew_per_interaction = Array{Array{Float64,1},1}()
automation_only_posterior_entropy_per_interaction = Array{Array{Float64,1},1}()
automation_only_posterior_error_per_interaction = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_automation_only_$(i).jld" reward_history aAI_history posterior_history
    
    cumrew_history = vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)])

    cumrew_history_filtered = Float64[]
    posterior_entropy_filtered = Float64[]
    posterior_error_filtered = Float64[]
    if typeof(aAI_history[1]) == Nothing
        push!(cumrew_history_filtered, cumrew_history[1])
        push!(posterior_entropy_filtered, entropy(posterior_history[1,:]))
        push!(posterior_error_filtered, posterior_error(posterior_history[1,:], true_user_model[i], user_models[i]))
    end
    for j in 2:length(aAI_history)-1
        if typeof(aAI_history[j]) == Nothing
            push!(cumrew_history_filtered, cumrew_history[j])
            push!(posterior_entropy_filtered, entropy(posterior_history[j,:]))
            push!(posterior_error_filtered, posterior_error(posterior_history[j,:], true_user_model[i], user_models[i]))
        end
    end
    for _ in 1:length(cumrew_history)-length(cumrew_history_filtered)
        push!(cumrew_history_filtered, cumrew_history[end])
        push!(posterior_entropy_filtered, entropy(posterior_history[end,:]))
        push!(posterior_error_filtered, posterior_error(posterior_history[end,:], true_user_model[i], user_models[i]))
    end

    push!(automation_only_cumrew_per_interaction, cumrew_history_filtered)
    push!(automation_only_posterior_entropy_per_interaction, posterior_entropy_filtered)
    push!(automation_only_posterior_error_per_interaction, posterior_error_filtered)
end

IRL_cumrew = Array{Array{Float64,1},1}()
IRL_posterior_entropy = Array{Array{Float64,1},1}()
IRL_posterior_error = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_IRL_$(i).jld" reward_history posterior_history
    push!(IRL_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
    push!(IRL_posterior_entropy, [entropy(posterior_history[j,:]) for j in 1:size(posterior_history,1)])
    push!(IRL_posterior_error, posterior_error(posterior_history, true_user_model[i], user_models[i]))
end

IRL_0_OPT_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_IRL_$(i)_OPT_0.jld" reward_history
    push!(IRL_0_OPT_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

IRL_10_OPT_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_IRL_$(i)_OPT_10.jld" reward_history
    push!(IRL_10_OPT_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

IRL_20_OPT_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_IRL_$(i)_OPT_20.jld" reward_history
    push!(IRL_20_OPT_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

IRL_30_OPT_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_IRL_$(i)_OPT_30.jld" reward_history
    push!(IRL_30_OPT_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

IRL_40_OPT_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_IRL_$(i)_OPT_40.jld" reward_history
    push!(IRL_40_OPT_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

oracle_cumrew = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_ORACLE_$(i).jld" reward_history
    push!(oracle_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
end

@load "inventory_experiment/E0_not_modeled_assumed_pessimistic.jld" problem_spec true_user_model user_models
AIAD_pess_cumrew = Array{Array{Float64,1},1}()
AIAD_pess_posterior_entropy = Array{Array{Float64,1},1}()
AIAD_pess_posterior_error = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_assumed_pessimistic_$(i).jld" reward_history posterior_history
    push!(AIAD_pess_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
    push!(AIAD_pess_posterior_entropy, [entropy(posterior_history[j,:]) for j in 1:size(posterior_history,1)])
    push!(AIAD_pess_posterior_error, posterior_error(posterior_history, true_user_model[i], user_models[i]))
end

@load "inventory_experiment/E0_not_modeled_assumed_optimistic.jld" problem_spec true_user_model user_models
AIAD_opti_cumrew = Array{Array{Float64,1},1}()
AIAD_opti_posterior_entropy = Array{Array{Float64,1},1}()
AIAD_opti_posterior_error = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_assumed_optimistic_$(i).jld" reward_history posterior_history
    push!(AIAD_opti_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
    push!(AIAD_opti_posterior_entropy, [entropy(posterior_history[j,:]) for j in 1:size(posterior_history,1)])
    push!(AIAD_opti_posterior_error, posterior_error(posterior_history, true_user_model[i], user_models[i]))
end

@load "inventory_experiment/E0_not_modeled.jld" problem_spec true_user_model user_models
AIAD_not_modeled_cumrew = Array{Array{Float64,1},1}()
AIAD_not_modeled_posterior_entropy = Array{Array{Float64,1},1}()
AIAD_not_modeled_posterior_error = Array{Array{Float64,1},1}()
for i in EXP_IDXS
    @load "inventory_experiment/E0_AIAD_not_modeled_$(i).jld" reward_history posterior_history
    push!(AIAD_not_modeled_cumrew, vcat(0.0, [discounted_reward(reward_history[1:j], problem_spec[i].discounting) for j in 1:length(reward_history)]))
    push!(AIAD_not_modeled_posterior_entropy, [entropy(posterior_history[j,:]) for j in 1:size(posterior_history,1)])
    push!(AIAD_not_modeled_posterior_error, posterior_error(posterior_history, true_user_model[i], user_models[i]))
end


# save the collected data for easy access
@save "inventory_experiment/experiment_results.jld" AIAD_cumrew AIAD_posterior_entropy AIAD_posterior_error AIAD_automate_cumrew AIAD_automate_cumrew_per_interaction AIAD_automate_posterior_entropy_per_interaction AIAD_automate_posterior_error_per_interaction automation_only_cumrew automation_only_cumrew_per_interaction automation_only_posterior_entropy_per_interaction automation_only_posterior_error_per_interaction IRL_cumrew IRL_posterior_entropy IRL_posterior_error IRL_0_OPT_cumrew IRL_10_OPT_cumrew IRL_20_OPT_cumrew IRL_30_OPT_cumrew IRL_40_OPT_cumrew oracle_cumrew AIAD_not_modeled_cumrew AIAD_not_modeled_posterior_entropy AIAD_not_modeled_posterior_error AIAD_opti_cumrew AIAD_opti_posterior_entropy AIAD_opti_posterior_error AIAD_pess_cumrew AIAD_pess_posterior_entropy AIAD_pess_posterior_error

mean(hcat(AIAD_cumrew...),dims=2)[end], std(hcat(AIAD_cumrew...),dims=2)[end] / sqrt(20)
mean(hcat(AIAD_not_modeled_cumrew...),dims=2)[end], std(hcat(AIAD_not_modeled_cumrew...),dims=2)[end] / sqrt(20)
mean(hcat(AIAD_pess_cumrew...),dims=2)[end], std(hcat(AIAD_pess_cumrew...),dims=2)[end] / sqrt(20)
mean(hcat(AIAD_opti_cumrew...),dims=2)[end], std(hcat(AIAD_opti_cumrew...),dims=2)[end] / sqrt(20)


RUN_LENGTH = length(AIAD_cumrew[1])-1

agg(X, f) = reshape(f(hcat(X...), dims = 2), length(X[1]))

colorstyle = :seaborn_colorblind6
stds = 1.0
alpha = 0.1

p = plot(legend = :bottomright, legend_column = 1, xlabel = "time step", ylabel = "cumulative reward", grid = true, gridalpha = 0.07, ylims = (0.0, Inf), size = (490,350))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_cumrew, mean)[1:end], ribbon = agg(AIAD_cumrew, std)[1:end] ./ sqrt(length(AIAD_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automate_cumrew, mean)[1:end], ribbon = agg(AIAD_automate_cumrew, std)[1:end] ./ sqrt(length(AIAD_automate_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_cumrew, mean)[1:end], ribbon = agg(automation_only_cumrew, std)[1:end] ./ sqrt(length(automation_only_cumrew)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_cumrew, mean)[1:end], ribbon = agg(IRL_cumrew, std)[1:end] ./ sqrt(length(IRL_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "unassisted agent")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_10_OPT_cumrew,mean)[1:end], ribbon = agg(IRL_10_OPT_cumrew,std)[1:end] ./ sqrt(length(IRL_10_OPT_cumrew)) .* stds, fillalpha = alpha, color = color = palette(colorstyle)[3], label = "IRL + automation (best)")
plot!(p, collect(0:RUN_LENGTH), agg(oracle_cumrew,mean)[1:end], ribbon = agg(oracle_cumrew,std)[1:end] ./ sqrt(length(oracle_cumrew)) .* stds, fillalpha = alpha, color = "black", label = "oracle + automation", linestyle = :dash)
savefig(p, "inventory_experiment/inventory_management_cum_rew.pdf")

p = plot(legend = :bottomright, legend_column = 1, xlabel = "time step", ylabel = "cumulative reward", grid = true, gridalpha = 0.07, ylims = (0.0, Inf), size = (490,350))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_cumrew, mean)[1:end], ribbon = agg(AIAD_cumrew, std)[1:end] ./ sqrt(length(AIAD_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_not_modeled_cumrew, mean)[1:end], ribbon = agg(AIAD_not_modeled_cumrew, std)[1:end] ./ sqrt(length(AIAD_not_modeled_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "AIAD, no bias assumed")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_pess_cumrew, mean)[1:end], ribbon = agg(AIAD_pess_cumrew, std)[1:end] ./ sqrt(length(AIAD_pess_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[3], label = "AIAD, pessimism assumed")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_opti_cumrew, mean)[1:end], ribbon = agg(AIAD_opti_cumrew, std)[1:end] ./ sqrt(length(AIAD_opti_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[4], label = "AIAD, optimism assumed")
savefig(p, "inventory_experiment/ablation.pdf")

p = plot(legend = :bottomright, legend_column = 1, xlabel = "number of agent interactions", ylabel = "cumulative reward", grid = true, gridalpha = 0.07, ylims = (0.0, Inf), size = (450,270))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_cumrew, mean), ribbon = agg(AIAD_cumrew, std) ./ sqrt(length(AIAD_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automate_cumrew_per_interaction, mean), ribbon = agg(AIAD_automate_cumrew_per_interaction, std) ./ sqrt(length(AIAD_automate_cumrew_per_interaction)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_cumrew_per_interaction, mean), ribbon = agg(automation_only_cumrew_per_interaction, std) ./ sqrt(length(automation_only_cumrew_per_interaction)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], linestyle = :solid, label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_cumrew, mean), ribbon = agg(IRL_cumrew, std) ./ sqrt(length(IRL_cumrew)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "unassisted agent")
plot!(p, [0, 10, 20, 30, 40], 
     [agg(IRL_0_OPT_cumrew, mean)[end], agg(IRL_10_OPT_cumrew, mean)[end], agg(IRL_20_OPT_cumrew, mean)[end], agg(IRL_30_OPT_cumrew, mean)[end], agg(IRL_40_OPT_cumrew, mean)[end]],
     yerr = [agg(IRL_0_OPT_cumrew, std)[end], agg(IRL_10_OPT_cumrew, std)[end], agg(IRL_20_OPT_cumrew, std)[end], agg(IRL_30_OPT_cumrew, std)[end], agg(IRL_40_OPT_cumrew, std)[end]] ./ sqrt(length(IRL_cumrew)) .* stds,
     fillalpha = alpha, color = palette(colorstyle)[3], label = "IRL + automation", linestyle = :dash)
plot!(p, [40, 50], [agg(IRL_40_OPT_cumrew, mean)[end], agg(IRL_cumrew, mean)[end]], color = palette(colorstyle)[3], label = missing, linestyle = :dash)
savefig(p, "inventory_experiment/inventory_management_cum_rew_per_interaction.pdf")

p = plot(legend = :bottomleft, legend_column = 1, xlabel = "number of agent interactions", ylabel = "posterior entropy", grid = true, gridalpha = 0.07, ylims = (0.0, Inf), size = (490,350))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_posterior_entropy, mean)[1:end], ribbon = agg(AIAD_posterior_entropy, std)[1:end] ./ sqrt(length(AIAD_posterior_entropy)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automate_posterior_entropy_per_interaction, mean)[1:end], ribbon = agg(AIAD_automate_posterior_entropy_per_interaction, std)[1:end] ./ sqrt(length(AIAD_automate_posterior_entropy_per_interaction)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_posterior_entropy_per_interaction, mean)[1:end], ribbon = agg(automation_only_posterior_entropy_per_interaction, std)[1:end] ./ sqrt(length(automation_only_posterior_entropy_per_interaction)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_posterior_entropy, mean)[1:end], ribbon = agg(IRL_posterior_entropy, std)[1:end] ./ sqrt(length(IRL_posterior_entropy)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "IRL")
savefig(p, "inventory_experiment/inventory_management_posterior_entropy_per_interaction.pdf")

p = plot(legend = :bottomleft, legend_column = 1, xlabel = "number of agent interactions", ylabel = "mean inference error in θ", grid = true, gridalpha = 0.07, ylims = (0.0, Inf), size = (490,350))
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_posterior_error, mean)[1:end], ribbon = agg(AIAD_posterior_error, std)[1:end] ./ sqrt(length(AIAD_posterior_error)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], label = "AIAD (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(AIAD_automate_posterior_error_per_interaction, mean)[1:end], ribbon = agg(AIAD_automate_posterior_error_per_interaction, std)[1:end] ./ sqrt(length(AIAD_automate_posterior_error_per_interaction)) .* stds, fillalpha = alpha, color = palette(colorstyle)[1], linestyle = :dash, label = "AIAD + automation (ours)")
plot!(p, collect(0:RUN_LENGTH), agg(automation_only_posterior_error_per_interaction, mean)[1:end], ribbon = agg(automation_only_posterior_error_per_interaction, std)[1:end] ./ sqrt(length(automation_only_posterior_error_per_interaction)) .* stds, fillalpha = 0.12, alpha = 0.8, color = palette(colorstyle)[5], label = "partial automation")
plot!(p, collect(0:RUN_LENGTH), agg(IRL_posterior_error, mean)[1:end], ribbon = agg(IRL_posterior_error, std)[1:end] ./ sqrt(length(IRL_posterior_error)) .* stds, fillalpha = alpha, color = palette(colorstyle)[2], label = "IRL")
savefig(p, "inventory_experiment/inventory_management_posterior_error_per_interaction.pdf")
