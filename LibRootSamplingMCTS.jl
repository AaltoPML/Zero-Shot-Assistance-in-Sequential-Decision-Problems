module RootSamplingMCTS

using POMDPs
using POMDPSimulators
using MCTS
using Random
using CPUTime
using ProgressMeter
using POMDPModelTools

import MCTS.build_tree, MCTS.insert_node!
import MCTS.MCTSTree
import MCTS: clear_tree!, simulate
import Random.rand!
using MCTS: DPWTree, convert_estimator, insert_state_node!, insert_action_node!

export RandomMDP, rand!, update_weights!, RootSamplingMCTSPlanner, RootSamplingDPWPlanner

# wrapper around a class of MDPs. The idea is that every time your code calls rand!(...) a new underlying MDP will be sampled from a class of MDPs which you have implemented as part of your own RandomMDP implementation. All POMDPs.jl-related calls on a RandomMDP are passed on to the underlying MDP.
abstract type RandomMDP{S, A} <: MDP{S, A}
    # underlying_MDP <: MDP{S,A}
end

function rand!(::RandomMDP) end
function update_weights!(::RandomMDP, ::Array{Float64,1}) end

# pass function calls on RandomMDP to underlying_MDP (implemented for at least those functions which MCTS will use, you can implement more yourself)
POMDPs.actions(m::RandomMDP, s) = POMDPs.actions(m.underlying_MDP, s)
POMDPs.gen(m::RandomMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG) = POMDPs.gen(m.underlying_MDP, s, a, rng)
POMDPs.discount(m::RandomMDP) = POMDPs.discount(m.underlying_MDP)
POMDPs.initialstate(m::RandomMDP) = POMDPs.initialstate(m.underlying_MDP)
POMDPs.isterminal(m::RandomMDP, s) = POMDPs.isterminal(m.underlying_MDP, s)
POMDPs.reward(m::RandomMDP, s, a, sp) = POMDPs.reward(m.underlying_MDP, s, a, sp)


################################################################################
#                      Root Sampling version of DPW MCTS                       #
################################################################################

mutable struct RootSamplingDPWPlanner{P<:RandomMDP, S, A, SE, NA, RCB, RNG} <: AbstractMCTSPlanner{P}
    solver::DPWSolver
    mdp::P
    tree::Union{Nothing, DPWTree{S,A}}
    solved_estimate::SE
    next_action::NA
    reset_callback::RCB
    rng::RNG
end

function RootSamplingDPWPlanner(solver::DPWSolver, mdp::P) where P<:RandomMDP
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return RootSamplingDPWPlanner{P,
                                  statetype(P),
                                  actiontype(P),
                                  typeof(se),
                                  typeof(solver.next_action),
                                  typeof(solver.reset_callback),
                                  typeof(solver.rng)}(solver,
                                                      mdp,
                                                      nothing,
                                                      se,
                                                      solver.next_action,
                                                      solver.reset_callback,
                                                      solver.rng)
end

POMDPs.solve(solver::DPWSolver, mdp::RandomMDP) = RootSamplingDPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::RootSamplingDPWPlanner)
    p.tree = nothing
end

"""
Construct an MCTSDPW tree and choose the best action.
"""
POMDPs.action(p::RootSamplingDPWPlanner, s) = first(action_info(p, s))

"""
Construct an MCTSDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::RootSamplingDPWPlanner, s; tree_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(p.mdp)
        A = actiontype(p.mdp)
        # to be safe the tree is never kept
        tree = DPWTree{S,A}(p.solver.n_iterations)
        p.tree = tree
        snode = insert_state_node!(tree, s, p.solver.check_repeat_state)

        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_us = CPUtime_us()
        for i = 1:p.solver.n_iterations
            nquery += 1
            rand!(p.mdp)
            simulate(p, snode, p.solver.depth)
            p.solver.show_progress ? next!(progress) : nothing
            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end
        p.reset_callback(p.mdp, s)
        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        best_Q = -Inf
        sanode = 0
        for child in tree.children[snode]
            if tree.q[child] > best_Q
                best_Q = tree.q[child]
                sanode = child
            end
        end
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of MCTSDPW.
"""
function simulate(dpw::RootSamplingDPWPlanner, snode::Int, d::Int)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0
    elseif d == 0
        return estimate_value(dpw.solved_estimate, dpw.mdp, s, d)
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dpw.next_action, dpw.mdp, s, DPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.mdp, s, a)
                insert_action_node!(tree, snode, a, n0,
                                    init_Q(sol.init_Q, dpw.mdp, s, a),
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in actions(dpw.mdp, s)
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end

    best_UCB = -Inf
    sanode = 0
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        c = sol.exploration_constant # for clarity
        if (ltn <= 0 && n == 0) || c == 0.0
            UCB = q
        else
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end

    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    sp_in_sachildren = false
    if (dpw.solver.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
        sp, r = @gen(:sp, :r)(dpw.mdp, s, a, dpw.rng)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            # check if state already exists within sanode's children
            spnode = nothing
            for (node_i, _) in dpw.tree.transitions[sanode]
                if dpw.tree.s_labels[node_i] == sp
                    sp_in_sachildren = true
                    spnode = node_i
                    break
                end
            end
            if !sp_in_sachildren
                spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
                new_node = true
            end
        end

        push!(tree.transitions[sanode], (spnode, r))

        if !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r = rand(dpw.rng, tree.transitions[sanode])
    end

    if new_node
        q = r + discount(dpw.mdp)*estimate_value(dpw.solved_estimate, dpw.mdp, sp, d-1)
    else
        q = r + discount(dpw.mdp)*simulate(dpw, spnode, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end

end # module RootSamplingMCTS
