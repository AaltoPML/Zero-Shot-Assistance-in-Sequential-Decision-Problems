using POMDPs

struct SearchTree{S,A}
    s_labels::Vector{S} # states corresponding to the state nodes
    sa_children::Vector{Vector{Int}} # entry K holds the SA node children of s_labels[K]

    a_labels::Vector{A} # actions of the action nodes
    depth::Vector{Int} # entry K holds the tree depth of node a_labels[K]
    q::Vector{Float64} # entry K indicates the accrued return starting from a_labels[K] (inclusive)
    path_value::Vector{Float64} # entry K indicates the return accrued up to and including a_labels[K]
    transitions::Vector{Pair{Int,Float64}} # entry K indicates the state a_labels[K] transitions to and the reward for that transition
    a_parent::Vector{Union{Int,Missing}} # entry K indicates the index in a_labels of the action parent of a_labels[K] (the parent of a_labels[K]'s parent). Missing if a_label[K] has no action parents.
    frontier::Vector{Bool} # entry K indicates whether a_labels[K] is part of the frontier

    function SearchTree{S,A}(sz::Int=1000) where {S,A}
        return new(sizehint!(Vector{S}(), sz),
                   sizehint!(Vector{Int}[], sz),
                   sizehint!(A[], sz),
                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(Vector{Pair{Int,Float64}}(), sz),
                   sizehint!(Vector{Union{Int,Missing}}(), sz),
                   sizehint!(Vector{Bool}(), sz))
    end
end

struct BFSSolver{P <: MDP}
    mdp::P
    planning_depth::Int
    n_iterations::Int
end

function find_q_values(solver::BFSSolver, s)
    tree = SearchTree{statetype(solver.mdp), actiontype(solver.mdp)}()
    @assert typeof(s) == statetype(solver.mdp) "state has incorrect type"

    ## initial set-up
    push!(tree.s_labels, s)
    push!(tree.sa_children, Int[])
    s_idx = length(tree.s_labels)

    for a in actions(solver.mdp, s)
        r,sp = @gen(:r,:sp)(solver.mdp, s, a)
        push!(tree.q, r)
        push!(tree.path_value, r)
        push!(tree.a_labels, a)
        push!(tree.depth, 1)
        push!(tree.a_parent, missing)
        a_idx = length(tree.a_labels)
        push!(tree.sa_children[s_idx], a_idx)
        push!(tree.frontier, 1 == solver.planning_depth ? false : true)
        push!(tree.s_labels, sp)
        push!(tree.sa_children, Int[])
        push!(tree.transitions, Pair(length(tree.s_labels),r))
    end

    for _ in 1:solver.n_iterations
        if !any(tree.frontier)
            # frontier empty, stop planning
            break
        end
        a_idx = argmax(tree.q .- (.!tree.frontier .* Inf))
        @inbounds tree.frontier[a_idx] = false

        @inbounds s_idx, _ = tree.transitions[a_idx]
        @inbounds s = tree.s_labels[s_idx]
        @inbounds a_depth = tree.depth[a_idx]

        best_r = -Inf
        for a in actions(solver.mdp, s)
            r,sp = @gen(:r,:sp)(solver.mdp, s, a)
            push!(tree.q, r)
            push!(tree.path_value, @inbounds r * discount(solver.mdp)^(a_depth) + tree.path_value[a_idx])
            push!(tree.a_labels, a)
            push!(tree.depth, a_depth + 1)
            push!(tree.a_parent, a_idx)
            push!(tree.sa_children[s_idx], length(tree.a_labels))
            push!(tree.frontier, (a_depth + 1) == solver.planning_depth ? false : true)
            push!(tree.s_labels, sp)
            push!(tree.sa_children, Int[])
            push!(tree.transitions, Pair(length(tree.s_labels),r))

            best_r = r > best_r ? r : best_r
        end

        path_to_root = [a_idx]
        while !ismissing(tree.a_parent[path_to_root[end]])
            push!(path_to_root, tree.a_parent[path_to_root[end]])
        end
        # set the q-value at the top of the tree (we don't need them further down)
        tree.q[path_to_root[end]] = best_r * discount(solver.mdp)^(tree.depth[a_idx]) + tree.path_value[a_idx]
    end

    q_values = Dict{actiontype(solver.mdp), Float64}()
    for a_idx in tree.sa_children[1]
        q_values[tree.a_labels[a_idx]] = tree.q[a_idx]
    end
    return q_values
end
