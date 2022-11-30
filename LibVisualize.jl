using Plots
using Printf

"""
Plot the user's policy.

Condensed means that the zero probability actions have been removed.
"""
function plot_policy(um, s, a; condensed = true)
    a, p = get_policy(um, s, a)
    
    nonzero = p .> (condensed ? 0.0 : -Inf)
    plt = bar(string.(a[nonzero]), p[nonzero], label = missing, ylabel = "p(action)", bins = 1:length(a[nonzero]), normalize = false, orientation = :v, xrotation=60, xticks = (0.5:1:length(nonzero), string.(a[nonzero])))
    return plt
end

function plot_state!(p::Plots.Plot, s::Daytrip, POIs::Array{POI,1} = POI[])
    locations = map(x -> (x.coord_x, x.coord_y), POIs)
    categories = map(x -> x.category, POIs)
    
    plot!(p, map(x -> (x.coord_x, x.coord_y), vcat(s, s[1])), label=missing, color = "black")
    
    # make sure Daytrip POIs are plotted
    for p in s
        if !(p in POIs)
            push!(locations, (p.coord_x, p.coord_y))
            push!(categories, p.category)
        end
    end
    
    for c in instances(POICategory)
        scatter!(p, locations[categories .== c], lab = lowercase(string(c)))
    end
    p
end

function plot_state(s::Daytrip, POIs::Array{POI,1} = POI[])
    p = Plots.plot(palette = :seaborn_colorblind6)
    return plot_state!(p, s, POIs)
end

function animate_state_history(sh::Array{Daytrip,1}, POIs::Array{POI,1} = POI[]; fps::Int64 = 1, filename::String = "tmp.gif")
    anim = @animate for (i, s) in enumerate(sh)
        p = Plots.plot(palette = :seaborn_colorblind6, title = """It $i -- $(@sprintf "%.0f" duration(s)) min ($(@sprintf "%.0f" travel_time(s)) walking) -- $(@sprintf "%.2f" cost(s)) euro""")
        plot_state!(p, s, POIs)
        p
    end
    return gif(anim, "anim.gif", fps = 1)
end

function plot_multiple!(p, data::Array{Float64,2}, color, label; individual = false, alpha = 0.2, stderr = true, stds = 2)
    N = size(data)[2]
    if individual
        for i in 1:N
            plot!(p, data[:,i]; label=missing, linestyle = :dot, color=color, alpha = alpha)
        end
        mean_data = mean(data,dims=2)
        plot!(p, mean_data; label=label, linestyle = :solid, color=color)
    else
        mean_data = mean(data,dims=2)
        std_data = std(data,dims=2)
        if stderr
            std_data ./= sqrt(N)
        end
        plot!(p, mean_data; ribbon = std_data*2, label=label, linestyle = :solid, color=color, fillalpha = alpha)
    end
    return p
end

function plot_multiple!(p, f::Function, N::Int64, color, label; individual = false, alpha = 0.2, stderr = true, stds = 2)
    data = hcat([f() for _ in 1:N]...)
    
    return plot_multiple!(p, data, color, label; individual = individual, alpha = alpha, stderr = stderr, stds = stds)
end
