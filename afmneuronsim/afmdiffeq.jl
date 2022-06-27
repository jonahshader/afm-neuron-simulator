# include("graph/afmgraph_methods.jl")
# include("afmcomponent.jl")
# include("utils.jl")

using LinearAlgebra
using SparseArrays
using DifferentialEquations
using RecursiveArrayTools
using Plots


# export root, set_root!
# export nnm
# export inm
# export nom
# export iom
# export tspan, set_tspan!
# export u0, set_u0!
# export input_functions
# export ode_problem
# export sol

export solve_parts!
export input_to_spikes
export build_model_parts
export rebuild_model_parts!
export input_to_spikes
export peak_output
export input
export output
export output_max
export output_binary
export build_neuron_labels
export plot_Θ
export plot_dΘ
export plot_output
export plot_input
export parameter_mask_view

export root
export reduced_graph
export nnm
export inm
export nom
export iom
export tspan
export u0
export sol



mutable struct AFMModelParts{T<:AbstractFloat}
    # graph::Graph{T}
    # nnm::AbstractMatrix{T}
    # inm::AbstractMatrix{T}
    # nom::AbstractMatrix{T}
    # iom::AbstractMatrix{T}
    # u0::AbstractMatrix{T}
    # tspan::Tuple{T, T}
    # input_functions::Vector{Function}
    # ode_problem::ODEProblem
    # sol::Union{OrdinaryDiffEq.ODECompositeSolution, Nothing}
    root::Component
    reduced_graph::Graph{T}
    nom::AbstractMatrix{T}
    iom::AbstractMatrix{T}
    tspan::Tuple{T, T}
    u0::AbstractMatrix{T}
    p::Tuple
    input_functions::Vector{Function}
    ode_problem::ODEProblem
    sol::Union{OrdinaryDiffEq.ODECompositeSolution, Nothing}
end

# internal interface to reduce refactoring required
root(parts::AFMModelParts) = parts.root
reduced_graph(parts::AFMModelParts) = parts.reduced_graph
nnm(parts::AFMModelParts) = p(parts)[7]
inm(parts::AFMModelParts) = p(parts)[8]
nom(parts::AFMModelParts) = parts.nom
iom(parts::AFMModelParts) = parts.iom
raw_mats(parts::AFMModelParts) = (nnm(parts), inm(parts), nom(parts), iom(parts))
tspan(parts::AFMModelParts) = parts.tspan
u0(parts::AFMModelParts) = parts.u0
p(parts::AFMModelParts) = parts.p
input_functions(parts::AFMModelParts) = parts.input_functions
ode_problem(parts::AFMModelParts) = parts.ode_problem
sol(parts::AFMModelParts) = parts.sol

function set_root!(parts::AFMModelParts, root::Component)
    parts.root = root
end
function set_reduced_graph!(parts::AFMModelParts, reduced_graph::Graph)
    parts.reduced_graph = reduced_graph
end
function set_nom!(parts::AFMModelParts, nom)
    parts.nom = nom
end
function set_iom!(parts::AFMModelParts, iom)
    parts.iom = iom
end
function set_tspan!(parts::AFMModelParts, tspan::Tuple)
    parts.tspan = tspan
end
function set_u0!(parts::AFMModelParts, u0::AbstractMatrix)
    parts.u0 = u0
end
function set_p!(parts::AFMModelParts, p::Tuple)
    parts.p = p
end
function set_input_functions!(parts::AFMModelParts, input_functions::Vector{Function})
    parts.input_functions = input_functions
end
function set_ode_problem!(parts::AFMModelParts, ode_problem::ODEProblem)
    parts.ode_problem = ode_problem
end
function set_sol!(parts::AFMModelParts, sol::Union{OrdinaryDiffEq.ODECompositeSolution, Nothing})
    parts.sol = sol
end
function solve_parts!(parts::AFMModelParts)
    set_sol!(parts, solve(ode_problem(parts)))
end

build_u0(parts::AFMModelParts) = build_u0(root(parts))

function build_model_parts(root::Component, tspan, input_functions::Vector{Function}=Vector{Function}())
    graph = Graph{Float64}(root)
    substitute_internal_io!(graph)
    mats = raw.(reduced_graph_to_labeled_matrix(graph))
    u0 = build_u0(root)
    neuron_params = build_neuron_params(root)
    model_input_temp = similar(neuron_params[1], length(root.input))
    n_voltage_temp = similar(neuron_params[1])
    n_arr_temp2 = similar(neuron_params[1])
    n_arr_accum = similar(neuron_params[1])

    p = (neuron_params..., mats[1], mats[2], model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, input_functions)
    prob = ODEProblem(afm_diffeq!, u0, tspan, p)
    # TODO: where i left off refactoring
    AFMModelParts{Float64}(root, graph, mats[3], mats[4], tspan, u0, p, input_functions, prob, nothing)
    # AFMModelParts{Float64}(Graph{Float64}(nodes, weights), raw(mats[1]), raw(mats[2]), raw(mats[3]), raw(mats[4]), u0, tspan, input_functions, prob, nothing)
end

function rebuild_model_parts!(parts::AFMModelParts; new_tspan=nothing, new_input_functions=nothing)
    if !isnothing(new_tspan)
        set_tspan!(parts, new_tspan)
    end
    if !isnothing(new_input_functions)
        set_input_functions!(parts, new_input_functions)
    end
    graph = Graph{Float64}(root(parts))
    substitute_internal_io!(graph)
    mats = raw.(reduced_graph_to_labeled_matrix(graph))
    u0 = build_u0(root(parts))
    neuron_params = build_neuron_params(root(parts))
    model_input_temp = similar(neuron_params[1], length(root(parts).input))
    n_voltage_temp = similar(neuron_params[1])
    n_arr_temp2 = similar(neuron_params[1])
    n_arr_accum = similar(neuron_params[1])

    p = (neuron_params..., mats[1], mats[2], model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, input_functions(parts))
    prob = ODEProblem(afm_diffeq!, u0, tspan(parts), p)

    set_reduced_graph!(parts, graph)
    set_nom!(parts, mats[3])
    set_iom!(parts, mats[4])
    set_u0!(parts, u0)
    set_p!(parts, p)
    set_ode_problem!(parts, prob)
    set_sol!(parts, nothing)
    nothing
end

function make_gaussian(a, b, c)
    function gaussian(t)
        a * ℯ^(-((t - b)^2)/(c^2))
    end
end

# takes a vector of inputs from 0 to 1 and converts them to spikes with magnitues proportional to the input
function input_to_spikes(inputs::Vector{Float64})::Vector{Function}
    input_funs = Vector{Function}()
    for i in inputs
        push!(input_funs, make_gaussian(0.0026 * i, 0.7e-12, 3e-13))
    end
    input_funs
end

function peak_output(parts::AFMModelParts, sol)
    output = zeros(Float64, size(parts.iom)[1])
    curr_output = zeros(Float64, size(parts.iom)[1])
    curr_output_accum = zeros(Float64, size(parts.iom)[1])
    curr_input = zeros(Float64, size(parts.iom)[2])
    for i in 1:size(sol)[3]
        dΘ = view(sol, [:, 2, i])
        mul!(curr_output, parts.iom, dΘ)
    end

    # mul!(model_arr_accum, iom, model_input_temp)
    # mul!(model_output_temp, nom, n_voltage_temp)
    # model_arr_accum .+= model_output_temp # accumulate current


end

function input(parts::AFMModelParts)
    # https://discourse.julialang.org/t/mapping-vector-of-functions-to-vector-of-numbers/20942
    transpose(sol(parts).t) .|> input_functions(parts)
    # (|>).(transpose(parts.sol.t), parts.input_functions)
end

function output(parts::AFMModelParts)
    # Θ_part = view(parts.sol, :, 1, :)
    dΘ_part = view(parts.sol, :, 2, :)
    # Θ_output = parts.nom * Θ_part
    dΘ_output = (nom(parts) * dΘ_part) + (iom(parts) * input(parts))

    # TODO: compute Θ of input functions?
end

function output_max(parts::AFMModelParts)
    dΘ_output = output(parts)
    findmax(dΘ_output, dims = 2)[1][:]
end

function output_binary(parts::AFMModelParts, threshold=2e12)
    output_max(parts) .> threshold
end

function afm_diffeq!(du, u, p, t)
    sigma, a, we, wex, beta, bias, nnm, inm, model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, input_functions = p
    Φ = view(u, :, 1)
    dΦ = view(u, :, 2)
    duΦ = view(du, :, 1)
    dudΦ = view(du, :, 2)

    duΦ .= dΦ

    # populate model_input_temp from list of functions of time
    for i in eachindex(input_functions)
        model_input_temp[i] = input_functions[i](t)
    end

    # model_input to neuron_input
    n_voltage_temp .= dΦ .* beta # n_voltage_temp represents the voltage generated from the neurons
    mul!(n_arr_accum, inm, model_input_temp)
    mul!(n_arr_temp2, nnm, n_voltage_temp)
    n_arr_accum .+= n_arr_temp2 # accumulate current

    n_arr_accum .+= bias # add bias current

    @. dudΦ = (sigma * n_arr_accum - a*dΦ - (we/2) * sin(2*Φ)) * wex
end

# function build_neuron_labels(nodes::Vector{Node})
#     neuron_nodes = filter(x->x.type == :neuron, nodes)
#     hcat(map(x->"Θ" * node_str(x), neuron_nodes)..., map(x->"dΘ" * node_str(x), neuron_nodes)...)
# end

function build_neuron_labels(nodes::Vector{Node})
    neuron_nodes = filter(x->x.type == :neuron, nodes)
    hcat(map(x->node_str(x), neuron_nodes)...)
end

# function build_neuron_node_paths(nodes::Vector{Node})
#     filter(x->x.type == :neuron, nodes)
# end

build_neuron_labels(parts::AFMModelParts) = build_neuron_labels(nodes(reduced_graph(parts)))

function plot_Θ(parts::AFMModelParts; args...)
    label = build_neuron_labels(parts)
    plot(parts.sol, vars=hcat(1:length(label)), label=label, yaxis="Θ"; args...)
end

function plot_dΘ(parts::AFMModelParts; args...)
    label = build_neuron_labels(parts)
    plot(parts.sol, vars=hcat(length(label):(length(label)*2)-1), label=label, yaxis="dΘ"; args...)
end

function is_subpath(subpath::String, path::String)
    subpath_pos = findfirst(subpath, path)
    if isnothing(subpath_pos)
        return false
    else
        return subpath_pos.start == 1
    end
end

# returns true if the path is only one deeper than subpath
function is_immediate_subpath(subpath::String, path::String)
    subpath_pos = findfirst(subpath, path)
    if isnothing(subpath_pos)
        return false
    else
        return subpath_pos.start == 1 && count(x->x=='[', path[subpath_pos.stop:end]) == 0
    end
end

function plot_dΘ(parts::AFMModelParts, path::String, full_depth::Bool = false; args...)
    neuron_nodes = filter(x->x.type == :neuron, nodes(reduced_graph(parts)))
    label = map(x->node_str(x), neuron_nodes)
    first = length(label)
    
    indices = if full_depth
        findall(x->is_subpath(path, x), label)
    else
        findall(x->is_immediate_subpath(path, x), label)
    end
    filtered_labels = label[indices]
    plot(parts.sol, vars=hcat((indices .+first .- 1)...), label=reshape(filtered_labels, (1, length(filtered_labels))), yaxis="dΘ"; args...)
end

function plot_Θ(parts::AFMModelParts, path::String, full_depth::Bool = false; args...)
    neuron_nodes = filter(x->x.type == :neuron, nodes(reduced_graph(parts)))
    label = map(x->node_str(x), neuron_nodes)
    first = 1
    
    indices = if full_depth
        findall(x->is_subpath(path, x), label)
    else
        findall(x->is_immediate_subpath(path, x), label)
    end
    filtered_labels = label[indices]
    plot(parts.sol, vars=hcat((indices .+first .- 1)...), label=reshape(filtered_labels, (1, length(filtered_labels))), yaxis="Θ"; args...)
end

function plot_output(parts::AFMModelParts, output_index::Int; args...)
    label = output_labels(root(parts))
    plot(sol(parts).t, transpose(output(parts))[:, output_index], label=reshape(label, (1, length(label)))[output_index]; args...)
end

function plot_output(parts::AFMModelParts; args...)
    label = output_labels(root(parts))
    plot(sol(parts).t, transpose(output(parts)), label=reshape(label, (1, length(label))); args...)
end

function plot_input(parts::AFMModelParts, input_index::Int; args...)
    label = input_labels(root(parts))
    plot(sol(parts).t, transpose(input(parts))[:, input_index], label=reshape(label, (1, length(label)))[input_index]; args...)
end

function plot_input(parts::AFMModelParts; args...)
    label = input_labels(root(parts))
    plot(sol(parts).t, transpose(input(parts)), label=reshape(label, (1, length(label))); args...)
end

function parameter_mask_view(root::Component)
    param_views = Vector{SubArray{Float64, 2}}()
    mask_views = Vector{SubArray{Bool, 2}}()

    unique_components = unique(map_component_depth_first(x->x, root))
    for c in unique_components
        push!(param_views, view(raw(weights(c)), :, :))
        push!(mask_views, view(raw(weights_trainable_mask(c)), :, :))
    end

    (VectorOfArray(param_views), VectorOfArray(mask_views))
end
