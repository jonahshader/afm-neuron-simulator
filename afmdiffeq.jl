include("graph/afmgraph_methods.jl")
include("afmcomponent.jl")
include("utils.jl")

using LinearAlgebra
using SparseArrays
using DifferentialEquations
using RecursiveArrayTools

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
function solve!(parts::AFMModelParts)
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

function build_plot_labels(nodes::Vector{Node})
    neuron_nodes = filter(x->x.type == :neuron, nodes)
    hcat(map(x->"Θ" * node_str(x), neuron_nodes)..., map(x->"dΘ" * node_str(x), neuron_nodes)...)
end

function plot_Θ(parts::AFMModelParts; args...)
    label = build_plot_labels(nodes(reduced_graph(parts)))
    first = 1
    last = length(label)÷2
    plot(parts.sol, vars=hcat(first:last), label=label[:, first:last], args...)
end

function plot_dΘ(parts::AFMModelParts; args...)
    label = build_plot_labels(nodes(reduced_graph(parts)))
    first = (length(label)÷2) + 1
    last = length(label)
    plot(parts.sol, vars=hcat(first:last), label=label[:, first:last], args...)
end

# TODO: make this better. should use output labels like other plotting functions
# should also plot all outputs instead of one
function plot_output(parts::AFMModelParts, output_index::Int; args...)
    if typeof(output(parts)[:]) <: Vector
        plot(sol(parts).t, output(parts)[:], args...)
    else
        plot(sol(parts).t, output(parts)[:][output_index], args...)
    end
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
