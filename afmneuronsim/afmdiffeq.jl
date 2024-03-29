# include("graph/afmgraph_methods.jl")
# include("afmcomponent.jl")
# include("utils.jl")

using LinearAlgebra
using SparseArrays
using DifferentialEquations
using RecursiveArrayTools
using Plots


# export root, set_root!Φ
# export nnm
# export inm
# export nom
# export iom
# export tspan, set_tspan!
# export u0, set_u0!
# export input_functions
# export ode_problem
# export sol

export AFMModelParts
export solve_parts!
export input_to_spikes
export build_model_parts
export build_and_solve
export rebuild_model_parts!
export change_input_functions!
export peak_output
export input_dΦ
export output_dΦ
export output_max
export output_binary
export build_neuron_labels
export plot_Φ
export plot_dΦ
export plot_output
export plot_input
export weight_mask_view

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
    sparse_::Union{Bool, Nothing}
    gpu::Bool
    sol::Union{SciMLBase.AbstractODESolution, Nothing}
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
sparse_(parts::AFMModelParts) = parts.sparse_
gpu(parts::AFMModelParts) = parts.gpu
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
function set_sparse_!(parts::AFMModelParts, sparse_::Bool)
    parts.sparse_ = sparse_
end
function set_gpu!(parts::AFMModelParts, gpu::Bool)
    parts.gpu = gpu
end
function set_sol!(parts::AFMModelParts, sol)
    parts.sol = sol
end

"""
    solve_parts!(parts::AFMModelParts)

Solves the ODE described by `parts` using DifferentialEquations.jl's solve function.
This populates the `sol` field of `parts`, which can be accessed using `sol(parts)`.
"""
function solve_parts!(parts::AFMModelParts; args...)
    set_sol!(parts, solve(ode_problem(parts); args...))
end

build_u0(parts::AFMModelParts) = build_u0(root(parts))

function build_model_parts(root::Component, tspan, input_functions::Vector{Function}=Vector{Function}(); sparse_=nothing, gpu=false, custom_state=nothing)
    stype = if gpu Float32 else Float64 end
    graph = Graph{stype}(root)
    substitute_internal_io!(graph)
    mats = raw.(reduced_graph_to_labeled_matrix(graph, sparse_, gpu))
    u0 = if isnothing(custom_state)
        build_u0(root)
    else
         custom_state
    end
    neuron_params = build_neuron_params(root, gpu)
    model_input_temp = similar(neuron_params[1], length(root.input))
    n_voltage_temp = similar(neuron_params[1])
    n_arr_temp2 = similar(neuron_params[1])
    n_arr_accum = similar(neuron_params[1])

    p = (neuron_params..., mats[1], mats[2], model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, (input_functions...,))
    prob = ODEProblem(afm_diffeq!, u0, tspan, p)
    # TODO: where i left off refactoring
    
    AFMModelParts{stype}(root, graph, mats[3], mats[4], tspan, u0, p, input_functions, prob, sparse_, gpu, nothing)
    # AFMModelParts{Float64}(Graph{Float64}(nodes, weights), raw(mats[1]), raw(mats[2]), raw(mats[3]), raw(mats[4]), u0, tspan, input_functions, prob, nothing)
end


function rebuild_model_parts!(parts::AFMModelParts; model_changed=true, new_tspan=nothing, new_input_functions=nothing, new_sparse=nothing, new_gpu=nothing)
    if !isnothing(new_sparse)
        set_sparse_!(parts, new_sparse)
    end
    if !isnothing(new_gpu)
        set_gpu!(parts, new_gpu)
    end
    if !isnothing(new_tspan)
        set_tspan!(parts, new_tspan)
    end
    if !isnothing(new_input_functions)
        set_input_functions!(parts, new_input_functions)
    end
    stype = if gpu(parts) Float32 else Float64 end
    prob = if model_changed
        graph = Graph{stype}(root(parts))
        substitute_internal_io!(graph)
        set_reduced_graph!(parts, graph)
        mats = raw.(reduced_graph_to_labeled_matrix(graph, sparse_(parts), gpu(parts)))
        set_nom!(parts, mats[3])
        set_iom!(parts, mats[4])
        parts.u0 = build_u0(root(parts), gpu(parts))
        neuron_params = build_neuron_params(root(parts), gpu(parts))
        model_input_temp = similar(neuron_params[1], length(root(parts).input))
        n_voltage_temp = similar(neuron_params[1])
        n_arr_temp2 = similar(neuron_params[1])
        n_arr_accum = similar(neuron_params[1])
    
        parts.p = (neuron_params..., mats[1], mats[2], model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, (input_functions(parts)...,))

        ODEProblem(afm_diffeq!, u0(parts), tspan(parts), parts.p)
    else
        parts.p = (parts.p[1:end-1]..., (input_functions(parts)...,))
        ODEProblem(afm_diffeq!, u0(parts), tspan(parts), parts.p)
    end

    # set_u0!(parts, u0)
    # set_p!(parts, p)
    set_ode_problem!(parts, prob)
    set_sol!(parts, nothing)
    nothing
end

function change_input_functions!(parts::AFMModelParts, new_input_functions)
    set_input_functions!(parts, new_input_functions)
    parts.p = (parts.p[1:end-1]..., (input_functions(parts)...,))
    set_ode_problem!(parts, ODEProblem(afm_diffeq!, u0(parts), tspan(parts), parts.p))
    set_sol!(parts, nothing)
    nothing
end

function build_and_solve(root::Component, tspan, input_functions::Vector{Function}=Vector{Function}(), sparse_=nothing, gpu=false; args...)
    parts = build_model_parts(root, tspan, input_functions; sparse_=sparse_, gpu=gpu)
    solve_parts!(parts; args...)
    parts
end

function continue!(parts::AFMModelParts, tspan, input_functions::Vector{Function}=nothing)
    set_u0!(parts, sol(parts)[end])
    set_tspan!(parts, tspan)
    if !isnothing(input_functions)
        set_input_functions!(parts, input_functions)
    end
    prob = ODEProblem(afm_diffeq!, u0(parts), tspan(parts), p(parts))
    set_ode_problem!(parts, prob)
    solve_parts!(parts)
end

# function build_and_continue(root::Component, parts::AFMModelParts, tspan, input_functions::Vector{Function}=Vector{Function}())
#     new_parts = build_model_parts(root, tspan, input_functions, sparse_(parts), gpu(parts), custom_state=sol(parts)[end])
#     solve_parts!(new_parts)
#     new_parts
# end

function make_gaussian(a, b, c)
    function gaussian(t)
        a * ℯ^(-((t - b)^2)/(c^2))
    end
end

"""
    input_to_spikes(inputs::Vector{Float64}, peak_current=0.0026, spike_center=7e-13, spike_width=3e-13)::Vector{Function}

Takes a vector of inputs with values between 0 and 1 and converts them to spikes with magnitudes proportional to the input.
The result is the vector of functions which take in time and return current. These are intended to be passed to `build_model_parts` as input functions.
The spikes produced by this function are Gaussian. Properties of the spike can be specified by overriding the default values, which are
`peak_current`, `spike_center`, and `spike_width`.
"""
#1.16875e-5
#peak_current=0.0001, spike_center=21e-13, spike_width=9e-13
function input_to_spikes(inputs::Vector{Float64}; peak_current=1.16875e-5, spike_center=60*PICO, spike_width=1.5e-11)::Vector{Function}
    input_funs = Vector{Function}()
    for i in inputs
        push!(input_funs, make_gaussian(peak_current * i, spike_center, spike_width))
    end
    input_funs
end

function peak_output(parts::AFMModelParts, sol)
    output = zeros(Float64, size(parts.iom)[1])
    curr_output = zeros(Float64, size(parts.iom)[1])
    curr_output_accum = zeros(Float64, size(parts.iom)[1])
    curr_input = zeros(Float64, size(parts.iom)[2])
    for i in 1:size(sol)[3]
        dΦ = view(sol, [:, 2, i])
        mul!(curr_output, parts.iom, dΦ)
    end

    # mul!(model_arr_accum, iom, model_input_temp)
    # mul!(model_output_temp, nom, n_voltage_temp)
    # model_arr_accum .+= model_output_temp # accumulate current


end

function input_dΦ(parts::AFMModelParts)
    # https://discourse.julialang.org/t/mapping-vector-of-functions-to-vector-of-numbers/20942
    transpose(sol(parts).t) .|> input_functions(parts)
end

# function input_Φ(parts::AFMModelParts)

# end

function output_dΦ(parts::AFMModelParts)
    # Φ_part = view(parts.sol, :, 1, :)
    dΦ_part = view(parts.sol, :, 2, :)
    # Φ_output = parts.nom * Φ_part
    (nom(parts) * dΦ_part) + (iom(parts) * input_dΦ(parts))

    # TODO: compute Φ of input functions?
end

"""
    output_Φ(parts::AFMModelParts)

Returns neuron-to-output matrix times neuron Φ. The input-to-output matrix is currently ignored, as that would require the integrals of the input functions to be known or computed.
"""
function output_Φ(parts::AFMModelParts)
    Φ_part = view(parts.sol, :, 1, :)
    nom(parts) * Φ_part
end

function output_max(parts::AFMModelParts)
    dΦ_output = output_dΦ(parts)
    findmax(dΦ_output, dims = 2)[1][:]
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
    model_input_temp .= t .|> input_functions
    # for i in eachindex(input_functions)
    #     model_input_temp[i] = input_functions[i](t)
    # end

    # model_input to neuron_input
    n_voltage_temp .= dΦ .* beta # n_voltage_temp represents the voltage generated from the neurons
    mul!(n_arr_accum, inm, model_input_temp)
    mul!(n_arr_temp2, nnm, n_voltage_temp)
    n_arr_accum .+= n_arr_temp2 # accumulate current

    n_arr_accum .+= bias # add bias current

    @. dudΦ = (sigma * n_arr_accum - a*dΦ - (we/2) * sin(2*Φ)) * wex
    nothing
end

function build_neuron_labels(nodes::Vector{Node})
    neuron_nodes = filter(x->x.type == :neuron, nodes)
    hcat(map(x->node_str(x), neuron_nodes)...)
end

build_neuron_labels(parts::AFMModelParts) = build_neuron_labels(nodes(reduced_graph(parts)))
"""
    plot_Φ(parts::AFMModelParts; args...)

Plots Φ of all neurons in `parts`. `parts` must be solved with `solve_parts!` prior to plotting. Additional args can be passed to the `plot` function.
e.g. `plot_Φ(parts, title="Φ of all neurons in parts")`
"""
function plot_Φ(parts::AFMModelParts; args...)
    label = build_neuron_labels(parts)
    plot(parts.sol, vars=hcat(1:length(label)), label=label, yaxis="Φ"; args...)
end

"""
    plot_dΦ(parts::AFMModelParts; args...)

Plots dΦ of all neurons in `parts`. `parts` must be solved with `solve_parts!` prior to plotting. Additional args can be passed to the `plot` function.
e.g. `plot_dΦ(parts, title="dΦ of all neurons in parts")`
"""
function plot_dΦ(parts::AFMModelParts; args...)
    label = build_neuron_labels(parts)
    plot(parts.sol, vars=hcat(length(label)+1:(length(label)*2)), label=label, yaxis="dΦ"; args...)
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

"""
    plot_dΦ(parts::AFMModelParts, path::String, full_depth::Bool = false; args...)

Plots dΦ of all neurons located at `path` within the component tree. If `full_depth` is true, then all neurons located at `path` or deeper are plotted.
`parts` must be solved with `solve_parts!` prior to plotting. The format of `path` is the same as is displayed with the regular `plot_dΦ` method.
e.g. `plot_dΦ(parts, "[xor1]", full_depth=true)` would plot dΦ of all neurons in the xor1 component and below, whereas if full_depth=false, only neurons in the xor1 component would be plotted.

Additional args can be passed to the `plot` function.
e.g. `plot_dΦ(parts, "[xor1]", title="dΦ of all neurons in parts")`
"""
function plot_dΦ(parts::AFMModelParts, path::String, full_depth::Bool = false; args...)
    neuron_nodes = filter(x->x.type == :neuron, nodes(reduced_graph(parts)))
    label = map(x->node_str(x), neuron_nodes)
    first = length(label) + 1
    indices = if full_depth
        findall(x->is_subpath(path, x), label)
    else
        findall(x->is_immediate_subpath(path, x), label)
    end

    @assert !isempty(indices) "No neurons selected to plot!"
    filtered_labels = label[indices]
    plot(parts.sol, vars=hcat((indices .+first .- 1)...), label=reshape(filtered_labels, (1, length(filtered_labels))), yaxis="dΦ"; args...)
end

"""
    plot_dΦ(parts::AFMModelParts, paths::Vector{String}, full_depth::Bool = false; args...)

Plots dΦ of all neurons located at `paths` within the component tree. If `full_depth` is true, then all neurons located at `path` or deeper are plotted.
`parts` must be solved with `solve_parts!` prior to plotting. The format of `path` is the same as is displayed with the regular `plot_dΦ` method.
e.g. `plot_dΦ(parts, ["[xor1]", "[xor2]"], full_depth=true)` would plot dΦ of all neurons in the xor1 and xor2 components and below, whereas if full_depth=false, only neurons in the xor1 and xor2 components would be plotted.

Additional args can be passed to the `plot` function.
e.g. `plot_dΦ(parts, ["[xor1]", "[xor2]"], title="dΦ of all neurons in parts")`
"""
function plot_dΦ(parts::AFMModelParts, paths::Vector{String}, full_depth::Bool = false; args...)
    neuron_nodes = filter(x->x.type == :neuron, nodes(reduced_graph(parts)))
    label = map(x->node_str(x), neuron_nodes)
    first = length(label) + 1
    indices = if full_depth
        findall(x->max([is_subpath(p, x) for p in paths]...), label)
    else
        findall(x->max([is_immediate_subpath(p, x) for p in paths]...), label)
    end

    @assert !isempty(indices) "No neurons selected to plot!"
    filtered_labels = label[indices]
    plot(parts.sol, vars=hcat((indices .+first .- 1)...), label=reshape(filtered_labels, (1, length(filtered_labels))), yaxis="dΦ"; args...)
end


"""
    plot_Φ(parts::AFMModelParts, path::String, full_depth::Bool = false; args...)

Plots Φ of all neurons located at `path` within the component tree. If `full_depth` is true, then all neurons located at `path` or deeper are plotted.
`parts` must be solved with `solve_parts!` prior to plotting. The format of `path` is the same as is displayed with the regular `plot_Φ` method.
e.g. `plot_Φ(parts, "[xor1]", full_depth=true)` would plot Φ of all neurons in the xor1 component and below, whereas if full_depth=false, only neurons in the xor1 component would be plotted.

Additional args can be passed to the `plot` function.
e.g. `plot_Φ(parts, "[xor1]", title="Φ of all neurons in parts")`
"""
function plot_Φ(parts::AFMModelParts, path::String, full_depth::Bool = false; args...)
    neuron_nodes = filter(x->x.type == :neuron, nodes(reduced_graph(parts)))
    label = map(x->node_str(x), neuron_nodes)
    first = 1
    
    indices = if full_depth
        findall(x->is_subpath(path, x), label)
    else
        findall(x->is_immediate_subpath(path, x), label)
    end

    @assert !isempty(indices) "No neurons selected to plot!"
    filtered_labels = label[indices]
    plot(parts.sol, vars=hcat((indices .+first .- 1)...), label=reshape(filtered_labels, (1, length(filtered_labels))), yaxis="Φ"; args...)
end

    
"""
    plot_Φ(parts::AFMModelParts, paths::Vector{String}, full_depth::Bool = false; args...)

Plots Φ of all neurons located at `paths` within the component tree. If `full_depth` is true, then all neurons located at `path` or deeper are plotted.
`parts` must be solved with `solve_parts!` prior to plotting. The format of `path` is the same as is displayed with the regular `plot_Φ` method.
e.g. `plot_Φ(parts, ["[xor1]", "[xor2]"], full_depth=true)` would plot Φ of all neurons in the xor1 and xor2 components and below, whereas if full_depth=false, only neurons in the xor1 and xor2 components would be plotted.

Additional args can be passed to the `plot` function.
e.g. `plot_Φ(parts, ["[xor1]", "[xor2]"], title="Φ of all neurons in parts")`
"""
function plot_Φ(parts::AFMModelParts, paths::Vector{String}, full_depth::Bool = false; args...)
    neuron_nodes = filter(x->x.type == :neuron, nodes(reduced_graph(parts)))
    label = map(x->node_str(x), neuron_nodes)
    first = 1
    
    indices = if full_depth
        findall(x->max([is_subpath(p, x) for p in paths]...), label)
    else
        findall(x->max([is_immediate_subpath(p, x) for p in paths]...), label)
    end

    @assert !isempty(indices) "No neurons selected to plot!"
    filtered_labels = label[indices]
    plot(parts.sol, vars=hcat((indices .+first .- 1)...), label=reshape(filtered_labels, (1, length(filtered_labels))), yaxis="Φ"; args...)
end

function plot_output(parts::AFMModelParts, output_index::Int; args...)
    label = output_labels(root(parts))
    plot(sol(parts).t, transpose(output_dΦ(parts))[:, output_index], label=reshape(label, (1, length(label)))[output_index], yaxis="Voltage"; args...)
end

function plot_output(parts::AFMModelParts; args...)
    label = output_labels(root(parts))
    plot(sol(parts).t, transpose(output_dΦ(parts)), label=reshape(label, (1, length(label))), yaxis="Voltage"; args...)
end

function plot_input(parts::AFMModelParts, input_index::Int; args...)
    label = input_labels(root(parts))
    plot(sol(parts).t, transpose(input_dΦ(parts))[:, input_index], label=reshape(label, (1, length(label)))[input_index]; args...)
end

function plot_input(parts::AFMModelParts; args...)
    label = input_labels(root(parts))
    plot(sol(parts).t, transpose(input_dΦ(parts)), label=reshape(label, (1, length(label))); args...)
end

function weight_mask_view(root::Component)
    param_views = Vector{SubArray{Float64, 2}}()
    mask_views = Vector{SubArray{Bool, 2}}()

    unique_components = unique(map_component_depth_first(x->x, root))
    for c in unique_components
        push!(param_views, view(raw(weights(c)), :, :))
        push!(mask_views, view(raw(weights_trainable_mask(c)), :, :))
    end

    (VectorOfArray(param_views), VectorOfArray(mask_views))
end
