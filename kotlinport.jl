using LinearAlgebra

const TERA = 10e12
const GIGA = 10e9
const FEMTO = 10e-15

const _fex = 27.5 * TERA
const _a = 0.01
const _fe = 1.75 * GIGA
const _sigma = 2.16 * TERA

const _wex = _fex * 2pi
const _we = _fe * 2pi
const _beta = 0.11 * FEMTO

mutable struct Component
    inputs::Int
    outputs::Int
    neurons::Int
    components::Vector{Component}
    theta::Vector{Float64}
    d_theta::Vector{Float64}
    dd_theta::Vector{Float64}
    bias::Vector{Float64}

    output_weights::Matrix{Float64}
    non_output_weights::Matrix{Float64}
    output::Vector{Float64}
end

function Component(inputs::Int, outputs::Int, neurons::Int)
    comp = Component(inputs, outputs, neurons, 
    Vector{Component}(), zeros(Float64, neurons), zeros(Float64, neurons),
    zeros(Float64, neurons), zeros(Float64, neurons) .+ 0.0023, zeros(Float64, 0, 0), 
    zeros(Float64, 0, 0), zeros(Float64, outputs))

    rebuild_mats!(comp)
    comp
end


c_outputs(c::Component) = sum(map(x->x.outputs, c.components))
c_inputs(c::Component) = sum(map(x->x.inputs, c.components))

function rebuild_mats!(c::Component)
    c.output_weights = randn(Float64, c.outputs, c_outputs(c) + c.neurons)
    c.non_output_weights = randn(Float64, c_inputs(c) + c.neurons, c.inputs + c_outputs(c) + c.neurons)
    nothing
end

function add_component!(parent::Component, child::Component)
    push!(parent.components, child)
    rebuild_mats!(parent)
    nothing
end

# # applies f to c tree in depth first order, returns array of results from applying f to every c
# function map_component_depth_first(f, c::Component)
#     return vcat([f(c)], map(x -> map_component_depth_first(f, x), c.components)...)
# end

# # same thing as map_component_depth_first, but does not put each result into an element in an array
# # instead every result is concatinated into one array
# function map_component_array_depth_first(f, c::Component)
#     return vcat(f(c), map(x -> map_component_depth_first(f, x), c.components)...)
# end

function compute_outputs!(c::Component)
    # TODO: handle zero size case?
    component_outputs = vcat(map(x -> compute_outputs!(x), c.components)...)
    mul!(c.output, c.output_weights, vcat(component_outputs, c.d_theta))
    c.output
end

function compute_dd_theta!(c::Component, input)
    non_output_weight_input = vcat(input, map(x->x.output, c.components)..., c.d_theta)
    mm = c.non_output_weights * non_output_weight_input
    component_inputs = view(mm, 1:c_inputs(c))
    neuron_inputs = view(mm, c_inputs(c)+1:length(mm))
    c.dd_theta = (_sigma * neuron_inputs - _a*c.d_theta-(_we/2.0)*sin.(c.theta * 2.0)) * _wex

    c_index = 0
    for comp in c.components
        c_input = view(component_inputs, c_index+1:(c_index+comp.inputs))
        compute_dd_theta!(comp, c_input)
        c_index += comp.inputs
    end
end

# root = Component(5, 10, 3)
# sub = Component(2, 11, 4)
# sub2 = Component(10, 25, 30)

# add_component!(sub, sub2)
# add_component!(root, sub)

# input = rand(Float64, 5)
# output = compute_outputs!(root)
# compute_dd_theta!(root, input)
# print(root.dd_theta)