include("Neurons.jl")

mutable struct Component
    inputs::Int
    outputs::Int

    components::Vector{Component}
    neurons::Neurons

    output_weights::Matrix{Float64}
    output_weights_input::Vector{Float64}
    output_weights_output::Vector{Float64}
    non_output_weights::Matrix{Float64}
    non_output_weights_input::Vector{Float64}
    non_output_weights_output::Vector{Float64}
end

function build_weights!(c::Component)
    c.output_weights = randn(Float64, 
    c.outputs, 
    length(c.neurons) + sum(map(x->x.outputs, c.components)))
    c.non_output_weights = randn(Float64, 
    length(c.neurons) + sum(map(x->x.inputs, c.components)), 
    c.inputs + length(c.neurons) + sum(map(x->x.outputs, c.components)))
end

function build_problem(root::Component)


end

function compute_component_output(comp::Component)

    for c in comp
        c.compute_component_output(c)
    end

    neuron_view = view(comp.output_weights_input, 1:length(comp.neurons))
    neuron_view .= 


end