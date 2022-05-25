include("Neurons.jl")

mutable struct BufMatrix
    m::Matrix{Float64}
    in::Vector{Float64}
    out::Vector{Float64}
end

BufMatrix(input_length::Int, output_length::Int) = BufMatrix(zeros(Float64, output_length, input_length), zeros(Float64, input_length), zeros(Float64, output_length))

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

    input::Vector{Float64}
    output::Vector{Float64}
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

    output_index = 1
    for c in comp.components
        compute_component_output(c)
        next_output_index = output_index + c.outputs
        neuron_view = view(comp.output_weights_input, output_index:(next_output_index-1))
        neuron_view .= 
        output_index = next_output_index
    end

    neuron_view = view(comp.output_weights_input, 1:length(comp.neurons))
    neuron_view .= 


end