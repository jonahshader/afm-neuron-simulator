include("Neurons.jl")

mutable struct BufMatrix
    m::Matrix{Float64}
    in::Vector{Float64}
    out::Vector{Float64}
end

BufMatrix(input_length::Int, output_length::Int) = BufMatrix(zeros(Float64, output_length, input_length), zeros(Float64, input_length), zeros(Float64, output_length))

mutable struct Component
    input::Vector{Float64}
    output::Vector{Float64}

    components::Vector{Component}
    neurons::Neurons


    input_to_neuron::BufMatrix
    input_to_component::BufMatrix

    neuron_to_neuron::BufMatrix
    neuron_to_component::BufMatrix
    neuron_to_output::BufMatrix

    component_to_neuron::BufMatrix
    component_to_component::BufMatrix
    component_to_output::BufMatrix


    # neuron_to_output::BufMatrix
    # component_to_output::BufMatri
    # neuron_to_neuron::BufMatrix
    # output_to_output::BufMatrix
    

end

input_length(c::Component) = length(c.input)
output_length(c::Component) = length(c.output)
neuron_length(c::Component) = length(c.neurons)
components(c::Component) = c.components

function build_weights!(c::Component)
    cinputs = sum(map(x->input_length(x), components(c)))
    coutputs = sum(map(x->output_length(x), components(c)))
    c.input_to_neuron = BufMatrix(input_length(c), neuron_length(c))
    c.input_to_component = BufMatrix(input_length(c), cinputs)

    c.neuron_to_neuron = BufMatrix(neuron_length(c), neuron_length(c))
    c.neuron_to_component = BufMatrix(neuron_length(c), cinputs)
    c.neuron_to_output = BufMatrix(neuron_length(c), output_length(c))

    c.component_to_neuron = BufMatrix(coutputs, neuron_length(c))
    c.component_to_component = BufMatrix(coutputs, cinputs)
    c.component_to_output = BufMatrix(coutputs, output_length(c))
    # c.component_to_output
    # c.output_weights = randn(Float64, 
    # c.outputs, 
    # length(c.neurons) + sum(map(x->x.outputs, c.components)))
    # c.non_output_weights = randn(Float64, 
    # length(c.neurons) + sum(map(x->x.inputs, c.components)), 
    # c.inputs + length(c.neurons) + sum(map(x->x.outputs, c.components)))
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