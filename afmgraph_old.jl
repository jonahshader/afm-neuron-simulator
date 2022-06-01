include("Neurons.jl")
include("labeledmatrix.jl")
include("labeledlength.jl")
include("labeledvector.jl")
include("utils.jl")

const Label = Union{String, Int, Tuple{Int}, Tuple{String}, Tuple{Int, Int}, Tuple{Int, String}, Tuple{String, Int}, Tuple{String, String}}

mutable struct Component{T<:AbstractFloat}
    input::LabeledLength{String}
    output::LabeledLength{String}
    components::LabeledVector{Component, String}

    neurons::Neurons
    neuron_labels::Dict{String, Int}

    weights::Vector{Tuple{Label, Label}}
end

function Component(input_length::Int, output_length::Int)
    Component(
        LabeledLength{String}(input_length),
        LabeledLength{String}(output_length),
        LabeledVector{Component, String}(Vector{Component}()),
        Neurons(),
        Dict{String, Int}(),
        Vector{Tuple{Label, Label}}()
    )
end

function Component(input_length::Int, outputs::Vector{String})
    output_length = length(outputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.output, outputs)
    comp
end

function Component(inputs::Vector{String}, output_length::Int)
    input_length = length(inputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.input, inputs)
    comp
end

function Component(inputs::Vector{String}, outputs::Vector{String})
    input_length = length(inputs)
    output_length = length(outputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.input, inputs)
    set_labels!(comp.output, outputs)
    comp
end

# adds a child component to a parent, with a name
function add_component!(parent::Component, child::Component, name::String)
    push!(parent.components, child)
    @assert !haslabel(parent.components, name)
    parent.components
end