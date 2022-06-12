
include("afmneurons.jl")
include("utils.jl")
include("labeledmatrix.jl")
include("labeledlength.jl")
include("labeledvector.jl")

const ComponentLabel = Union{String, Int}
const NeuronLabel = Union{Tuple{String}, Tuple{Int}}
const SubComponentLabel = Tuple{ComponentLabel, ComponentLabel}
const Label = Union{ComponentLabel, NeuronLabel, SubComponentLabel}


mutable struct Component
    input::LabeledLength{String}
    output::LabeledLength{String}

    components::LabeledVector{Component, String}

    neurons::Neurons
    neuron_labels::Dict{String, Int}

    weights::LabeledMatrix{Float64, Label}
    weights_trainable_mask::LabeledMatrix{Bool, Label}
end

function Component(input_length::Int, output_length::Int)
    comp = Component(
        LabeledLength{String}(input_length),
        LabeledLength{String}(output_length),
        LabeledVector{Component, String}(Vector{Component}()),
        Neurons(),
        Dict{String, Int}(),
        LabeledMatrix{Float64, Label}(zeros(Float64, 1, 1)),
        LabeledMatrix{Bool, Label}(zeros(Bool, 1, 1))
    )
    build_weights_matrix!(comp)
    comp
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
    # pass inputs and outputs to labels
    set_labels!(comp.output, outputs)
    set_labels!(comp.input, inputs)
    comp
end

input_length(c::Component) = length(c.input)
output_length(c::Component) = length(c.output)

# these methods return arrays of labels, which can be used for indexing
# they return Vector{Union{String, Int}}, so the user needs to convert them to the appropriate format
# when using them later for referencing. E.g. neurons are referenced with the type Tuple{Union{String, Int}}
# and components are referenced with the type Tuple{Union{String, Int}, Union{String, Int}}
inputs(c::Component) = indices_with_labels(input_length(c), c.input.labels)
outputs(c::Component) = indices_with_labels(output_length(c), c.output.labels)
neurons(c::Component) = indices_with_labels(length(c.neurons), c.neuron_labels)
components(c::Component) = indices_with_labels(length(c.components), c.components.labels)

function add_neurons!(c::Component, n::Int)
    add_neurons!(c.neurons, n)
    build_weights_matrix!(c)
end

function add_component!(parent::Component, child::Component, name::String)
    @assert !haslabel(parent.components, name)
    push_and_label!(parent.components, child, name)
    build_weights_matrix!(parent)
    nothing
end

function add_component!(parent::Component, child::Component)
    push!(parent.components, child)
    build_weights_matrix!(parent)
    nothing
end

Base.getindex(c::Component, key_or_index)::Component = c.components[key_or_index]

function set_weight!(c::Component, source::Label, destination::Label, weight::AbstractFloat)
    c.weights[destination, source] = weight
    nothing
end

# TODO: write a version of this that works with a LabeledMatrix. it won't need sources/destinations vectors
function set_weights!(c::Component, sources::Vector{Label}, destinations::Vector{Label}, weights::AbstractMatrix{Float64})
    for i in 1:length(sources)
        source = sources[i]
        for j in 1:length(destinations)
            destination = destinations[j]
            c.weights[destination, source] = weights[j, i]
        end
    end
    nothing
end

# computes the total number of sources inside this component. represents the number of cols in weights matrix
function internal_source_length(c::Component)::Int
    curr_length = input_length(c)
    curr_length += length(c.neurons)
    for sc::Component in c.components.vector
        curr_length += output_length(sc)
    end
    curr_length
end
# computes the total number of destinations inside this component. represents the number of rows in weights matrix
function internal_destination_length(c::Component)::Int
    curr_length = output_length(c)
    curr_length += length(c.neurons)
    for sc::Component in c.components.vector
        curr_length += input_length(sc)
    end
    curr_length
end

function sources(c::Component)::Vector{Label}
    sources = Vector{Label}()
    for i in inputs(c)
        push!(sources, i)
    end
    for n in neurons(c)
        push!(sources, (n,))
    end

    for clabel in indices_with_labels(c.components)
        for output in outputs(c[clabel])
            push!(sources, (clabel, output))
        end
    end
    sources
end

function destinations(c::Component)::Vector{Label}
    destinations = Vector{Label}()
    for i in outputs(c)
        push!(destinations, i)
    end
    for n in neurons(c)
        push!(destinations, (n,))
    end

    for clabel in indices_with_labels(c.components)
        for input in inputs(c[clabel])
            push!(destinations, (clabel, input))
        end
    end
    destinations
end

function build_weights_matrix!(c::Component)
    int_dest_len = internal_destination_length(c)
    int_src_len = internal_source_length(c)
    c.weights = LabeledMatrix{Float64, Label}(zeros(Float64, int_dest_len, int_src_len))
    c.weights_trainable_mask = LabeledMatrix{Bool, Label}(zeros(Bool, int_dest_len, int_src_len))
    dest = destinations(c)
    src = sources(c)
    set_labels!(c.weights, dest, src)
    set_labels!(c.weights_trainable_mask, dest, src)
    nothing
end
