const Label = Union{String, Int, Tuple{Int}, Tuple{String}, Tuple{Int, Int}, Tuple{Int, String}, Tuple{String, Int}, Tuple{String, String}}

include("Neurons.jl")
include("utils.jl")
include("labeledmatrix.jl")
include("labeledlength.jl")
include("labeledvector.jl")

mutable struct Component
    input::LabeledLength{String}
    output::LabeledLength{String}

    components::LabeledVector{Component, String}

    neurons::Neurons
    neuron_labels::Dict{String, Int}

    # weights::AbstractMatrix{Float64}
    # output weights goes from {Neurons, Component Outputs} -> Outputs
    # shape is output_length x sum(x -> x.output_length, components)
    output_weights::LabeledMatrix{Float64, Label}
    # non output weights are everyting - output_weights
    # this goes from {Inputs, Neurons, Component Outputs} -> {Neurons, Component Outputs}
    non_output_weights::LabeledMatrix{Float64, Label}
end

# outer constructors
function Component(input_length::Int, output_length::Int)::Component
    comp = Component(
        LabeledLength{String}(input_length), LabeledLength{String}(output_length),
        LabeledVector{Component, String}(Vector{Component}()),
        Neurons(), Dict{String, Int}(),
        LabeledMatrix{Float64, Label}(zeros(Float64, 1, 1)), LabeledMatrix{Float64, Label}(zeros(Float64, 1, 1)))
    build_weights_matrix!(comp)
    comp
end

function Component(input_length::Int, outputs::Vector{String})::Component
    output_length = length(outputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.output, outputs)
    comp
end

function Component(inputs::Vector{String}, output_length::Int)::Component
    input_length = length(inputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.input, inputs)
    comp
end

function Component(inputs::Vector{String}, outputs::Vector{String})::Component
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


# adds a child component to parent, with a name
function add_component!(parent::Component, child::Component, name::String)
    @assert !haslabel(parent.components, name)
    push!(parent.components, child, name)
    build_weights_matrix!(parent)
    nothing
end

# add a child component to a parent (no name)
function add_component!(parent::Component, child::Component)
    push!(parent.components, child)
    build_weights_matrix!(parent)
    nothing
end

# method for tree navigation of components via component[string/int] syntax
function Base.getindex(c::Component, key_or_index)::Component
    return c.components[key_or_index]
end

function set_weight!(c::Component, source::Label, destination::Label, weight::AbstractFloat)
    c.non_output_weights[destination, source] = weight
    nothing
end

function set_weight!(c::Component, source::Label, destination::Int, weight::AbstractFloat)
    c.output_weights[destination, source] = weight
    nothing
end

# these are specified opposite of the weight matrix: sources correspond to rows, destinations correspond to cols
# TODO: flip this around? not sure what makes sense yet
function set_weights!(c::Component, sources::Vector{Label}, destinations::Vector{Label}, weights::Matrix{Float64})
    for i in sources
        for j in destinations
            source_index = c.all_sources[i]
            destination_index = c.all_destinations[j]
            c.weights[destination_index, source_index] = weights[source_index, destination_index]
        end
    end
    nothing
end


# inputs are ints or strings, neurons are tuples of ints or strings of length one, components are tuples of ints or strings of length two (name, output)
# sources are voltages that can be passed through the weights to get current. 
# these consist of the current component's inputs, the current component's neuron's outputs, and the subcomponent's outputs
function sources(c::Component)::Vector{Label}
    sources = Vector{Label}()
    # all of c's inputs are sources, so add them to the sources vector
    for i in inputs(c)
        push!(sources, i)
    end
    # all of c's neurons are outputs, and we address them as tuples of size one
    for n in neurons(c)
        push!(sources, (n,))
    end
    component_labels_reversed = Dict{Int, String}(value => key for (key, value) in c.component_labels)
    for i in 1:length(c.components)
        for output in outputs(c[i])
            # TODO: this logic in the if statement can be abstracted away somehow. maybe a get_label_maybe method or something
            if haskey(component_labels_reversed, i)
                push!(sources, (component_labels_reversed[i], output))
            else
                push!(sources, (i, output))
            end
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
    component_labels_reversed = Dict{Int, String}(value => key for (key, value) in c.component_labels)
    for i in 1:length(c.components)
        for input in inputs(c[i])
            if haskey(component_labels_reversed, i)
                push!(destinations, (component_labels_reversed[i], input))
            else
                push!(destinations, (i, input))
            end
        end
    end
    destinations
end



# make unique shallow (only current component is made unique)
# make unique deep (current component and all children component are made unique)


# PRIVATE FUNCTIONS. TODO: move to other file and use here, just don't re export them



# TODO: rewrite this
# function total_neuron_count(c::Component, current_neuron_count::Int)::Int
#     current_neuron_count + length(c.neurons) + reduce(+, c.components; init=0)
# end

# p contains weights and runtime neurons
function build_p_matrices(c::Component)
    return map_component_depth_first(comp->comp.weights, c)
end

# takes a list of labels and returns a map from label => int, where label is in labels and int is the label's index
function build_label_to_indices_map(labels::Vector{Label})::Dict{Label, Int}
    dict = Dict{Label, Int}()
    for i in 1:length(labels)
        label = labels[i]
        dict[label] = i
    end
    dict
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

# creates a weights matrix with the appropriate shape
# TODO: create a version of this that resizes the matrix, maintaining the valid weights instead of replacing everything with zeros
function build_weights_matrix!(c::Component)
    d = internal_destination_length(c)
    s = internal_source_length(c)
    c.output_weights = LabeledMatrix{Float64, Label}(zeros(Float64, output_length(c), s - input_length(c)))
    c.non_output_weights = LabeledMatrix{Float64, Label}(zeros(Float64, d - output_length(c), s))

end

