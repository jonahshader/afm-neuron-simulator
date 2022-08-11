
# include("afmneurons.jl")
# include("utils.jl")
# include("labeledmatrix.jl")
# include("labeledlength.jl")
# include("labeledvector.jl")

export Component
export add_component!
export add_neuron!
export add_neurons!

export set_weight!
export set_weights!
export set_weight_trainable!
export set_weights_trainable!

export input_length
export output_length
export input_labels
export output_labels
export neuron_labels
export component_labels

export input_dΦ
export output_dΦ
export components
export neurons
export weights
export weights_trainable_mask

export getindex
export sources_length
export destinations_length
export sources
export destinations


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

"""
    Component(input_length::Int, output_length::Int)

Creates a component with `input_length` inputs and `output_length` outputs.
"""
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

"""
    Component(input_length::Int, outputs::Vector{String})

Creates a component with `input_length` inputs and `outputs` labeled outputs.
"""
function Component(input_length::Int, outputs::Vector{String})
    output_length = length(outputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.output, outputs)
    comp
end

"""
    Component(inputs::Vector{String}, output_length::Int)

Creates a component with `inputs` labeled inputs and `output_length` outputs.
"""
function Component(inputs::Vector{String}, output_length::Int)
    input_length = length(inputs)
    comp = Component(input_length, output_length)
    set_labels!(comp.input, inputs)
    comp
end

"""
    Component(inputs::Vector{String}, outputs::Vector{String})

Creates a component with `inputs` labeled inputs and `outputs` labeled outputs.
"""
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
input_labels(c::Component) = indices_with_labels(input_length(c), c.input.labels)
output_labels(c::Component) = indices_with_labels(output_length(c), c.output.labels)
neuron_labels(c::Component) = indices_with_labels(length(c.neurons), c.neuron_labels)
component_labels(c::Component) = indices_with_labels(length(c.components), c.components.labels)

# normal getters
input_dΦ(c::Component) = c.input
output_dΦ(c::Component) = c.output
components(c::Component) = c.components
neurons(c::Component) = c.neurons
weights(c::Component) = c.weights
weights_trainable_mask(c::Component) = c.weights_trainable_mask

function add_neurons!(c::Component, names::Vector{String}; args...)
    curr_neurons = length(neurons(c))
    add_neurons!(neurons(c), length(names); args...)
    for i in curr_neurons:(curr_neurons+length(names))
        name = names[i+1-curr_neurons]
        @assert !haskey(c.neuron_labels, name)
        c.neuron_labels[name] = i+1
    end
    build_weights_matrix!(c)
end

function add_neurons!(c::Component, n::Int; args...)
    add_neurons!(neurons(c), n; args...)
    build_weights_matrix!(c)
end

function add_neuron!(c::Component, name::String; args...)
    add_neurons!(neurons(c), 1; args...)
    @assert !haskey(c.neuron_labels, name) "Neuron with name $name already exists!"
    c.neuron_labels[name] = length(neurons(c))
    build_weights_matrix!(c)
end

function add_neuron!(c::Component; args...)
    add_neurons!(neurons(c), 1; args...)
    build_weights_matrix!(c)
end

function modify_neuron!(c::Component, label::Union{String, Int}; args...)
    @assert haskey(c.neuron_labels, label) "Neuron with label $label does not exist!"
    neuron_index = c.neuron_labels[label]
    modify_neuron!(neurons(c), neuron_index; args...)
end

function remove_neuron!(c::Component, name::String)
    @assert haskey(c.neuron_labels, name) "Neuron with name $name does not exist!"
    neuron_index = c.neuron_labels[name]
    remove_neuron!(neurons(c), neuron_index)
    delete!(c.neuron_labels, name)
    # remake neuron_labels so that indices above the removed neuron are shifted down by 1
    new_neuron_labels = Dict{String, Int}()
    for (name, index) in c.neuron_labels
        if index > neuron_index
            new_neuron_labels[name] = index - 1
        else
            new_neuron_labels[name] = index
        end
    end
    c.neuron_labels = new_neuron_labels
    build_weights_matrix!(c)
end

"""
    add_component!(parent::Component, child::Component, name::String)

Inserts `child` into `parent` and makes `child` addressable by `name` via `parent[name]`.
This function currently rebuilds the weight matrices, so make sure to assign weights after calling this function.
"""
function add_component!(parent::Component, child::Component, name::String)
    @assert !haslabel(parent.components, name)
    push_and_label!(parent.components, child, name)
    build_weights_matrix!(parent)
    nothing
end

"""
    add_component!(parent::Component, child::Component)

Inserts `child` into `parent` and makes `child` addressable by its index via `parent[index]`.
This function currently rebuilds the weight matrices, so make sure to assign weights after calling this function.
"""
function add_component!(parent::Component, child::Component)
    push!(parent.components, child)
    build_weights_matrix!(parent)
    nothing
end

function remove_component!(parent::Component, name::String)
    @assert haslabel(parent.components, name)
    remove!(parent.components, name)
    build_weights_matrix!(parent)
    nothing
end

function remove_component!(parent::Component, index::Int)
    remove!(parent.components, index)
    build_weights_matrix!(parent)
    nothing
end

"""
    Base.getindex(c::Component, key_or_index)

Get a child component. E.g. c["a"] returns the component named "a". c[3] returns the 3rd component, if it doesn't have a name.
These can be chained together to naviagte through the component tree.
"""
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

function set_weights!(c::Component, weights::AbstractMatrix{Float64})
    @assert size(weights) == size(raw(weights(c)))
    set_raw!(weights(c), weights)
    nothing
end

function set_weight_trainable!(c::Component, source::Label, destination::Label, trainable::Bool)
    c.weights_trainable_mask[destination, source] = trainable
    nothing
end

function set_weights_trainable!(c::Component, sources::Vector{Label}, destinations::Vector{Label}, trainable::AbstractMatrix{Bool})
    for i in 1:length(sources)
        source = sources[i]
        for j in 1:length(destinations)
            destination = destinations[j]
            c.weights_trainable_mask[destination, source] = trainable[j, i]
        end
    end
    nothing
end

# computes the total number of sources inside this component. represents the number of cols in weights matrix
function sources_length(c::Component)::Int
    curr_length = input_length(c)
    curr_length += length(c.neurons)
    for sc::Component in c.components.vector
        curr_length += output_length(sc)
    end
    curr_length
end

# computes the total number of destinations inside this component. represents the number of rows in weights matrix
function destinations_length(c::Component)::Int
    curr_length = output_length(c)
    curr_length += length(c.neurons)
    for sc::Component in c.components.vector
        curr_length += input_length(sc)
    end
    curr_length
end

function sources(c::Component)::Vector{Label}
    sources = Vector{Label}()
    for i in input_labels(c)
        push!(sources, i)
    end
    for n in neuron_labels(c)
        push!(sources, (n,))
    end

    for clabel in indices_with_labels(c.components)
        for output in output_labels(c[clabel])
            push!(sources, (clabel, output))
        end
    end
    sources
end

function destinations(c::Component)::Vector{Label}
    destinations = Vector{Label}()
    for i in output_labels(c)
        push!(destinations, i)
    end
    for n in neuron_labels(c)
        push!(destinations, (n,))
    end

    for clabel in indices_with_labels(c.components)
        for input in input_labels(c[clabel])
            push!(destinations, (clabel, input))
        end
    end
    destinations
end

function build_weights_matrix!(c::Component)
    weights_old = c.weights
    weights_trainable_mask_old = c.weights_trainable_mask
    int_dest_len = destinations_length(c)
    int_src_len = sources_length(c)
    c.weights = LabeledMatrix{Float64, Label}(zeros(Float64, int_dest_len, int_src_len))
    c.weights_trainable_mask = LabeledMatrix{Bool, Label}(zeros(Bool, int_dest_len, int_src_len))
    dest = destinations(c)
    src = sources(c)
    set_labels!(c.weights, dest, src)
    set_labels!(c.weights_trainable_mask, dest, src)

    # re-apply old weights and trainable mask
    for p in nonzero_pairs(weights_old)
        # only copy over weights that still exist
        if hasindex(c.weights, p[1]...)
            c.weights[p[1]...] = p[2]
        end
    end
    for p in nonzero_pairs(weights_trainable_mask_old)
        if hasindex(c.weights_trainable_mask, p[1]...)
            c.weights_trainable_mask[p[1]...] = p[2]
        end
    end
    nothing
end
