const Label = Union{String, Int, Tuple{Int}, Tuple{String}, Tuple{Int, Int}, Tuple{Int, String}, Tuple{String, Int}, Tuple{String, String}}

include("Neurons.jl")

mutable struct RuntimeNeurons
    sigma::Vector{Float64}
    a::Vector{Float64}
    we::Vector{Float64}
    wex::Vector{Float64}
    beta::Vector{Float64}
    bias::Vector{Float64}
end

mutable struct Component
    input_length::Int
    input_labels::Dict{String, Int}

    output_length::Int
    output_labels::Dict{String, Int}

    components::Vector{Component}
    component_labels::Dict{String, Int}

    neurons::Neurons
    neuron_labels::Dict{String, Int}

    weights::AbstractMatrix{Float64}
    all_sources::Dict{Label, Int}
    all_destinations::Dict{Label, Int}
end

# outer constructors
function Component(input_length::Int, output_length::Int)::Component
    comp = Component(
        input_length, Dict{String, Int}(),
        output_length, Dict{String, Int}(),
        Vector{Component}(), Dict{String, Int}(),
        Neurons(), Dict{String, Int}(),
        zeros(Float64, 1, 1),
        Dict{Label, Int}(),
        Dict{Label, Int}())
    build_weights_matrix!(comp)
    build_source_dest!(comp)
    comp
end

function Component(input_length::Int, outputs::Vector{String})::Component
    output_length = length(outputs)
    comp = Component(input_length, output_length)
    # pass inputs to input labels
    for i in 1:output_length
        comp.output_labels[outputs[i]] = i
    end
    comp
end

function Component(inputs::Vector{String}, output_length::Int)::Component
    input_length = length(inputs)
    comp = Component(input_length, output_length)
    # pass outputs to output labels
    for i in 1:input_length
        comp.input_labels[inputs[i]] = i
    end
    comp
end

function Component(inputs::Vector{String}, outputs::Vector{String})::Component
    input_length = length(inputs)
    output_length = length(outputs)
    comp = Component(input_length, output_length)
    # pass inputs and outputs to labels
    for i in 1:input_length
        comp.input_labels[inputs[i]] = i
    end
    for i in 1:output_length
        comp.output_labels[outputs[i]] = i
    end
    comp
end

input_length(c::Component) = c.input_length
output_length(c::Component) = c.output_length

# these methods return arrays of labels, which can be used for indexing
# they return Vector{String, Int}, so the user needs to convert them to the appropriate format
# when using them later for referencing. E.g. neurons are referenced with the type Tuple{Union{String, Int}}
# and components are referenced with the type Tuple{Union{String, Int}, Union{String, Int}}
inputs(c::Component) = indices_with_labels(c.input_length, c.input_labels)
outputs(c::Component) = indices_with_labels(c.output_length, c.output_labels)
neurons(c::Component) = indices_with_labels(length(c.neurons), c.neuron_labels)
components(c::Component) = indices_with_labels(length(c.components), c.component_labels)


# adds a child component to parent, with a name
function add_component!(parent::Component, child::Component, name::String)
    push!(parent.components, child)
    @assert !haskey(parent.component_labels, name)
    parent.component_labels[name] = length(parent.components)
    build_source_dest!(parent)
    build_weights_matrix!(parent)
    nothing
end

# add a child component to a parent (no name)
function add_component!(parent::Component, child::Component)
    push!(parent.components, child)
    nothing
end

# method for tree navigation of components via component[string] syntax
function Base.getindex(c::Component, key::String)::Component
    return c.components[c.component_labels[key]]
end

# method for tree navigation of components via component[int] syntax
function Base.getindex(c::Component, index::Int)::Component
    return c.components[index]
end

function set_weight!(c::Component, source::Label, destination::Label, weight::AbstractFloat)
    row = c.all_destinations[destination]
    col = c.all_sources[source]
    c.weights[row, col] = weight
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

# next is the build functions. the build function will output an ODEProblem
# sub build functions will be build_p, build u0, build fun


# make unique shallow (only current component is made unique)
# make unique deep (current component and all children component are made unique)


# PRIVATE FUNCTIONS. TODO: move to other file and use here, just don't re export them

# applies f to c tree in depth first order, returns array of results from applying f to every c
function map_component_depth_first(f, c::Component)
    return vcat([f(c)], map(x -> map_component_depth_first(f, x), c.components)...)
end

# same thing as map_component_depth_first, but does not put each result into an element in an array
# instead every result is concatinated into one array
function map_component_array_depth_first(f, c::Component)
    return vcat(f(c), map(x -> map_component_depth_first(f, x), c.components)...)
end

function total_neuron_count(c::Component, current_neuron_count::Int)::Int
    current_neuron_count + length(c.neurons) + reduce(+, c.components; init=0)
end

# p contains weights and runtime neurons
function build_p_matrices(c::Component)
    return map_component_depth_first(comp->comp.weights, c)
end

function build_source_dest!(c::Component)
    c.all_sources = build_label_to_indices_map(sources(c))
    c.all_destinations = build_label_to_indices_map(destinations(c))
    return
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
    curr_length = c.input_length
    curr_length += length(c.neurons)
    for sc::Component in c.components
        curr_length += output_length(sc)
    end
    curr_length
end
# computes the total number of destinations inside this component. represents the number of rows in weights matrix
function internal_destination_length(c::Component)::Int
    curr_length = c.output_length
    curr_length += length(c.neurons)
    for sc::Component in c.components
        curr_length += input_length(sc)
    end
    curr_length
end

function build_weights_matrix!(c::Component)
    rows = internal_destination_length(c)
    cols = internal_source_length(c)
    c.weights = zeros(Float64, rows, cols)
    return
end

# lists all ints in range 1:index_length, but replaces ints with string from str_to_int where possible
function indices_with_labels(index_length::Int, str_to_int::Dict{String, Int})::Vector{Union{String, Int}}
    output = Vector{Union{String, Int}}()
    reversed = Dict{Int, String}(value => key for (key, value) in str_to_int)
    for i in 1:index_length
        if haskey(reversed, i)
            push!(output, reversed[i])
        else
            push!(output, i)
        end
    end
    output
end