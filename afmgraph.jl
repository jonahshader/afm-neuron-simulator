include("afmneuron_rewritten.jl")
include("labeledmatrix.jl")
include("utils.jl")

using AutoHashEquals
using LinearAlgebra
using SparseArrays

const Path = Vector{Union{String, Int}}
@auto_hash_equals struct Node
    path::Path
    name::Union{String, Int}
    type::Symbol
end

struct Weight{T<:AbstractFloat}
    weight::T
    from::Node
    to::Node
end

mutable struct Graph{T<:AbstractFloat}
    nodes::Vector{Node}
    weights::Vector{Weight{T}}
end



function make_nodes_from_component_tree(root::Component)
    root_inputs = Vector{Node}()
    root_outputs = Vector{Node}()

    for input in inputs(root)
        push!(root_inputs, Node(Path(), input, :root_input))
    end

    for output in outputs(root)
        push!(root_outputs, Node(Path(), output, :root_output))
    end

    nodes = make_all_qualified_nodes(root, Path())

    vcat(root_inputs, root_outputs, nodes)
end

function make_all_qualified_nodes(comp::Component, current_path::Path)
    nodes = Vector{Node}()
    # add inputs, outputs, and neurons as nodes, with current_path
    # call this function for every component, and append that component's name to current_path

    # nodes = vcat(nodes, input_nodes, output_nodes) # this is in the wrong spot
    for neuron in neurons(comp)
        push!(nodes, Node(copy(current_path), neuron, :neuron))
    end

    for clabel in components(comp)
        c = comp[clabel]
        comp_input_nodes = Vector{Node}()
        comp_output_nodes = Vector{Node}()
        for input in inputs(c)
            push!(comp_input_nodes, Node(vcat(copy(current_path), clabel), input, :input))
        end
        for output in outputs(c)
            push!(comp_output_nodes, Node(vcat(copy(current_path), clabel), output, :output))
        end
        nodes = vcat(nodes, comp_input_nodes, comp_output_nodes)
        next_qualified_nodes = make_all_qualified_nodes(c, vcat(current_path, clabel))

        nodes = vcat(nodes, next_qualified_nodes)

        # # generate weights from root.weights
        # for p in nonzero_pairs(comp.weights)
        #     dest_node = get_node_from_label(nodes, p[1][1], false, current_path, is_root)
        #     src_node = get_node_from_label(nodes, p[1][2], false, current_path, is_root)
        #     push!(weights, Weight(p[2], src_node, dest_node))
        # end
    end

    nodes
end

# function make_neuron_qualified_nodes(comp::Component, current_path::Path)
#     nodes = Vector{Node}()

#     for neuron in neurons(comp)
#         push!(nodes, Node(copy(current_path), neuron, :neuron))
#     end

#     for clabel in components(comp)
#         c = comp[clabel]
#         next_qualified_nodes = make_neuron_qualified_nodes(c, vcat(current_path, clabel))
#         nodes = vcat(nodes, next_qualified_nodes)
#     end
#     nodes
# end


function make_weights_from_component_tree(root::Component, nodes::Vector{Node})
    make_weights(root, nodes, Path(), true)
end

function make_weights(comp::Component, nodes::Vector{Node}, current_path::Path, is_root::Bool=false)
    weights = Vector{Weight}()
    for p in nonzero_pairs(comp.weights)
        dest_node = get_node_from_label(nodes, p[1][1], false, current_path, is_root)
        src_node = get_node_from_label(nodes, p[1][2], true, current_path, is_root)
        push!(weights, Weight(p[2], src_node, dest_node))
    end
    for clabel in components(comp)
        c = comp[clabel]
        weights = vcat(weights, make_weights(c, nodes, vcat(current_path, clabel)))
    end
    weights
end

# ComponentLabel to Node
function label_to_node(label::ComponentLabel, source::Bool, path::Path, is_root::Bool=false)
    if is_root
        Node(path, label, source ? :root_input : :root_output)
    else
        Node(path, label, source ? :input : :output)
    end
end
# NeuronLabel to Node
function label_to_node(label::NeuronLabel, source::Bool, path::Path, is_root::Bool=false)
    Node(path, label[1], :neuron)
end
# SubComponentLabel to Node
function label_to_node(label::SubComponentLabel, source::Bool, path::Path, is_root::Bool=false)
    Node(vcat(path, label[1]), label[2], source ? :output : :input)
end

function get_node_from_label(nodes::Vector{Node}, label::Label, source::Bool, path::Path, is_root::Bool=false)
    target_node = label_to_node(label, source, path, is_root)
    for node in nodes
        if node == target_node
            return node
        end
    end
    error("Could not find node $target_node from the nodes $nodes")
    target_node # this is just to make the compiler happy
end

# function labeled_matrix_pair_to_weight(pair::Tuple{})
# function labeled_matrix_pair_to_weight(pair_and_value::Tuple{Tuple{}})

# TODO: modify make functions here to generate weights in addition to nodes
# TODO: add UUIDs to components


function incoming_weights(weights::Vector{Weight}, node::Node)
    incoming = Vector{Weight}()
    for weight in weights
        if weight.to == node
            push!(incoming, weight)
        end
    end
    incoming
end

function outgoing_weights(weights::Vector{Weight}, node::Node)
    outgoing = Vector{Weight}()
    for weight in weights
        if weight.from == node
            push!(outgoing, weight)
        end
    end
    outgoing
end

function substitute_node!(weights::Vector{Weight}, nodes::Vector{Node}, to_sub::Node)
    incoming = incoming_weights(weights, to_sub)
    outgoing = outgoing_weights(weights, to_sub)

    for i in incoming
        for o in outgoing
            @assert i.from != i.to
            push!(weights, Weight(i.weight * o.weight, i.from, o.to))
        end
    end

    for i in incoming
        deleteat!(weights, findall(x->x == i, weights))
    end
    for o in outgoing
        deleteat!(weights, findall(x->x == o, weights))
    end

    deleteat!(nodes, findall(x->x == to_sub, nodes))
    nothing
end

function substitute_internal_io!(weights::Vector{Weight}, nodes::Vector{Node})
    to_subs = filter(x->(x.type == :input || x.type == :output), nodes)
    for to_sub in to_subs
        substitute_node!(weights, nodes, to_sub)
    end
    nothing
end

function graph_to_labeled_matrix(weights::Vector{Weight}, nodes::Vector{Node})
    neuron_nodes = filter(x->x.type == :neuron, nodes)
    root_input_nodes = filter(x->x.type == :root_input, nodes)
    root_output_nodes = filter(x->x.type == :root_output, nodes)

    neuron_to_neuron_matrix = LabeledMatrix{Float64, Node}(sparse(zeros(Float64, length(neuron_nodes), length(neuron_nodes))))
    root_input_to_neuron_matrix = LabeledMatrix{Float64, Node}(sparse(zeros(Float64, length(neuron_nodes), length(root_input_nodes))))
    neuron_to_root_output_matrix = LabeledMatrix{Float64, Node}(sparse(zeros(Float64, length(root_output_nodes), length(neuron_nodes))))
    root_input_to_root_output_matrix = LabeledMatrix{Float64, Node}(sparse(zeros(Float64, length(root_output_nodes), length(root_input_nodes))))
    set_labels!(neuron_to_neuron_matrix, neuron_nodes, neuron_nodes)
    set_labels!(root_input_to_neuron_matrix, neuron_nodes, root_input_nodes)
    set_labels!(neuron_to_root_output_matrix, root_output_nodes, neuron_nodes)
    set_labels!(root_input_to_root_output_matrix, root_output_nodes, root_input_nodes)

    n_to_n_weights = filter(x->(x.from.type == :neuron && x.to.type == :neuron), weights)
    for w in n_to_n_weights
        neuron_to_neuron_matrix[w.to, w.from] = w.weight
    end
    root_input_to_n_weights = filter(x->(x.from.type == :root_input && x.to.type == :neuron), weights)
    for w in root_input_to_n_weights
        root_input_to_neuron_matrix[w.to, w.from] = w.weight
    end
    n_to_root_output_weights = filter(x->(x.from.type == :neuron && x.to.type == :root_output), weights)
    for w in n_to_root_output_weights
        neuron_to_root_output_matrix[w.to, w.from] = w.weight
    end
    root_input_to_root_output_weights = filter(x->(x.from.type == :root_input && x.to.type == :root_output), weights)
    for w in root_input_to_root_output_weights
        root_input_to_root_output_matrix[w.to, w.from] = w.weight
    end

    neuron_to_neuron_matrix, root_input_to_neuron_matrix, neuron_to_root_output_matrix, root_input_to_root_output_matrix
end