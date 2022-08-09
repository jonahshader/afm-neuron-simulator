include("afmgraph.jl")

using SparseArrays
using CUDA
using CUDA.CUSPARSE

const SPARSITY_THRESHOLD = 0.9

function Graph{T}(comp::Component) where {T<:AbstractFloat}
    nodes = make_nodes(comp)
    weights = make_weights_from_component_tree(comp, nodes, T)
    Graph{T}(nodes, weights)
end

# Top level function to create all nodes from a component tree
function make_nodes(comp::Component)
    root_inputs = Dict{Node, Node}()
    root_outputs = Dict{Node, Node}()
    # root_inputs = Vector{Node}()
    # root_outputs = Vector{Node}()

    for input in input_labels(comp)
        new_node = Node(Path(), input, :root_input)
        root_inputs[new_node] = new_node
        # push!(root_inputs, Node(Path(), input, :root_input))
    end

    for output in output_labels(comp)
        new_node = Node(Path(), output, :root_output)
        root_outputs[new_node] = new_node
        # push!(root_outputs, Node(Path(), output, :root_output))
    end

    nodes = make_all_qualified_nodes_sublevel(comp, Path())

    # vcat(root_inputs, root_outputs, nodes)
    merge(root_inputs, root_outputs, nodes)
end

# Top level function for creating weights from a component tree and nodes
# nodes can be dict or vector
function make_weights_from_component_tree(root::Component, nodes, T::DataType)
    weight_dict = Dict{Weight{T}, Weight{T}}()
    make_weights_sublevel(root, nodes, weight_dict, Path(), T, true)
end

# Returns all weights of which the destination is the specified node.
function incoming_weights(weights::Vector{Weight{T}}, node::Node) where {T<:AbstractFloat}
    incoming = Vector{Weight{T}}()
    for weight in weights
        if to(weight) == node
            push!(incoming, weight)
        end
    end
    incoming
end

# Returns all weights of which the source is the specified node.
function outgoing_weights(weights::Vector{Weight{T}}, node::Node) where {T<:AbstractFloat}
    outgoing = Vector{Weight{T}}()
    for weight in weights
        if from(weight) == node
            push!(outgoing, weight)
        end
    end
    outgoing
end

# Top level function for substituting all internal io nodes in the graph
function substitute_internal_io!(graph::Graph)
    # create node to incoming weights and node to outgoing weights dicts
    node_to_incoming_weights = Dict{Node, Vector{Weight{Float64}}}()
    node_to_outgoing_weights = Dict{Node, Vector{Weight{Float64}}}()
    for weight in values(weights(graph))
        # if !(from(weight) in node_to_outgoing_weights)
        if !haskey(node_to_outgoing_weights, from(weight))
            node_to_outgoing_weights[from(weight)] = Vector{Weight{Float64}}()
        end

        push!(node_to_outgoing_weights[from(weight)], weight)

        # if !(to(weight) in node_to_incoming_weights)
        if !haskey(node_to_incoming_weights, to(weight))
            node_to_incoming_weights[to(weight)] = Vector{Weight{Float64}}()
        end

        push!(node_to_incoming_weights[to(weight)], weight)
    end

    to_subs = filter(x->(type(x) == :input || type(x) == :output), keys(nodes(graph)))
    for to_sub in to_subs
        substitute_node!(graph, to_sub, node_to_outgoing_weights, node_to_incoming_weights)
    end
    graph
end

sparsity(x) = Float64(count(iszero, x)) / length(x)

# Populates a labeled weight matrix with the weights of the reduced graph
function reduced_graph_to_labeled_matrix(graph::Graph, sparse_=true, gpu=false)
    # node_vec = [x for x in values(nodes(graph))]
    # weights_vec = [x for x in values(weights(graph))]
    node_vec = values(nodes(graph))
    weights_vec = values(weights(graph))
    
    neuron_nodes = filter(x->type(x) == :neuron, node_vec)
    root_input_nodes = filter(x->type(x) == :root_input, node_vec)
    root_output_nodes = filter(x->type(x) == :root_output, node_vec)

    neuron_to_neuron_matrix_raw = zeros(Float64, length(neuron_nodes), length(neuron_nodes))
    root_input_to_neuron_matrix_raw = zeros(Float64, length(neuron_nodes), length(root_input_nodes))
    neuron_to_root_output_matrix_raw = zeros(Float64, length(root_output_nodes), length(neuron_nodes))
    root_input_to_root_output_matrix_raw = zeros(Float64, length(root_output_nodes), length(root_input_nodes))

    neuron_to_neuron_matrix = LabeledMatrix{Float64, Node}(neuron_to_neuron_matrix_raw)
    root_input_to_neuron_matrix = LabeledMatrix{Float64, Node}(root_input_to_neuron_matrix_raw)
    neuron_to_root_output_matrix = LabeledMatrix{Float64, Node}(neuron_to_root_output_matrix_raw)
    root_input_to_root_output_matrix = LabeledMatrix{Float64, Node}(root_input_to_root_output_matrix_raw)

    set_labels!(neuron_to_neuron_matrix, neuron_nodes, neuron_nodes)
    set_labels!(root_input_to_neuron_matrix, neuron_nodes, root_input_nodes)
    set_labels!(neuron_to_root_output_matrix, root_output_nodes, neuron_nodes)
    set_labels!(root_input_to_root_output_matrix, root_output_nodes, root_input_nodes)

    n_to_n_weights = filter(x->(type(from(x)) == :neuron && type(to(x)) == :neuron), weights_vec)
    for w in n_to_n_weights
        neuron_to_neuron_matrix[to(w), from(w)] += weight(w)
    end
    root_input_to_n_weights = filter(x->(type(from(x)) == :root_input && type(to(x)) == :neuron), weights_vec)
    for w in root_input_to_n_weights
        root_input_to_neuron_matrix[to(w), from(w)] += weight(w)
    end
    n_to_root_output_weights = filter(x->(type(from(x)) == :neuron && type(to(x)) == :root_output), weights_vec)
    for w in n_to_root_output_weights
        neuron_to_root_output_matrix[to(w), from(w)] += weight(w)
    end
    root_input_to_root_output_weights = filter(x->(type(from(x)) == :root_input && type(to(x)) == :root_output), weights_vec)
    for w in root_input_to_root_output_weights
        root_input_to_root_output_matrix[to(w), from(w)] += weight(w)
    end

    if isnothing(sparse_)
        if sparsity(neuron_to_neuron_matrix_raw) > SPARSITY_THRESHOLD
            neuron_to_neuron_matrix_raw = sparse(neuron_to_neuron_matrix_raw)
            dropzeros!(neuron_to_neuron_matrix_raw)
            # println("nnm autodetected as sparse")
        end
        if sparsity(root_input_to_neuron_matrix_raw) > SPARSITY_THRESHOLD
            root_input_to_neuron_matrix_raw = sparse(root_input_to_neuron_matrix_raw)
            dropzeros!(root_input_to_neuron_matrix_raw)
            # println("inm autodetected as sparse")
        end
        if sparsity(neuron_to_root_output_matrix_raw) > SPARSITY_THRESHOLD
            neuron_to_root_output_matrix_raw = sparse(neuron_to_root_output_matrix_raw)
            dropzeros!(neuron_to_root_output_matrix_raw)
            # println("nom autodetected as sparse")
        end
        if sparsity(root_input_to_root_output_matrix_raw) > SPARSITY_THRESHOLD
            root_input_to_root_output_matrix_raw = sparse(root_input_to_root_output_matrix_raw)
            dropzeros!(root_input_to_root_output_matrix_raw)
            # println("iom autodetected as sparse")
        end
    elseif sparse_
        neuron_to_neuron_matrix_raw = sparse(neuron_to_neuron_matrix_raw)
        root_input_to_neuron_matrix_raw = sparse(root_input_to_neuron_matrix_raw)
        neuron_to_root_output_matrix_raw = sparse(neuron_to_root_output_matrix_raw)
        root_input_to_root_output_matrix_raw = sparse(root_input_to_root_output_matrix_raw)

        dropzeros!(neuron_to_neuron_matrix_raw)
        dropzeros!(root_input_to_neuron_matrix_raw)
        dropzeros!(neuron_to_root_output_matrix_raw)
        dropzeros!(root_input_to_root_output_matrix_raw)
    end

    if gpu
        neuron_to_neuron_matrix_raw = cu(neuron_to_neuron_matrix_raw)
        root_input_to_neuron_matrix_raw = cu(root_input_to_neuron_matrix_raw)
        neuron_to_root_output_matrix_raw = cu(neuron_to_root_output_matrix_raw)
        root_input_to_root_output_matrix_raw = cu(root_input_to_root_output_matrix_raw)
    end

    # since these are references to the matrices, we need to reassign the raw matrices of the labeled matrices
    set_raw!(neuron_to_neuron_matrix, neuron_to_neuron_matrix_raw)
    set_raw!(root_input_to_neuron_matrix, root_input_to_neuron_matrix_raw)
    set_raw!(neuron_to_root_output_matrix, neuron_to_root_output_matrix_raw)
    set_raw!(root_input_to_root_output_matrix, root_input_to_root_output_matrix_raw)

    neuron_to_neuron_matrix, root_input_to_neuron_matrix, neuron_to_root_output_matrix, root_input_to_root_output_matrix
end

# Private method for recursively creating nodes from a component tree, excluding the top level
function make_all_qualified_nodes_sublevel(comp::Component, current_path::Path)
    nodes = Dict{Node, Node}()
    # nodes = Vector{Node}()
    # add inputs, outputs, and neurons as nodes, with current_path
    # call this function for every component, and append that component's name to current_path

    # nodes = vcat(nodes, input_nodes, output_nodes) # this is in the wrong spot
    for neuron in neuron_labels(comp)
        new_node = Node(copy(current_path), neuron, :neuron)
        nodes[new_node] = new_node
        # push!(nodes, Node(copy(current_path), neuron, :neuron))
    end

    for clabel in component_labels(comp)
        c = comp[clabel]
        comp_input_nodes = Dict{Node, Node}()
        comp_output_nodes = Dict{Node, Node}()
        # comp_input_nodes = Vector{Node}()
        # comp_output_nodes = Vector{Node}()
        for input in input_labels(c)
            new_node = Node(vcat(copy(current_path), clabel), input, :input)
            comp_input_nodes[new_node] = new_node
            # push!(comp_input_nodes, Node(vcat(copy(current_path), clabel), input, :input))
        end
        for output in output_labels(c)
            new_node = Node(vcat(copy(current_path), clabel), output, :output)
            comp_output_nodes[new_node] = new_node
            # push!(comp_output_nodes, Node(vcat(copy(current_path), clabel), output, :output))
        end
        # nodes = vcat(nodes, comp_input_nodes, comp_output_nodes)
        # nodes = merge(nodes, comp_input_nodes, comp_output_nodes)
        merge!(nodes, comp_input_nodes, comp_output_nodes)
        next_qualified_nodes = make_all_qualified_nodes_sublevel(c, vcat(current_path, clabel))

        # nodes = vcat(nodes, next_qualified_nodes)
        # nodes = merge(nodes, next_qualified_nodes)
        merge!(nodes, next_qualified_nodes)
    end

    nodes
end

# Private method for recursively creating weights from a component tree, excluding the top level
# nodes can be dict or vector
function make_weights_sublevel(comp::Component, nodes, weight_dict, current_path::Path, T::DataType, is_root::Bool=false)
    # weight_list = Vector{Weight{T}}()
    # weight_dict = Dict{Weight{T}, Weight{T}}()
    for p in nonzero_pairs(weights(comp))
        dest_node = get_node_from_label(nodes, p[1][1], false, current_path, is_root)
        src_node = get_node_from_label(nodes, p[1][2], true, current_path, is_root)
        # push!(weight_list, Weight(convert(T, p[2]), src_node, dest_node))
        new_weight = Weight(convert(T, p[2]), src_node, dest_node)
        weight_dict[new_weight] = new_weight
    end
    for clabel in component_labels(comp)
        c = comp[clabel]
        # weight_list = vcat(weight_list, make_weights_sublevel(c, nodes, vcat(current_path, clabel), T))
        merge!(weight_dict, make_weights_sublevel(c, nodes, weight_dict, vcat(current_path, clabel), T))
    end
    # weight_list
    weight_dict
end

# # Private method.
# # Given the node to_sub, all incoming and outgoing weights will be replaced 
# # by the product of each incoming weight times each outgoing weight.
# # The resulting weight configuration is equivalent.
# function substitute_node!(graph::Graph, to_sub::Node)
    
#     incoming = incoming_weights(weights(graph), to_sub)
#     outgoing = outgoing_weights(weights(graph), to_sub)

#     for i in incoming
#         for o in outgoing
#             @assert from(i) != to(i)
#             push!(weights(graph), Weight(weight(i) * weight(o), from(i), to(o)))
#         end
#     end

#     for i in incoming
#         deleteat!(weights(graph), findall(x->x == i, weights(graph)))
#     end
#     for o in outgoing
#         deleteat!(weights(graph), findall(x->x == o, weights(graph)))
#     end

#     deleteat!(nodes(graph), findall(x->x == to_sub, nodes(graph)))
#     nothing
# end

# Private method.
# Same as above, but uses sets instead of vectors to achieve O(n) time instead of O(n^2) time.
# function substitute_node!(weights::Set{Weight}, nodes::Set{Node}, to_sub::Node)


# TODO: this method will require two dicts: one that maps node to outgoing weights, and one that maps node to incoming weights.
# first create a mapping from each node to an empty vector of weights
# iterate through each weight and check if its source is equal to the node. if it is, add the weight to the vector of weights
# same can be done for dict that maps node to outgoing weights
# this will be done in linear time! :D
function substitute_node!(graph::Graph, to_sub::Node, node_to_outgoing_weights::Dict{Node, Vector{Weight}}, node_to_incoming_weights::Dict{Node, Vector{Weight}})
    
    # incoming = incoming_weights(weights(graph), to_sub)
    # outgoing = outgoing_weights(weights(graph), to_sub)
    incoming = node_to_incoming_weights[to_sub]
    outgoing = node_to_outgoing_weights[to_sub]

    for i in incoming
        for o in outgoing
            @assert from(i) != to(i)
            # push!(weights(graph), Weight(weight(i) * weight(o), from(i), to(o)))
            new_weight = Weight(weight(i) * weight(o), from(i), to(o))
            weights(graph)[new_weight] = new_weight
            # populate node_to_outgoing_weights and node_to_incoming_weights
            push!(node_to_outgoing_weights[from(i)], new_weight)
            push!(node_to_incoming_weights[to(o)], new_weight)

        end
    end

    for i in incoming
        # deleteat!(weights(graph), findall(x->x == i, weights(graph)))
        delete!(weights(graph), i)
    end
    for o in outgoing
        # deleteat!(weights(graph), findall(x->x == o, weights(graph)))
        delete!(weights(graph), o)
    end

    delete!(nodes(graph), to_sub)
    delete!(node_to_outgoing_weights, to_sub)
    delete!(node_to_incoming_weights, to_sub)

    # deleteat!(nodes(graph), findall(x->x == to_sub, nodes(graph)))
    nothing
end

