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
    # nodes = Dict{Node, Node}()
    nodes = Vector{Node}()

    for input in input_labels(comp)
        new_node = Node(Path(), input, :root_input)
        # nodes[new_node] = new_node
        push!(nodes, new_node)
    end

    for output in output_labels(comp)
        new_node = Node(Path(), output, :root_output)
        # nodes[new_node] = new_node
        push!(nodes, new_node)
    end

    make_all_qualified_nodes_sublevel!(comp, nodes, Path())
    nodes
end

# Top level function for creating weights from a component tree and nodes
# nodes can be dict or vector
function make_weights_from_component_tree(root::Component, nodes, T::DataType)
    # weight_dict = Dict{Weight{T}, Weight{T}}()
    weight_vector = Vector{Weight{T}}()
    make_weights_sublevel!(root, nodes, weight_vector, Path(), T, true)
end

# Top level function for substituting all internal io nodes in the graph
function substitute_internal_io!(graph::Graph)
    # create node to incoming weights and node to outgoing weights dicts
    node_to_incoming_weights = Dict{Node, Vector{Weight{Float64}}}()
    node_to_outgoing_weights = Dict{Node, Vector{Weight{Float64}}}()

    for weight in weights_vector(graph)
        if !haskey(node_to_outgoing_weights, from(weight))
            node_to_outgoing_weights[from(weight)] = Vector{Weight{Float64}}()
        end

        push!(node_to_outgoing_weights[from(weight)], weight)

        if !haskey(node_to_incoming_weights, to(weight))
            node_to_incoming_weights[to(weight)] = Vector{Weight{Float64}}()
        end

        push!(node_to_incoming_weights[to(weight)], weight)
    end

    to_subs = filter(x->(type(x) == :input || type(x) == :output), nodes_vector(graph))
    weights_dict = make_weights_dict(graph)
    nodes_dict = make_nodes_dict(graph)
    for to_sub in to_subs
        substitute_node!(weights_dict, nodes_dict, to_sub, node_to_outgoing_weights, node_to_incoming_weights)
    end

    set_nodes_vector!(graph, [x for x in values(nodes_dict)])
    set_weights_vector!(graph, [x for x in values(weights_dict)])
    graph
end

sparsity(x) = Float64(count(iszero, x)) / length(x)

# Populates a labeled weight matrix with the weights of the reduced graph
function reduced_graph_to_labeled_matrix(graph::Graph, sparse_=true, gpu=false)
    # node_vec = keys(nodes(graph))
    # weights_vec = keys(weights(graph))
    node_vec = nodes_vector(graph)
    weights_vec = weights_vector(graph)
    
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

# Private method for recursively populating nodes from a component tree, excluding the top level
function make_all_qualified_nodes_sublevel!(comp::Component, nodes, current_path::Path)
    # add inputs, outputs, and neurons as nodes, with current_path
    # call this function for every component, and append that component's name to current_path

    for neuron in neuron_labels(comp)
        new_node = Node(copy(current_path), neuron, :neuron)
        # nodes[new_node] = new_node
        push!(nodes, new_node)
    end

    for clabel in component_labels(comp)
        c = comp[clabel]
        for input in input_labels(c)
            new_node = Node(vcat(copy(current_path), clabel), input, :input)
            # nodes[new_node] = new_node
            push!(nodes, new_node)
        end
        for output in output_labels(c)
            new_node = Node(vcat(copy(current_path), clabel), output, :output)
            # nodes[new_node] = new_node
            push!(nodes, new_node)
        end
        make_all_qualified_nodes_sublevel!(c, nodes, vcat(current_path, clabel))
    end

    nothing
end

# Private method for recursively populating weights from a component tree, excluding the top level
# populates weight_dict
function make_weights_sublevel!(comp::Component, nodes, weight_vec, current_path::Path, T::DataType, is_root::Bool=false)
    for p in nonzero_pairs(weights(comp))
        dest_node = get_node_from_label(nodes, p[1][1], false, current_path, is_root)
        src_node = get_node_from_label(nodes, p[1][2], true, current_path, is_root)
        new_weight = Weight(convert(T, p[2]), src_node, dest_node)
        # weight_vec[new_weight] = new_weight
        push!(weight_vec, new_weight)
    end
    for clabel in component_labels(comp)
        c = comp[clabel]
        make_weights_sublevel!(c, nodes, weight_vec, vcat(current_path, clabel), T)
    end
    weight_vec
end

# Private method.
# Given the node to_sub, all incoming and outgoing weights will be replaced 
# by the product of each incoming weight times each outgoing weight.
# The resulting weight configuration is equivalent.
function substitute_node!(weights_dict, nodes_dict, to_sub::Node, node_to_outgoing_weights::Dict{Node, Vector{Weight}}, node_to_incoming_weights::Dict{Node, Vector{Weight}})
    
    # incoming = incoming_weights(weights(graph), to_sub)
    # outgoing = outgoing_weights(weights(graph), to_sub)
    incoming = node_to_incoming_weights[to_sub]
    outgoing = node_to_outgoing_weights[to_sub]

    for i in incoming
        for o in outgoing
            @assert from(i) != to(i)
            new_weight = Weight(weight(i) * weight(o), from(i), to(o))
            # weights(graph)[new_weight] = new_weight
            # push!(weights_vector(graph), new_weight)
            weights_dict[new_weight] = new_weight
            # populate node_to_outgoing_weights and node_to_incoming_weights
            push!(node_to_outgoing_weights[from(i)], new_weight)
            push!(node_to_incoming_weights[to(o)], new_weight)

        end
    end

    for i in incoming
        delete!(weights_dict, i)
    end
    for o in outgoing
        delete!(weights_dict, o)
    end

    delete!(nodes_dict, to_sub)
    delete!(node_to_outgoing_weights, to_sub)
    delete!(node_to_incoming_weights, to_sub)
    nothing
end

