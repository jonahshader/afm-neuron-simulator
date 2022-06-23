include("../afmcomponent.jl")

using AutoHashEquals

const Path = Vector{Union{String, Int}}
@auto_hash_equals struct Node
    path::Path
    name::Union{String, Int}
    type::Symbol
end

path(node::Node) = node.path
name(node::Node) = node.name
type(node::Node) = node.type

function set_name!(node::Node, name)
    node.name = name
end

function set_type!(node::Node, type)
    node.type = type
end

# Creates a string representation of a node, with the path and name.
function node_str(node::Node)
    str = ""
    for p in path(node)
        str *= "[" * string(p) * "]"
    end
    str *= string(name(node))
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

# Looks for a node with the given label in the vector of nodes.
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

