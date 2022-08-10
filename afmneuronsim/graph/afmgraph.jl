include("node.jl")
include("weight.jl")

using LinearAlgebra
using SparseArrays

mutable struct Graph{T<:AbstractFloat}
    nodes::Vector{Node}
    weights::Vector{Weight{T}}
end

nodes_vector(graph::Graph) = graph.nodes
make_nodes_dict(graph::Graph) = Dict([x=>x for x in nodes_vector(graph)])

weights_vector(graph::Graph) = graph.weights
make_weights_dict(graph::Graph) = Dict([x=>x for x in weights_vector(graph)])

function set_nodes_vector!(graph::Graph, nodes::Vector{Node})
    graph.nodes = nodes
end

function set_weights_vector!(graph::Graph, weights)
    graph.weights = weights
end

