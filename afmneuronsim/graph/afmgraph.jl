include("node.jl")
include("weight.jl")

using LinearAlgebra
using SparseArrays

mutable struct Graph{T<:AbstractFloat}
    # nodes::Vector{Node}
    nodes::Dict{Node, Node}
    # weights::Vector{Weight{T}}
    weights::Dict{Weight{T}, Weight{T}}
end

nodes(graph::Graph) = graph.nodes
weights(graph::Graph) = graph.weights

