include("../afmcomponent.jl")
include("../labeledmatrix.jl")
include("../utils.jl")

include("node.jl")
include("weight.jl")

using AutoHashEquals
using LinearAlgebra
using SparseArrays


mutable struct Graph{T<:AbstractFloat}
    nodes::Vector{Node}
    weights::Vector{Weight{T}}
end

nodes(graph::Graph) = graph.nodes
weights(graph::Graph) = graph.weights

function Graph{T}(comp::Component) where {T<:AbstractFloat}
    nodes = make_nodes(comp)
    weights = make_weights_from_component_tree(comp, nodes)
    Graph{T}(nodes, weights)
end

