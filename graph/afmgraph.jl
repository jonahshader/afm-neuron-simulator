include("../afmcomponent.jl")
include("../labeledmatrix.jl")
include("../utils.jl")

include("node.jl")
include("weight.jl")

using LinearAlgebra
using SparseArrays


mutable struct Graph{T<:AbstractFloat}
    nodes::Vector{Node}
    weights::Vector{Weight{T}}
end

nodes(graph::Graph) = graph.nodes
weights(graph::Graph) = graph.weights

