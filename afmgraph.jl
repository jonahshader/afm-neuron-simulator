
struct Node
    path::Vector{Union{String, Int}}
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
