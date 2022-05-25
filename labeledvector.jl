
mutable struct LabeledVector{T, L}
    vector::AbstractVector{T}
    labels::Dict{L, Int}
    labels_reversed::Dict{Int, L}
end

LabeledVector(vector::T) where {T, L} = LabeledVector{T, L}(vector, Dict{L, Int}(), Dict{Int, L}())
function LabeledVector(vector::T, init_labels::AbstractVector{L}) where {T, L}
    v = LabeledVector(vector)
    for i in enumerate(init_labels)
        set_label(v, i...)
    end
end

get_vector(v::LabeledVector) = v.vector

Base.getindex(v::LabeledVector, i1::Int)= v.vector[i1]
Base.getindex(v::LabeledVector{T, L}, i1::L) where {T, L} = v.vector[v.labels[i1]]

function Base.setindex!(v::LabeledVector, x, i1::Int)
    v.vector[i1] = x
end

function Base.setindex(v::LabeledVector{T, L}, x, i1::L) where {T, L}
    v.vector[labels[i1]] = x
end

function get_label(v::LabeledVector, index::Int)
    if haskey(v.labels_reversed, index)
        return v.labels_reversed[index]
    else
        return nothing
    end
end

function get_index(v::LabeledVector{T, L}, label::L) where {T, L}
    if haskey(v.labels, label)
        return v.labels[label]
    else
        return nothing
    end
end

function set_label!(v::LabeledVector{T, L}, index::Int, label::L) where {T, L}
    @assert !haskey(v.lables, label)
    v.labels[label] = index
    v.labels_reversed[index] = label
    nothing
end

function set_labels!(v::LabeledVector{T, L}, labels::AbstractVector{L}) where {T, L}
    for i in enumerate(labels)
        set_label(v, i...)
    end
    nothing
end