
mutable struct LabeledVector{T, L}
    vector::AbstractVector{T}
    labels::Dict{L, Int}
    labels_reversed::Dict{Int, L}
end

function raw(v::LabeledVector)
    v.vector
end

function set_raw!(l::LabeledVector, v)
    l.vector = v
end

LabeledVector{T, L}(vector::AbstractVector{T}) where {T, L} = LabeledVector{T, L}(vector, Dict{L, Int}(), Dict{Int, L}())
function LabeledVector{T, L}(vector::AbstractVector{T}, init_labels::AbstractVector{L}) where {T, L}
    v = LabeledVector(vector)
    for i in enumerate(init_labels)
        set_label!(v, i...)
    end
end

get_vector(v::LabeledVector) = v.vector
Base.length(lv::LabeledVector) = length(lv.vector)

Base.push!(lv::LabeledVector{T, L}, x::T) where {T, L} = Base.push!(lv.vector, x)
function push_and_label!(lv::LabeledVector{T, L}, x::T, label::L) where {T, L}
    push!(lv.vector, x)
    set_label!(lv, length(lv.vector), label)
end

function haslabel(lv::LabeledVector{T, L}, label::L) where {T, L}
    haskey(lv.labels, label)
end

function Base.setindex!(v::LabeledVector, x, i1::Int)
    v.vector[i1] = x
    nothing
end

function Base.setindex!(v::LabeledVector{T, L}, x, i1::L) where {T, L}
    v.vector[labels[i1]] = x
    nothing
end

function remove!(lv::LabeledVector{T, L}, i1::Int) where {T, L}
    # find the label to remove using the reverse lookup
    label = get_label(lv, i1)
    # remove the label from the labels dict
    delete!(lv.labels, label)
    # remove the label from the labels_reversed dict
    delete!(lv.labels_reversed, i1)
    # remove the element from the vector
    deleteat!(lv.vector, i1)
    # shift down all indices that are greater than i1 using list comp
    new_lables = Dict{L, Int}()
    new_reverse_lables = Dict{Int, L}()
    for (x, y) in lv.labels
        if y > i1
            new_lables[x] = y - 1
            new_reverse_lables[y - 1] = x
        else
            new_lables[x] = y
            new_reverse_lables[y] = x
        end
    end
    lv.labels = new_lables
    lv.labels_reversed = new_reverse_lables
end

function remove!(lv::LabeledVector{T, L}, i1::L) where {T, L}
    remove!(lv, labels[i1])
end

function get_label(v::LabeledVector, index::Int)
    if haskey(v.labels_reversed, index)
        return v.labels_reversed[index]
    else
        return nothing
    end
end

function Base.getindex(v::LabeledVector{T, L}, i1::Int) where {T, L}
    return v.vector[i1]
end

function Base.getindex(v::LabeledVector{T, L}, label::L) where {T, L}
    if haskey(v.labels, label)
        return v.vector[v.labels[label]]
    else
        return nothing
    end
end

function haslabel(v::LabeledVector{T, L}, label::L) where {T, L}
    return haskey(v.labels, label)
end

function set_label!(v::LabeledVector{T, L}, index::Int, label::L) where {T, L}
    @assert !haskey(v.labels, label)
    v.labels[label] = index
    v.labels_reversed[index] = label
    nothing
end

function set_labels!(v::LabeledVector{T, L}, labels::AbstractVector{L}) where {T, L}
    for i in enumerate(labels)
        set_label!(v, i...)
    end
    nothing
end

function indices_with_labels(v::LabeledVector{T, L}) where {T, L}
    output = Vector{Union{Int, L}}()
    for i in 1:length(v.vector)
        label = get_label(v, i)
        if isnothing(label)
            push!(output, i)
        else
            push!(output, label)
        end
    end
    output
end