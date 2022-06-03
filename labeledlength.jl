mutable struct LabeledLength{L}
    length::Int
    labels::Dict{L, Int}
    labels_reversed::Dict{Int, L}
end

function raw(v::LabeledLength)
    v.length
end

function LabeledLength{L}(length::Int) where {L}
    LabeledLength{L}(length, Dict{L, Int}(), Dict{Int, L}())
end

function get_label(l::LabeledLength, index::Int)
    if haskey(l.labels_reversed, index)
        return l.labels_reversed[index]
    else
        return nothing
    end
end

function get_index(l::LabeledLength{L}, label::L) where {L}
    if haskey(l.labels, label)
        return l.labels[label]
    else
        return nothing
    end
end

function set_label!(l::LabeledLength{L}, index::Int, label::L) where {L}
    @assert !haskey(l.labels, label)
    l.labels[label] = index
    l.labels_reversed[index] = label
end

function set_labels!(l::LabeledLength{L}, labels::AbstractVector{L}) where {L}
    for i in enumerate(labels)
        set_label!(l, i...)
    end
end

function Base.length(l::LabeledLength)
    return l.length
end

