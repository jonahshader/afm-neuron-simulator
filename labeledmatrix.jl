mutable struct LabeledMatrix{T, L}
    matrix::AbstractMatrix{T}
    row_labels::Dict{L, Int}
    row_labels_reversed::Dict{Int, L}
    col_labels::Dict{L, Int}
    col_labels_reversed::Dict{Int, L}
end

LabeledMatrix{T, L}(matrix::AbstractMatrix{T}) where {T, L} = LabeledMatrix{T, L}(matrix, Dict{L, Int}(), Dict{Int, L}(), Dict{L, Int}(), Dict{Int, L}())

get_matrix(m::LabeledMatrix) = m.matrix

Base.getindex(m::LabeledMatrix{T, L}, i1::Int, i2::Int) where {T, L} = m.matrix[i1, i2]
Base.getindex(m::LabeledMatrix{T, L}, i1::Int, i2::L) where {T, L} = m.matrix[i1, m.col_labels[i2]]
Base.getindex(m::LabeledMatrix{T, L}, i1::L, i2::Int) where {T, L} = m.matrix[m.row_labels[i1], i2]
Base.getindex(m::LabeledMatrix{T, L}, i1::L, i2::L) where {T, L} = m.matrix[m.row_labels[i1], m.col_labels[i2]]

function Base.setindex!(m::LabeledMatrix{T, L}, x, i1::Int, i2::Int) where {T, L}
    m.matrix[i1, i2] = x
end

function Base.setindex!(m::LabeledMatrix{T, L}, x, i1::Int, i2::L) where {T, L}
    m.matrix[i1, m.col_labels[i2]] = x
end

function Base.setindex!(m::LabeledMatrix{T, L}, x, i1::L, i2::Int) where {T, L}
    m.matrix[m.row_labels[i1], i2] = x
end

function Base.setindex!(m::LabeledMatrix{T, L}, x, i1::L, i2::L) where {T, L}
    m.matrix[m.row_labels[i1], m.col_labels[i2]] = x
end

function get_row_label(m::LabeledMatrix, row::Int)
    if haskey(m.row_labels_reversed, row)
        return m.row_labels_reversed[row]
    else
        return nothing
    end
end

function get_col_label(m::LabeledMatrix, col::Int)
    if haskey(m.col_labels_reversed, col)
        return m.col_labels_reversed[col]
    else
        return nothing
    end
end

function set_row_label!(m::LabeledMatrix{T, L}, row::Int, label::L) where {T, L}
    @assert !haskey(m.row_labels, label)
    m.row_labels[label] = row
    m.row_labels_reversed[row] = label
    nothing
end

function set_col_label!(m::LabeledMatrix{T, L}, col::Int, label::L) where {T, L}
    @assert !haskey(m.col_labels, label)
    m.col_labels[label] = col
    m.col_labels_reversed[col] = label
    nothing
end

function set_labels!(m::LabeledMatrix{T, L}, row_labels::AbstractVector{L}, col_labels::AbstractVector{L}) where {T, L}
    for i in enumerate(row_labels)
        set_row_label!(m, i...)
    end
    for i in enumerate(col_labels)
        set_col_label!(m, i...)
    end
    nothing
end