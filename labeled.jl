mutable struct LabeledMatrix{T}
    matrix::AbstractMatrix{T}
    row_labels::Dict{String, Int}
    col_labels::Dict{String, Int}
end

LabeledMatrix(matrix::T) where {T} = LabeledMatrix{T}(matrix, Dict{String, Int}(), Dict{String, Int}())

Base.getindex(m::LabeledMatrix, i1::Int64, i2::Int64) = m.matrix[i1, i2]
Base.getindex(m::LabeledMatrix, i1::Int64, i2::String) = m.matrix[i1, m.col_labels[i2]]
Base.getindex(m::LabeledMatrix, i1::String, i2::Int) = m.matrix[m.row_labels[i1], i2]
Base.getindex(m::LabeledMatrix, i1::String, i2::String) = m.matrix[m.row_labels[i1], m.col_labels[i2]]

set_row_label!(m::LabeledMatrix, )