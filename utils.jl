# lists all ints in range 1:index_length, but replaces ints with string from str_to_int where possible
function indices_with_labels(index_length::Int, str_to_int::Dict{String, Int})::Vector{Union{String, Int}}
    output = Vector{Union{String, Int}}()
    reversed = Dict{Int, String}(value => key for (key, value) in str_to_int)
    for i in 1:index_length
        if haskey(reversed, i)
            push!(output, reversed[i])
        else
            push!(output, i)
        end
    end
    output
end