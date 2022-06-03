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

# applies f to c tree in depth first order, returns array of results from applying f to every c
function map_component_depth_first(f, c)
    return vcat([f(c)], map(x -> map_component_depth_first(f, x), c.components)...)
end

# same thing as map_component_depth_first, but does not put each result into an element in an array
# instead every result is concatinated into one array
function map_component_array_depth_first(f, c)
    if isempty(c.components)
        return f(c)
    else
        return vcat(f(c), map(x -> map_component_array_depth_first(f, x), c.components)...)
    end
    
end