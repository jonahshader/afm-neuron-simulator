include("afmneuron_rewritten.jl")

const Path = Vector{Union{String, Int}}
struct Node
    path::Path
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

function make_graph_from_component_tree(root::Component)
    root_inputs = Vector{Node}()
    root_outputs = Vector{Node}()

    for input in inputs(root)
        push!(root_inputs, Node(Path(), input, :root_input))
    end

    for output in outputs(root)
        push!(root_outputs, Node(Path(), output, :root_output))
    end

    vcat(root_inputs, root_outputs, make_qualified_nodes(root, Path(), root_inputs, root_outputs))
end

function make_qualified_nodes(root::Component, current_path::Path, input_nodes::Vector{Node}, output_nodes::Vector{Node})
    nodes = Vector{Node}()
    # add inputs, outputs, and neurons as nodes, with current_path
    # call this function for every component, and append that component's name to current_path

    # nodes = vcat(nodes, input_nodes, output_nodes) # this is in the wrong spot
    for neuron in neurons(root)
        push!(nodes, Node(copy(current_path), neuron, :neuron))
    end

    # my_input_nodes = Vector{Node}()
    # for input in inputs(root)
    #     new_input_node = Node(copy(current_path), input, :input)
    #     push!(my_input_nodes, new_input_node)
    #     # push!(nodes, new_input_node)
    # end

    # my_output_nodes = Vector{Node}()
    # for output in outputs(root)
    #     new_output_node = Node(copy(current_path), output, :output)
    #     push!(my_output_nodes, new_output_node)
    #     # push!(nodes, new_output_node)
    # end

    for clabel in components(root)
        c = root[clabel]
        comp_input_nodes = Vector{Node}()
        comp_output_nodes = Vector{Node}()
        for input in inputs(c)
            push!(comp_input_nodes, Node(vcat(copy(current_path), clabel), input, :input))
        end
        for output in outputs(c)
            push!(comp_output_nodes, Node(vcat(copy(current_path), clabel), output, :output))
        end

        nodes = vcat(nodes, comp_input_nodes, comp_output_nodes, make_qualified_nodes(c, vcat(current_path, clabel), comp_input_nodes, comp_output_nodes))
    end

    nodes
end