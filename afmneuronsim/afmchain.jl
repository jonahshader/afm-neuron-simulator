function make_chain(layer_sizes::Vector{Int}; init_scale=1.0)
    
    base_comp = Component(layer_sizes[1], layer_sizes[end])
    layers = Vector{Component}()
    for i in 1:length(layer_sizes)-2
        layer = make_layer(layer_sizes[i], layer_sizes[i+1], init_scale=init_scale)
        add_component!(base_comp, layer, "layer_$i")
        push!(layers, layer)
    end

    # dense connection from last layer to output
    last_layer_name = "layer_$(length(layers))"
    scale = Float32(init_scale / sqrt(layer_sizes[1] + layer_sizes[2]))
    for layer_output in output_labels(layers[end])
        for base_output in output_labels(base_comp)
            set_weight!(base_comp, (last_layer_name, layer_output), base_output, randn() * scale)
        end
    end
    set_nonzeros_trainable!(base_comp)

    # connect input to first layer
    first_layer_name = "layer_1"
    for (a, b) in zip(input_labels(base_comp), input_labels(layers[1]))
        set_weight!(base_comp, a, (first_layer_name, b), :pass)
    end

    # connect layers
    for i in 1:length(layers)-1
        connect_layer(base_comp, layers[i], "layer_$i", layers[i+1], "layer_$(i+1)")
    end

    return base_comp
end

# layer is dense + neurons on end
function make_layer(input_size, output_size; init_scale=1.0)
    scale = Float32(init_scale / sqrt(input_size + output_size))
    layer = Component(input_size, output_size)
    add_neurons!(layer, output_size)

    for input in input_labels(layer)
        for neuron in neuron_labels(layer)
            set_weight!(layer, input, (neuron,), randn() * scale)
        end
    end
    set_nonzeros_trainable!(layer)

    for (neuron, output) in zip(neuron_labels(layer), output_labels(layer))
        set_weight!(layer, (neuron,), output, :pass)
    end

    return layer
end

function connect_layer(base::Component, first_layer::Component, first_layer_name::String, next_layer::Component, next_layer_name::String)
    for (a, b) in zip(output_labels(first_layer), input_labels(next_layer))
        set_weight!(base, (first_layer_name, a), (next_layer_name, b), :pass)
    end
end