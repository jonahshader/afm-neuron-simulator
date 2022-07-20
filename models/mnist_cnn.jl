include("../afmneuronsim/afmneuronsim.jl")
using .AFMNeuronSim


const INIT_SCALE = 0.4
set_defaults!(a=0.01, bias=0.0002)

function make_2d_component(input_diameter, output_diameter)
    inputs = Vector{String}()
    outputs = Vector{String}()

    for i in 1:input_diameter
        for j in 1:input_diameter
            append!(inputs, "$j,$i")
        end
    end
    for i in 1:output_diameter
        for j in 1:output_diameter
            append!(outputs, "$j,$i")
        end
    end

    append!(inputs, "clock")

    Component(inputs, outputs)
end

function make_2d_component_1d_output(input_diameter, outputs)
    inputs = Vector{String}()

    for i in 1:input_diameter
        for j in 1:input_diameter
            append!(inputs, "$j,$i")
        end
    end

    append!(inputs, "clock")

    Component(inputs, outputs)
end

function init_component_weights!(comp::Component, in, nn, io, no)
    for input in input_labels(comp)
        for neuron in neuron_labels(comp)
            set_weight!(comp, input, neuron, randn() * in)
        end
    end

    for neuron1 in neuron_labels(comp)
        for neuron2 in neuron_labels(comp)
            set_weight!(comp, neuron1, neuron2, randn() * nn)
        end
    end

    for input in input_labels(comp)
        for output in output_labels(comp)
            set_weight!(comp, input, output, randn() * io)
        end
    end

    for neuron in neuron_labels(comp)
        for output in output_labels(comp)
            set_weight!(comp, neuron, output, randn() * no)
        end
    end
    nothing
end

function 


# function make_convolution(comp::Component, kernel::Component, input_layer_name, output_layer_name, kernel_name, radius, input_size::Tuple, output_features, neuron_count)
#     input_diameter = radius * 2 + 1

#     kernel = make_2d_component_1d_output(input_diameter, output_features)
#     add_neurons!(kernel, neuron_count)
#     init_component_weights!(kernel, INIT_SCALE, INIT_SCALE * .5, INIT_SCALE * .5, 1.0)
#     set_nonzeros_trainable!(kernel)

#     add_component!(comp, kernel, kernel_name)

# end

function convolve_kernel(comp::Component, kernel::Component, input_name, input_layer_sizes, kernel_input_size, input_features, output_features, kernel_name; stride=1, pad=false)
    if pad
        for i in 1:stride:input_size
            for j in 1:stride:input_size
                add_component!(comp, kernel, "$kernel_name,$j,$i")
            end
        end
    else
        for i in 1:stride:input_size-(kernel_input_size-1)
            for j in 1:stride:input_size-(kernel_input_size-1)

            end
        end
    end
end

# function connect_kernel_to_region(comp::Component, input_name, kernel_name)

function make_convolutional_layer(input_shape, kernel_size, )
    layer = make_2d_component(input_shape[1], input_shape)
end

function mnist_model()

end