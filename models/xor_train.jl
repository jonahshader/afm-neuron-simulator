include("../afmneuronsim/includes.jl")

using DifferentialEquations
using Plots

set_weight_scalar(1.0)

function make_model()
    # the +1 is for the clock input
    input_size = 2 + 1
    output_size = 1

    xor = Component(input_size, output_size)
    add_neurons!(xor, 15)
    init_component_weights!(xor, 0.4, 0.0, 0.0, 1.0)

    # # increase the size of the clock weights
    # for neuron in neuron_labels(xor)
    #     set_weight!(xor, input_size, (neuron,), randn() * 0.4)
    # end

    set_nonzeros_trainable!(xor)

    xor
end

function custom_loss!(parts::AFMModelParts)
    inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]] * 1.0
    outputs = [[0], [1], [1], [0]] * 1.0

    return loss_mse!(parts, inputs, outputs)
end

function run(model)
    parts = build_model_parts(model, (0.0, 3e-11), input_to_spikes([0.0, 0.0, 1.0]))

    train!(parts, custom_loss!, 30, 100)
    parts
end

# to use, run
# include("models/xor_train.jl")
# model = make_model();
# run(model);


# you can call model = make_model(); several times to get a promising untrained model
# you can test out the model by calling:
# plot_output(build_and_solve(model, (0.0, 3e-11), input_to_spikes([0.0, 0.0, 1.0])))
# or any other plotting functions, with any inputs