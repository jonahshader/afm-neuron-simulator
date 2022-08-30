# include("../afmneuronsim/afmneuronsim.jl")
# using .AFMNeuronSim
include("../afmneuronsim/includes.jl")

using DifferentialEquations
using Plots
using MLDatasets

using Flux.Data: DataLoader
using Flux: onehot

set_weight_scalar(1.0)

# load MNIST images and return loader
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28^2, :)
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end



function make_model()
    input_size = (28*28) + 1
    output_size = 10

    mnist_classifier = Component(input_size, output_size)
    add_neurons!(mnist_classifier, 50)
    init_component_weights!(mnist_classifier, 0.04, 0.0, 0.0, 1.0)

    # increase the size of the clock weights
    for neuron in neuron_labels(mnist_classifier)
        set_weight!(mnist_classifier, input_size, (neuron,), randn() * 0.4)
    end

    set_nonzeros_trainable!(mnist_classifier)

    mnist_classifier
end

# function generate_loss_fun(batch_size, xtrain, ytrain)
#     function custom_loss_fun!(parts::AFMModelParts)
#         inputs = Vector{Vector{Float64}}()
#         outputs = Vector{Vector{Float64}}()
#         for _ in 1:batch_size
#             selection = rand(1:size(ytrain)[1])
#             x = xtrain[:, selection]
#             y = ytrain[selection]
#             y = onehot(y, 0:9) * 1f0
    
#             push!(inputs, x * 1.0)
#             push!(inputs[end], 1.0) # add clock
#             push!(outputs, y * 1.0)
#         end

#         return loss_logitcrossentropy!(parts, inputs, outputs)
#     end

#     return custom_loss_fun!
# end


# function generate_better_loss_fun(batch_size, xtrain, ytrain)
#     inputs = Vector{Vector{Float64}}()
#     outputs = Vector{Vector{Float64}}()
#     for _ in 1:batch_size
#         selection = rand(1:size(ytrain)[1])
#         x = xtrain[:, selection]
#         y = ytrain[selection]
#         y = onehot(y, 0:9) * 1f0

#         push!(inputs, x * 1.0)
#         push!(inputs[end], 1.0) # add clock
#         push!(outputs, y * 1.0)
#     end

#     function custom_loss_fun!(parts::AFMModelParts)
#         return loss_logitcrossentropy!(parts, inputs, outputs)
#     end

#     return custom_loss_fun!
# end

function loss_fun_builder(xtrain, ytrain)
    function custom_loss_fun(parts)
        return loss_logitcrossentropy!(parts, xtrain, ytrain)
    end
    return custom_loss_fun
end

function make_batch_gen(batch_size, xtrain, ytrain)
    function batch_gen()
        inputs = Vector{Vector{Float64}}()
        outputs = Vector{Vector{Float64}}()
        for _ in 1:batch_size
            selection = rand(1:size(ytrain)[1])
            x = xtrain[:, selection]
            y = ytrain[selection]
            y = onehot(y, 0:9) * 1f0
    
            push!(inputs, x * 1.0)
            push!(inputs[end], 1.0) # add clock
            push!(outputs, y * 1.0)
        end
    
        return (inputs, outputs)
    end
    return batch_gen
end


function run()
    model = make_model()
    parts = build_model_parts(model, (0.0, 2e-11), input_to_spikes(randn((28^2) + 1)))


    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28^2, :)
    
    batch_gen_fun = make_batch_gen(60, xtrain, ytrain)


    train!(parts, loss_fun_builder, batch_gen_fun, 40, 50, a=0.001, sd=0.005)
    (parts, batch_gen_fun)
end

# using Revise; include("models/mnist_simple.jl"); (parts, l_fun) = run();
# using Revise; include("models/mnist_simple.jl"); (parts, batch_gen_fun) = run();