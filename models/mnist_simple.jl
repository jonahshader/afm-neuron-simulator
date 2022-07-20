include("../afmneuronsim/afmneuronsim.jl")
using .AFMNeuronSim

using DifferentialEquations
using Plots
using MLDatasets
using Flux.Data: DataLoader
using Flux: onehot

# load MNIST images and return loader
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28^2, :)
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

function init_component_weights!(comp::Component, in, nn, io, no)
    for input in input_labels(comp)
        for neuron in neuron_labels(comp)
            set_weight!(comp, input, (neuron,), randn() * in)
        end
    end

    for neuron1 in neuron_labels(comp)
        for neuron2 in neuron_labels(comp)
            set_weight!(comp, (neuron1,), (neuron2,), randn() * nn)
        end
    end

    for input in input_labels(comp)
        for output in output_labels(comp)
            set_weight!(comp, input, output, randn() * io)
        end
    end

    for neuron in neuron_labels(comp)
        for output in output_labels(comp)
            set_weight!(comp, (neuron,), output, randn() * no)
        end
    end
    nothing
end

function make_model()
    input_size = (28*28) + 1
    output_size = 10

    mnist_classifier = Component(input_size, output_size)
    add_neurons!(mnist_classifier, 100)
    init_component_weights!(mnist_classifier, 0.04, 0.02, 0.0, 1.0)
    set_nonzeros_trainable!(mnist_classifier)

    mnist_classifier
end

function generate_loss_fun(batch_size, xtrain, ytrain)
    function custom_loss_fun!(parts::AFMModelParts)
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

        return loss!(parts, inputs, outputs)
    end

    return custom_loss_fun!
end

function run()
    model = make_model()
    parts = build_model_parts(model, (0.0, 6e-11), input_to_spikes(randn((28^2) + 1)))

    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28^2, :)
    l_fun = generate_loss_fun(30, xtrain, ytrain)

    train!(parts, l_fun, 30, 10)
    (parts, l_fun)
end

# using Revise; include("models/mnist_simple.jl"); (parts, l_fun) = run()