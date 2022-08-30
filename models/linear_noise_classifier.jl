include("../afmneuronsim/afmneuronsim.jl")
using .AFMNeuronSim


set_weight_scalar(1.0)

using DifferentialEquations
using Plots

const input_size = 5
lin_noise_classifier = Component(input_size + 1, 1)

# clock and three inputs
kernel = Component(4, 1)
add_neurons!(kernel, 10)

fully_connected = Component(4, 1)
add_neurons!(fully_connected, 15)

add_component!(lin_noise_classifier, kernel)
add_component!(lin_noise_classifier, kernel)
add_component!(lin_noise_classifier, kernel)

add_component!(lin_noise_classifier, fully_connected)

const init_noise_scale = 0.2

# connect kernels to clock and 3 inputs
for i in 1:3
  # connect clock to kernel
  set_weight!(lin_noise_classifier, 1, (i, 1), 1.0)
  # connect kernel output to fully_connected
  set_weight!(lin_noise_classifier, (i, 1), (4, i+1), 1.0)
  for j in 1:3
    # connect window to kernels
    set_weight!(lin_noise_classifier, i+j, (i, j), 1.0)
  end
end

# connect clock to fully_connected
set_weight!(lin_noise_classifier, 1, (4, 1), 1.0)

# TODO: connect fully_connected to output
# TODO: populate connections within fully_connected to be ... fully_connected lol


# set neuron to neuron weights, disallowing weights to self
for i in neuron_labels(lin_noise_classifier)
  for j in neuron_labels(lin_noise_classifier)
    if i != j
        set_weight!(lin_noise_classifier, i, j, randn() * init_noise_scale * 0.25)
        set_weight_trainable!(lin_noise_classifier, i, j, true)
    end
  end
end

# set input to neuron weights
for i in input_labels(lin_noise_classifier)
  for j in neuron_labels(lin_noise_classifier)
    set_weight!(lin_noise_classifier, i, j, randn() * init_noise_scale)
    set_weight_trainable!(lin_noise_classifier, i, j, true)
  end
end

# set neuron to output weights
for i in neuron_labels(lin_noise_classifier)
  set_weight!(lin_noise_classifier, i, 1, randn() * init_noise_scale)
  set_weight_trainable!(lin_noise_classifier, i, 1, true)
end


# set input to neuron weights
for i in input_labels(kernel)
  for j in neuron_labels(kernel)
    set_weight!(kernel, i, j, randn() * init_noise_scale)
    set_weight_trainable!(kernel, i, j, true)
  end
end

# set neuron to neuron weights
for i in neuron_labels(kernel)
  for j in neuron_labels(kernel)
    set_weight!(kernel, i, j, randn() * init_noise_scale * 0.25)
    set_weight_trainable!(kernel, i, j, true)
  end
end

# set neuron to output weights
for i in neuron_labels(kernel)
  for j in output_labels(kernel)
    set_weight!(kernel, i, j, randn() * init_noise_scale)
    set_weight_trainable!(kernel, i, j, true)
  end
end



function generate_loss_fun(samples, input_size)
    function custom_loss_fun!(parts::AFMModelParts)
        inputs = Vector{Vector{Float64}}()
        outputs = Vector{Vector{Float64}}()
        for i in 1:samples
            linear_input = Vector{Float64}()
            linear_output = Vector{Float64}()

            # generate linear sample
            slope = randn()
            intercept = randn()
            push!(linear_input, 1.0) # clock input (like bias neuron i guess)
            for j in 1:input_size
                push!(linear_input, intercept + slope * ((j-1)-(input_size-1)/2.0))
            end
            push!(linear_output, 1.0)

            noise_input = Vector{Float64}()
            noise_output = Vector{Float64}()
            # generate noise sample
            push!(noise_input, 1.0) # clock input (like bias neuron i guess)
            for j in 1:input_size
                push!(noise_input, randn())
            end
            push!(noise_output, 0.0)

            push!(inputs, linear_input)
            push!(outputs, linear_output)
            push!(inputs, noise_input)
            push!(outputs, noise_output)
        end

        return loss!(parts, inputs, outputs)
    end

    return custom_loss_fun!
end

function simple_loss_fun(pts::AFMModelParts)
    return loss!(pts, [1.0, 0.5, 0.5, 0.25, 0.0], [1.0])
end

parts = build_model_parts(lin_noise_classifier, (0.0, 6e-12), input_to_spikes([1.0, 1.0, 1.0, 1.0, 1.0]))

custom_loss_fun = generate_loss_fun(10, 5)
# train!(parts, custom_loss_fun, 50, 10)