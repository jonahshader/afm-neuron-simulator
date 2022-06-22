include("../afmcomponent.jl")
include("../graph/afmgraph.jl")
include("../afmdiffeq.jl")
include("../afmtraining.jl")

using DifferentialEquations
using Plots

const input_size = 5
lin_noise_classifier = Component(input_size, 1)
add_neurons!(lin_noise_classifier, 10)

const init_noise_scale = 0.2

# set neuron to neuron weights, disallowing weights to self
for i in neuron_labels(lin_noise_classifier)
  for j in neuron_labels(lin_noise_classifier)
    if i != j
        # set_weight!(lin_noise_classifier, i, j, 0.0)
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
            for j in 1:input_size
                push!(linear_input, intercept + slope * ((j-1)-(input_size-1)/2.0))
            end
            push!(linear_output, 1.0)

            noise_input = Vector{Float64}()
            noise_output = Vector{Float64}()
            # generate noise sample
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

# custom_loss_fun = generate_loss_fun(10, 5)
# train!(parts, custom_loss_fun, 10, 20)