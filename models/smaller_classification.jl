include("../afmneuronsim/includes.jl")

using Images
using Flux: onehot

# set custom defaults
set_weight_scalar(1.0)
set_default_a(0.01)
set_default_bias(0.0002)

# each image becomes its own class. this will need to be modified when we have more than one image per class.
# this method loads the images and generates variations by adding gaussian noise to the images
function load_data(;variations_per_class=1, noise_scalar=0.1)
    path = "models/smaller_classification_dataset"
    image_names = readdir(path)

    xtrain = Vector{Vector{Float64}}()
    ytrain = Vector{Vector{Float64}}()
    viewable_images = Vector{Matrix{Gray{Float64}}}()
    for (i, name) in enumerate(image_names)
        image_path = "$path/$name"
        image = convert(Matrix{Gray{Float64}}, load(image_path))


        for _ in 1:variations_per_class
            # apply noise to the image
            modified_image = image .+ randn(size(image)...) .* noise_scalar
            push!(viewable_images, modified_image)
            push!(xtrain, convert(Vector{Float64}, reshape(modified_image, length(modified_image))))
            push!(xtrain[end], 1.0) # add clock
            push!(ytrain, convert(Vector{Float64}, onehot(i, 1:length(image_names))))
        end
    end

    xtrain, ytrain, viewable_images
end

function make_model()
    input_size = (7*7) + 1
    output_size = 4

    classifier = Component(input_size, output_size)
    add_neurons!(classifier, 50)
    init_component_weights!(classifier, 0.04, 0.0, 0.0, 1.0)

    # increase the size of the clock weights
    for neuron in neuron_labels(classifier)
        set_weight!(classifier, input_size, (neuron,), randn() * 0.4)
    end

    set_nonzeros_trainable!(classifier)

    classifier
end

function loss_fun_builder(xtrain, ytrain)
    function custom_loss_fun(parts)
        return loss_logitcrossentropy!(parts, xtrain, ytrain; peak_current=0.0001, spike_center=21e-13, spike_width=9e-13)
    end
    return custom_loss_fun
end

function run()
    model = make_model()
    parts = build_model_parts(model, (0.0, 20 * PICO), input_to_spikes(zeros(Float64, 7 * 7 + 1)))

    xtrain, ytrain, viewable_images_train = load_data(variations_per_class=100, noise_scalar=0.25)
    xtest, ytest, viewable_images_test = load_data(variations_per_class=100, noise_scalar=0.25)

    loss_fun = loss_fun_builder(xtrain, ytrain)
    loss_fun_test = loss_fun_builder(xtest, ytest)

    m, v, train_loss, test_loss = train!(parts, loss_fun, 20, 10000; validation_loss_fun=loss_fun_test)
    # parts, loss_fun, xtrain, ytrain, viewable_images, m, v
    return parts, loss_fun, loss_fun_test, xtrain, ytrain, xtest, ytest, viewable_images_train, viewable_images_test, m, v, train_loss, test_loss
end

# parts, loss_fun, xtrain, ytrain, viewable_images, m, v = run();
# parts, loss_fun, loss_fun_test, xtrain, ytrain, xtest, ytest, viewable_images_train, viewable_images_test, m, v, train_loss, test_loss = run();
# build_and_solve(parts.root, (0.0, 2e-11), input_to_spikes(xtrain[1]; peak_current=0.0001, spike_center=21e-13, spike_width=9e-13)) |> plot_output