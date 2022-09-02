include("../afmneuronsim/includes.jl")

using Images

using Flux: onehot

set_weight_scalar(1.0)
set_default_a(0.01)
set_default_bias(0.0002)

# each image becomes its own class. this will need to be modified when we have more than one image per class.
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
        return loss_logitcrossentropy!(parts, xtrain, ytrain)
    end
    return custom_loss_fun
end

function run()
    model = make_model()
    parts = build_model_parts(model, (0.0, 20 * PICO), input_to_spikes(zeros(Float64, 7 * 7 + 1)))

    xtrain, ytrain, viewable_images = load_data(variations_per_class=100, noise_scalar=0.25)

    loss_fun = loss_fun_builder(xtrain, ytrain)

    m, v = train!(parts, loss_fun, 20, 30)
    parts, loss_fun, xtrain, ytrain, viewable_images, m, v
end

# parts, loss_fun, xtrain, ytrain, viewable_images, m, v = run();
# build_and_solve(parts.root, (0.0, 2e-11), input_to_spikes(xtrain[1])) |> plot_output