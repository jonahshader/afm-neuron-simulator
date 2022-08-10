# include("afmcomponent.jl")
# include("afmdiffeq.jl")

using Flux.Losses: logitcrossentropy

export set_nonzeros_trainable!
export train!
export loss_mse!
export loss_logitcrossentropy!

mutable struct InstanceEval{T}
    params::T
    eval::Float64
end

function set_nonzeros_trainable!(comp::Component)
    set_raw!(weights_trainable_mask(comp), raw(weights(comp)) .!= 0)
end

function loss_mse!(parts::AFMModelParts, single_input, single_target_output; rebuild_model=true, peak_output=9e11, args...)
    if rebuild_model
        rebuild_model_parts!(parts, input_to_spikes(single_input), args...)
    else
        change_input_functions!(parts, input_to_spikes(single_input))
    end
    solve_parts!(parts, dense=false)
    sum(((output_max(parts) ./ peak_output) .- single_target_output) .^ 2) / length(single_target_output)
end

function loss_mse!(parts::AFMModelParts, input::Vector{Vector{T}}, target_output::Vector{Vector{T}}) where {T<:AbstractFloat}
    total_loss = 0.0
    full_rebuild = true
    for io_pair in zip(input, target_output)
        total_loss += loss_mse!(parts, io_pair[1], io_pair[2]; rebuild_model=full_rebuild)
        full_rebuild = false
    end
    total_loss / length(input)
end

function loss_logitcrossentropy!(parts::AFMModelParts, single_input, single_target_output, peak_output=9e11; rebuild_model=true)
    if rebuild_model
        rebuild_model_parts!(parts, new_input_functions=input_to_spikes(single_input))
    else
        change_input_functions!(parts, input_to_spikes(single_input))
    end
    solve_parts!(parts, dense=false)
    logitcrossentropy(output_max(parts) ./ peak_output, single_target_output)
end

function loss_logitcrossentropy!(parts::AFMModelParts, input::Vector{Vector{T}}, target_output::Vector{Vector{T}}) where {T<:AbstractFloat}
    total_loss = 0.0
    # TODO: parallelize
    full_rebuild = true
    for io_pair in zip(input, target_output)
        total_loss += loss_logitcrossentropy!(parts, io_pair[1], io_pair[2], rebuild_model=full_rebuild)
        full_rebuild = false 
    end
    total_loss / length(input)
end

# TODO: docs. this train!'s loss_fun takes in parts and returns loss
# below is a version that takes in a loss_fun_builder and a batch_fun
function train!(parts::AFMModelParts, loss_fun::Function, population_size::Int, iterations::Int; a=0.01, sd=0.01, m=nothing, v=nothing)
    init_params, mask = weight_mask_view(root(parts))
    zero = deepcopy(init_params)
    zero .-= zero

    population = Vector{InstanceEval{typeof(zero)}}()
    center_params = deepcopy(init_params)
    gradient_approx = deepcopy(zero)
    
    if isnothing(m)
        m = deepcopy(zero)
    end
    if isnothing(v)
        v = deepcopy(zero)
    end
    mh = deepcopy(m)
    vh = deepcopy(v)

    for i in 1:population_size
        push!(population, InstanceEval{typeof(zero)}(mutate!(deepcopy(init_params), mask, sd), 0.0))
    end

    for i in 1:iterations
        println("Iteration: ", i)
        # mutate
        for p in population
            mutate!(p.params, mask, sd)
        end

        # evaluate
        evaluate_all!(parts, init_params, population, loss_fun)

        # report average performance
        performance = sum(x -> x.eval, population) / length(population)
        println("iteration: ", i, " performance: ", performance)

        # apply rank transform
        sort!(population, by = p -> p.eval)
        for (i, p) in enumerate(population)
            p.eval = (((i-1) / (population_size-1)) * 2) - 1
        end

        # compute gradient approximation
        gradient_approx .= zero
        for p in population
            gradient_approx .+= p.params * p.eval
        end
        gradient_approx ./= population_size

        # update center_params with gradient approximation
        adam!(center_params, mask, gradient_approx, a, 0.9, 0.999, m, v, mh, vh, i)

        # copy center_params to population
        for p in population
            p.params .= center_params
        end
    end

    init_params .= center_params
end

# every iteration a new batch is generated as opposed to the above function which delegates data handling to the loss_fun
# this function decouples the batch generation from the loss_fun

# loss_fun_builder takes in a batch and returns a function that takes in parts and returns loss
# batch_generator takes in nothing and returns a tuple (xtrain, ytrain)
function train!(parts::AFMModelParts, loss_fun_builder::Function, batch_generator::Function, population_size::Int, iterations::Int; a=0.01, sd=0.01, m=nothing, v=nothing)
    init_params, mask = weight_mask_view(root(parts))
    zero = deepcopy(init_params)
    zero .-= zero

    population = Vector{InstanceEval{typeof(zero)}}()
    center_params = deepcopy(init_params)
    gradient_approx = deepcopy(zero)
    
    if isnothing(m)
        m = deepcopy(zero)
    end
    if isnothing(v)
        v = deepcopy(zero)
    end
    mh = deepcopy(m)
    vh = deepcopy(v)

    for i in 1:population_size
        push!(population, InstanceEval{typeof(zero)}(mutate!(deepcopy(init_params), mask, sd), 0.0))
    end

    for i in 1:iterations
        println("Iteration: ", i)
        # mutate
        for p in population
            mutate!(p.params, mask, sd)
        end

        # evaluate
        (xtrain, ytrain) = batch_generator()
        loss_fun = loss_fun_builder(xtrain, ytrain)
        evaluate_all!(parts, init_params, population, loss_fun)

        # report average performance
        performance = sum(x -> x.eval, population) / length(population)
        println("iteration: ", i, " performance: ", performance)

        # apply rank transform
        sort!(population, by = p -> p.eval)
        for (i, p) in enumerate(population)
            p.eval = (((i-1) / (population_size-1)) * 2) - 1
        end

        # compute gradient approximation
        gradient_approx .= zero
        for p in population
            gradient_approx .+= p.params * p.eval
        end
        gradient_approx ./= population_size

        # update center_params with gradient approximation
        adam!(center_params, mask, gradient_approx, a, 0.9, 0.999, m, v, mh, vh, i)

        # copy center_params to population
        for p in population
            p.params .= center_params
        end
    end

    init_params .= center_params
end

function evaluate!(parts::AFMModelParts, init_params, eval_instance, loss_fun)
    init_params .= eval_instance.params
    eval_instance.eval = loss_fun(parts)
end

function evaluate_all!(parts::AFMModelParts, init_params, population, loss_fun)
    for p in population
        evaluate!(parts, init_params, p, loss_fun)
        print(".")
    end
    nothing
end

function mutate!(params, mask, sd)
    params .+= mask .* randn.() .* sd
    params
end

function adam!(params, mask, gradients, a, beta1, beta2, m, v, mh, vh, t)
    m .= beta1 .* m + (1-beta1) .* gradients
    v .= beta2 .* v + (1-beta2) .* gradients .^ 2
    mh .= m ./ (1-beta1^t)
    vh .= v ./ (1-beta2^t)
    params .-= (a .* mh ./ (sqrt.(vh) .+ 1e-8)) .* mask # .* mask might not be necessary
end