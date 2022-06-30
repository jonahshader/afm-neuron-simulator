# include("afmcomponent.jl")
# include("afmdiffeq.jl")

mutable struct InstanceEval{T}
    params::T
    eval::Float64
end

function set_nonzeros_trainable!(comp::Component)
    set_raw!(weights_trainable_mask(comp), raw(weights(comp)) .!= 0)
end

function loss!(parts::AFMModelParts, input::Vector{Vector{T}}, target_output::Vector{Vector{T}}; args...) where {T<:AbstractFloat}
    total_loss = 0.0
    for io_pair in zip(input, target_output)
        total_loss += loss!(parts, io_pair[1], io_pair[2], args...)
    end
    total_loss / length(input)
end

function loss!(parts::AFMModelParts, single_input, single_target_output; peak_output=8e12)
    rebuild_model_parts!(parts, new_input_functions=input_to_spikes(single_input))
    solve_parts!(parts)
    sum(((output_max(parts) ./ peak_output) .- single_target_output) .^ 2) / length(single_target_output)
end

function train!(parts::AFMModelParts, loss_fun::Function, population_size::Int, iterations::Int, a=0.01, m=nothing, v=nothing)
    init_params, mask = parameter_mask_view(root(parts))
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
        push!(population, InstanceEval{typeof(zero)}(mutate!(deepcopy(init_params), mask, 0.01), 0.0))
    end

    for i in 1:iterations
        println("Iteration: ", i)
        # mutate
        for p in population
            mutate!(p.params, mask, 0.01)
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
        # center_params .-= gradient_approx # .* 100000
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