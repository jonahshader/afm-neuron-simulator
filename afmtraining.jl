include("afmcomponent.jl")
include("afmdiffeq.jl")

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
    solve!(parts)
    sum(((output_max(parts) ./ peak_output) .- single_target_output) .^ 2) / length(single_target_output)
end

function train!(parts::AFMModelParts, loss_fun::Function, population_size::Int, iterations::Int)
    init_params, mask = parameter_mask_view(root(parts))
    zero = deepcopy(init_params)
    zero .-= zero

    population = Vector{InstanceEval{typeof(zero)}}()
    center_params = deepcopy(init_params)

    for i in 1:population_size
        push!(population, InstanceEval{typeof(zero)}(mutate!(deepcopy(init_params), mask, i / population_size), 0.0))
    end

    for i in 1:iterations
        # mutate
        for p in population
            mutate!(p.params, mask, 0.01)
        end

        # evaluate
        evaluate_all!(parts, init_params, population, loss_fun)

        # apply rank transform
        sort!(population, by = p -> p.eval)
        for (i, p) in population
            p.eval = (((i-1) / (population_size-1)) * 2) - 1
        end

        # compute weighted average
        center_params .= zero
        for p in population
            center_params .+= p.params * p.eval
        end
        center_params ./= population_size
        
    end
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