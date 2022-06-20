include("afmcomponent.jl")
include("afmdiffeq.jl")

function set_nonzeros_trainable!(comp::Component)

    comp.weights_trainable_mask.matrix = comp.weights.matrix .!= 0
end

function loss!(parts::AFMModelParts, input::Vector{Vector{T}}, target_output::Vector{Vector{T}}) where {T<:AbstractFloat}
    total_loss = 0.0
    for io_pair in zip(input, target_output)
        total_loss += loss!(parts, io_pair[1], io_pair[2])
    end
    total_loss / length(input)
end

function loss!(parts::AFMModelParts, single_input, single_target_output; peak_output=8e12)
    rebuild_model_parts!(parts, new_input_functions=input_to_spikes(single_input))
    solve!(parts)
    sum(((output_max(parts) ./ peak_output) .- single_target_output) .^ 2) / length(single_target_output)
end