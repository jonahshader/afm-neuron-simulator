include("afmcomponent.jl")
include("afmdiffeq.jl")

function set_nonzeros_trainable!(comp::Component)
    comp.weights_trainable_mask.matrix = comp.weights.matrix .!= 0
end

function loss(input, target_output)

end

function loss!(root::Component, parts::AFMModelParts, single_input, single_target_output)
    rebuild_model_parts!(root, parts; input_functions=input_to_spikes(single_input))
    solve!(parts)
    sum((output_max(parts) .- single_target_output .* 3e12) .^ 2) / length(single_target_output)
end