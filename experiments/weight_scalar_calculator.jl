include("../afmneuronsim/includes.jl")

set_weight_scalar(1.0)

function make_model()
    top = Component(1, 0)
    first_spike_distiller = Component(1, 1)
    add_neurons!(first_spike_distiller, 5)
    distill_neuron_weight = 0.7
    # set input weight
    set_weight!(first_spike_distiller, 1, (1,), distill_neuron_weight)
    for i in 1:4
        set_weight!(first_spike_distiller, (i,), (i + 1,), distill_neuron_weight)
    end
    # set output weight
    set_weight!(first_spike_distiller, (5,), 1, distill_neuron_weight)


    add_component!(top, first_spike_distiller, "first_spike_distiller")

    parallel_neurons = Component(1, 0)

    parallel_neuron_weights = Vector{Float64}()
    num_p_n = 10
    for i in 1:num_p_n
        push!(parallel_neuron_weights, (2.2 * i / num_p_n) ^ 2)
    end

    parallel_neuron_names = [string(x) for x in parallel_neuron_weights]

    add_neurons!(parallel_neurons, parallel_neuron_names) 
    # set different weights from input to each neuron
    for i in 1:num_p_n
        set_weight!(parallel_neurons, 1, (parallel_neuron_names[i],), parallel_neuron_weights[i])
    end

    add_component!(top, parallel_neurons, "parallel_neurons")

    # create top level weights
    set_weight!(top, 1, ("first_spike_distiller", 1), 1.0)
    set_weight!(top, ("first_spike_distiller", 1), ("parallel_neurons", 1), 1.0)
    return top
end

function run()
    top = make_model()
    input_funs = input_to_spikes([1.0])
    ts = (0.0, 9e-11)
    # parts = build_model_parts(top, ts, input_funs)
    # solve_parts!(parts)
    # build_and_solve(top, ts, input_funs) |> plot_Φ
    plot_Φ(build_and_solve(top, ts, input_funs), "[para")
end