include("../afmneuronsim/includes.jl")

set_weight_scalar(1.0)

function make_model()
    top = Component(1, 0)
    first_spike_distiller = Component(1, 1)
    add_neurons!(first_spike_distiller, 5)
    distill_neuron_weight = 0.7 * 10
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
    input_funs = input_to_spikes([1.0], spike_width=1.2e-12)
    ts = (0.0, 600 * PICO)
    # parts = build_model_parts(top, ts, input_funs)
    # solve_parts!(parts)
    # build_and_solve(top, ts, input_funs) |> plot_Φ
    plot_Φ(build_and_solve(top, ts, input_funs, dense=false), "[parallel_neurons]")
end

function test_chain(w, picos=2000, num_neurons=20)
    comp = Component(0, 0)
    ts = (0.0, picos * PICO)
    add_neuron!(comp, Φ_init=0.95)
    add_neurons!(comp, num_neurons-1)
    for i in 1:num_neurons-1
        set_weight!(comp, (i,), (i+1,), w)
    end

    build_and_solve(comp, ts, input_to_spikes([0.0]), dense=false)
end

