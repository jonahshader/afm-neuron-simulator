include("../afmneuronsim/includes.jl")

function test_input_spike(picos=1000, num_neurons=20)
    comp = Component(1, 0)
    ts = (0.0, picos * PICO)
    add_neurons!(comp, num_neurons)
    set_weight!(comp, 1, (1,), 1.0)
    for i in 1:num_neurons-1
        set_weight!(comp, (i,), (i+1,), 1.0)
    end

    build_and_solve(comp, ts, input_to_spikes([1.0]), dense=false)
end