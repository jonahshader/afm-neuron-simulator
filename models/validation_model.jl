# include("../afmneuronsim/afmneuronsim.jl")
# using .AFMNeuronSim
include("../afmneuronsim/includes.jl")


set_defaults!(a=0.1, bias=0.000198)
k_zero = 0.011 / (_beta *_sigma)
# k_zero = 0.009 / (_beta *_sigma)

n = 100
chain = Component(1, 1)
add_neurons!(chain, 1, Φ_init=0.9)
add_neurons!(chain, n-1)

set_weight!(chain, 1, (1,), 1.0) # connect input to first neuron
for i in 1:n-1
    set_weight!(chain, (i,), (i+1,), k_zero) # connect ith neuron to i+1th neuron
end
set_weight!(chain, (n,), 1, 1.0) # connect last neuron to output

parts = build_model_parts(chain, (0.0, 9e-10), input_to_spikes([0.0]));

solve_parts!(parts);
plot_Φ(parts)