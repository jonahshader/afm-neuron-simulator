# include("../afmneuronsim/afmneuronsim.jl")
# using .AFMNeuronSim
# include("../afmneuronsim/includes.jl")


set_defaults!(a=0.1, bias=0.000198)
k_zero = 0.011 / (_beta *_sigma)
# k_zero = 0.009 / (_beta *_sigma)

n = 10
chain = Component(0, 0)
add_neurons!(chain, 1, Φ_init=0.9)
add_neurons!(chain, n-1)
# add_neuron!(chain, "a", Φ_init=0.9)
# add_neurons!(chain, ["b", "c", "d", "e", "f", "g", "h", "i", "j"])

# set_weight!(chain, 1, (1,), 1.0) # connect input to first neuron
for i in 1:n-1
    set_weight!(chain, (i,), (i+1,), k_zero) # connect ith neuron to i+1th neuron
end
# set_weight!(chain, (n,), 1, 1.0) # connect last neuron to output

# set_weight!(chain, 1, ("a",), 1.0) # connect input to first neuron
# # manually set weights for the letters
# set_weight!(chain, ("a",), ("b",), k_zero)
# set_weight!(chain, ("b",), ("c",), k_zero)
# set_weight!(chain, ("c",), ("d",), k_zero)
# set_weight!(chain, ("d",), ("e",), k_zero)
# set_weight!(chain, ("e",), ("f",), k_zero)
# set_weight!(chain, ("f",), ("g",), k_zero)
# set_weight!(chain, ("g",), ("h",), k_zero)
# set_weight!(chain, ("h",), ("i",), k_zero)
# set_weight!(chain, ("i",), ("j",), k_zero)
# set_weight!(chain, ("j",), 1, 1.0) # connect last neuron to output

parts = build_model_parts(chain, (0.0, 9e-10), input_to_spikes([0.0]));

solve_parts!(parts, dense=false);
plot_Φ(parts)