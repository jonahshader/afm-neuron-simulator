# include("../afmneuronsim/afmneuronsim.jl")
# using .AFMNeuronSim
# include("../afmneuronsim/includes.jl")


set_defaults!(a=0.1, bias=0.000198)
k_zero = 0.011 / (_beta *_sigma)
# k_zero = 0.009 / (_beta *_sigma)
set_weight_scalar(1.0)

n = 10
chain = Component(0, 0)
# first neuron will spike because Φ_init > threshold
add_neurons!(chain, 1, Φ_init=0.9)
add_neurons!(chain, n-1)
# connect neurons in series
for i in 1:n-1
    set_weight!(chain, (i,), (i+1,), k_zero) # connect ith neuron to i+1th neuron
end

parts = build_and_solve(chain, (0.0, 9e-10), input_to_spikes([0.0]), dense=false);
plot_Φ(parts)