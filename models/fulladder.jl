# include("../afmneuronsim/afmneuronsim.jl")
# using .AFMNeuronSim
include("../afmneuronsim/includes.jl")

# using DifferentialEquations
# using Plots

full_adder = Component(["a", "b", "c-in"], ["sum", "c-out"])
xor = Component(2, 1)
add_component!(full_adder, xor, "xor1")
add_component!(full_adder, xor, "xor2")
and = Component(2, 1)
add_component!(full_adder, and, "and1")
add_component!(full_adder, and, "and2")
or = Component(2, 1)
add_component!(full_adder, or, "or1")

scale = 0.7
set_weight!(full_adder, "a", ("xor1", 1), scale)
set_weight!(full_adder, "a", ("and2", 1), scale)
set_weight!(full_adder, "b", ("xor1", 2), scale)
set_weight!(full_adder, "b", ("and2", 2), scale)
set_weight!(full_adder, "c-in", ("xor2", 2), scale)
set_weight!(full_adder, "c-in", ("and1", 1), scale)
set_weight!(full_adder, ("xor1", 1), ("xor2", 1), scale)
set_weight!(full_adder, ("xor1", 1), ("and1", 2), scale)
set_weight!(full_adder, ("xor2", 1), "sum", scale)
set_weight!(full_adder, ("and1", 1), ("or1", 1), scale)
set_weight!(full_adder, ("and2", 1), ("or1", 2), scale)
set_weight!(full_adder, ("or1", 1), "c-out", scale)

add_neurons!(and, 1)
set_weight!(and, 1, (1,), scale * .5)
set_weight!(and, 2, (1,), scale * .5)
set_weight!(and, (1,), 1, scale)

add_neurons!(or, 1)
set_weight!(or, 1, (1,), scale)
set_weight!(or, 2, (1,), scale)
set_weight!(or, (1,), 1, scale)

add_component!(xor, and, "and1")
add_component!(xor, or, "or1")
set_weight!(xor, 1, ("and1", 1), 1.0)
set_weight!(xor, 2, ("and1", 2), 1.0)
set_weight!(xor, 1, ("or1", 1), 1.0)
set_weight!(xor, 2, ("or1", 2), 1.0)
set_weight!(xor, ("and1", 1), 1, -1.0)
set_weight!(xor, ("or1", 1), 1, 1.0)

# adding a neuron here should no longer reset the weights
add_neurons!(full_adder, 1)

input_funs = input_to_spikes([1.0, 1.0, 1.0])
ts = (0.0, 9e-11)
parts = build_model_parts(full_adder, ts, input_funs)

solve_parts!(parts)

plot_dΦ(parts)
plot_Φ(parts)

