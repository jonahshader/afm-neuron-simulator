include("afmneuron_rewritten.jl")
include("afmgraph.jl")
full_adder = Component(["a", "b", "c-in"], ["sum", "c-out"])
xor = Component(2, 1)
add_component!(full_adder, xor, "xor1")
add_component!(full_adder, xor, "xor2")
and = Component(2, 1)
add_component!(full_adder, and, "and1")
add_component!(full_adder, and, "and2")
or = Component(2, 1)
add_component!(full_adder, or, "or1")
delay = Component(1, 1)
add_component!(or, delay, "delay1")

add_neurons!(delay, 3)

add_neurons!(full_adder, 5)
add_neurons!(and, 3)
add_neurons!(or, 1)
add_neurons!(xor, 2)

set_weight!(full_adder, "a", ("xor1", 1), 8.0)
# println(full_adder)

