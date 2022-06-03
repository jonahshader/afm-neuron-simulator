include("afmneuron_rewritten.jl")
include("afmgraph.jl")
include("afmdiffeq.jl")

using DifferentialEquations
using Plots

full_adder = Component(["a", "b", "c-in"], ["sum", "c-out"])
xor = Component(2, 1)
add_component!(full_adder, xor, "xor1")
add_component!(full_adder, xor, "xor2")
and = Component(2, 1)
add_component!(full_adder, and, "and1")
add_component!(full_adder, and, "and2")
or = Component(2, 1)
add_component!(full_adder, or, "or1")

add_neurons!(full_adder, 5)

# set_weight!(full_adder, source=”A”, dest=(“xor1”, 1), weight=8.0)
# set_weight!(full_adder, source=”A”, dest=(“and2”, 2), weight=8.0)
# set_weight!(full_adder, source=”B”, dest=(“xor1”, 2), weight=8.0)
# set_weight!(full_adder, source=”B”, dest=(“and2”, 2), weight=8.0)
# set_weight!(full_adder, source=”C-in”, dest=(“xor2”, 2), weight=8.0)
# set_weight!(full_adder, source=”C-in”, dest=(“and1”,1), weight=8.0)
# set_weight!(full_adder, source=(“xor1”, 1), dest=(“xor2”,1), weight=8.0)
# set_weight!(full_adder, source=(“xor1”, 1), dest=(“and1”,2), weight=8.0)
# set_weight!(full_adder, source=(“xor2”, 1), dest=”Sum”, weight=8.0)
# set_weight!(full_adder, source=(“and1”, 1), dest=(“or1”, 1), weight=8.0)
# set_weight!(full_adder, source=(“and2”, 1), dest=(“or1”, 2), weight=8.0)
# set_weight!(full_adder, source=(“or1”, 1), dest=”C-out”, weight=8.0)

set_weight!(full_adder, "a", ("xor1", 1), 0.8)
set_weight!(full_adder, "a", ("and2", 2), 0.8)
set_weight!(full_adder, "b", ("xor1", 2), 0.8)
set_weight!(full_adder, "b", ("and2", 2), 0.8)
set_weight!(full_adder, "c-in", ("xor2", 2), 0.8)
set_weight!(full_adder, "c-in", ("and1", 1), 0.8)
set_weight!(full_adder, ("xor1", 1), ("xor2", 1), 0.8)
set_weight!(full_adder, ("xor1", 1), ("and1", 2), 0.8)
set_weight!(full_adder, ("xor2", 1), "sum", 0.8)
set_weight!(full_adder, ("and1", 1), ("or1", 1), 0.8)
set_weight!(full_adder, ("and2", 1), ("or1", 2), 0.8)
set_weight!(full_adder, ("or1", 1), "c-out", 0.8)

add_neurons!(and, 1)
println(and.weights)
set_weight!(and, 1, (1,), 0.4)
set_weight!(and, 2, (1,), 0.4)
set_weight!(and, (1,), 1, 0.8)

set_weight!(or, 1, 1, 1.0)
set_weight!(or, 2, 1, 1.0)

add_component!(xor, and, "and1")
add_component!(xor, or, "or1")
# add_neurons!()
set_weight!(xor, 1, ("and1", 1), 1.0)
set_weight!(xor, 2, ("and1", 2), 1.0)
set_weight!(xor, 1, ("or1", 1), 1.0)
set_weight!(xor, 2, ("or1", 2), 1.0)
set_weight!(xor, ("and1", 1), 1, -1.0)
set_weight!(xor, ("or1", 1), 1, 1.0)


nodes = make_nodes_from_component_tree(full_adder)
weights = make_weights_from_component_tree(full_adder, nodes)

substitute_internal_io!(weights, nodes)

mats = graph_to_labeled_matrix(weights, nodes)

# println(full_adder)

input_functions = Vector{Function}()
push!(input_functions, make_gaussian(1e13 * _beta, 0.5e-12, 0.5e-12))
push!(input_functions, make_gaussian(1e13 * _beta, 0.5e-12, 0.5e-12))
push!(input_functions, make_gaussian(1e13 * _beta, 0.5e-12, 0.5e-12))
# for i in 1:3
#     push!(input_functions, x->x)
# end

u0 = build_u0(full_adder)
p = build_p(full_adder, mats..., input_functions)

tspan = (0.0, 8e-12)
prob = ODEProblem(afm_diffeq!, u0, tspan, p)

@time sol = solve(prob)

plot(sol)