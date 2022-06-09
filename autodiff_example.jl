include("example.jl")

using DiffEqSensitivity
using Zygote
using ComponentArrays

function sum_of_solution(u0, p)
    _prob = remake(parts.ode_problem, u0=u0, p=p)
    sum(solve(_prob, sensealg=QuadratureAdjoint()))
end

mats = graph_to_labeled_matrix(parts.graph.weights, parts.graph.nodes)
p = build_p(full_adder, mats..., input_functions)
du01,dp1 = Zygote.gradient(sum_of_solution, parts.u0, ComponentArray(p))