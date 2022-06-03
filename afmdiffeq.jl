include("afmgraph.jl")
include("afmneuron_rewritten.jl")
include("utils.jl")

using LinearAlgebra
using SparseArrays

function build_u0(root::Component)
    θ_init = map_component_array_depth_first(x->x.neurons.θ_init, root)
    dθ_init = map_component_array_depth_first(x->x.neurons.dθ_init, root)
    hcat(θ_init, dθ_init)
end

function build_neuron_params(root::Component)
    sigma = map_component_array_depth_first(x->x.neurons.sigma, root)
    a = map_component_array_depth_first(x->x.neurons.a, root)
    we = map_component_array_depth_first(x->x.neurons.we, root)
    wex = map_component_array_depth_first(x->x.neurons.wex, root)
    beta = map_component_array_depth_first(x->x.neurons.beta, root)
    bias = map_component_array_depth_first(x->x.neurons.bias, root)
    (sigma, a, we, wex, beta, bias)
end

function build_p(root::Component, nnm, inm, nom, iom, input_functions::Vector{Function})
    neuron_p = build_neuron_params(root)
    model_input_temp = similar(neuron_p[1], length(root.input))
    n_voltage_temp = similar(neuron_p[1])
    n_arr_temp2 = similar(neuron_p[1])
    n_arr_accum = similar(neuron_p[1])
    model_output_temp = similar(neuron_p[1], length(root.output))
    model_arr_accum = similar(neuron_p[1], length(root.output))

    (neuron_p..., raw(nnm), raw(inm), raw(nom), raw(iom), model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, model_output_temp, model_arr_accum, input_functions)
end

function make_gaussian(a, b, c)
    function gaussian(t)
        a * ℯ^(-((t - b)^2)/c)
    end
end

function afm_diffeq!(du, u, p, t)
    sigma, a, we, wex, beta, bias, nnm, inm, nom, iom, model_input_temp, n_voltage_temp, n_arr_temp2, n_arr_accum, model_output_temp, model_arr_accum, input_functions = p
    Φ = view(u, :, 1)
    dΦ = view(u, :, 2)
    duΦ = view(du, :, 1)
    dudΦ = view(du, :, 2)

    duΦ .= dΦ

    # populate model_input_temp from list of functions of time
    for i in eachindex(input_functions)
        model_input_temp[i] = input_functions[i](t)
    end

    # model_input to neuron_input
    n_voltage_temp .= dΦ .* beta # n_voltage_temp represents the voltage generated from the neurons
    mul!(n_arr_accum, inm, model_input_temp)
    mul!(n_arr_temp2, nnm, n_voltage_temp)
    n_arr_accum .+= n_arr_temp2 # accumulate current

    mul!(model_arr_accum, iom, model_input_temp)
    mul!(model_output_temp, nom, n_voltage_temp)
    model_arr_accum .+= model_output_temp # accumulate current

    n_arr_accum .+= bias # add bias current

    @. dudΦ = (sigma * n_arr_accum - a*dΦ - (we/2) * sin(2*Φ)) * wex
end