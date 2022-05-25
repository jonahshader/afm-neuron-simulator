include("kotlinport.jl")
include("utils.jl")


build_theta(c::Component) = map_component_array_depth_first(x->x.theta, c)
build_d_theta(c::Component) = map_component_array_depth_first(x->x.d_theta, c)

build_u0(c::Component) = hcat(build_theta(c), build_d_theta(c))

# function map_component_array_depth_first(f, c)
#     return vcat(f(c), map(x -> map_component_depth_first(f, x), c.components)...)
# end

function total_neurons(c::Component)::Int
    c.neurons + sum(x->total_neurons(x), c.components, init=0) 
end

function sub_neurons(c::Component)::Int
    sum(x->x.neurons, c.components, init=0)
end

function compute_outputs!(c::Component, d_theta::AbstractVector{Float64})
    my_d_theta = view(d_theta, 1:c.neurons)
    next_d_theta = view(d_theta, c.neurons+1:total_neurons(c))
    

    for comp in c.components
        comp_d_theta = view(next_d_theta, 1:total_neurons(comp))
        compute_outputs!(c, comp_d_theta)
        next_d_theta = view(next_d_theta, total_neurons(comp)+1:total_neurons(c))
    end

    component_outputs = vcat(map(x -> x.output, c.components)...)
    mul!(c.output, c.output_weights, vcat(component_outputs, my_d_theta))
    nothing
end

function compute_dd_theta!(c::Component, input, theta::AbstractVector{Float64}, d_theta::AbstractVector{Float64})
    my_theta = view(theta, 1:c.neurons)
    my_d_theta = view(d_theta, 1:c.neurons)
    next_theta = view(theta, c.neurons+1:total_neurons(c))
    next_d_theta = view(d_theta, c.neurons+1:total_neurons(c))
    # my_dd_theta = (_sigma * neuron_)

    # TODO: this does not cover every case. come up with a better solution
    if isempty(c.components)
        if c.neurons > 0
            non_output_weight_input = vcat(input, my_d_theta)
        else
            non_output_weight_input = vcat(input)
        end
    else
        non_output_weight_input = vcat(input, map(x->x.output, c.components)..., my_d_theta)
    end
    
    mm = c.non_output_weights * non_output_weight_input
    component_inputs = view(mm, 1:c_inputs(c))
    neuron_inputs = view(mm, c_inputs(c)+1:length(mm))
    c.dd_theta = (_sigma * neuron_inputs - _a*my_d_theta-(_we/2.0)*sin.(my_theta * 2.0)) * _wex
    
    c_index = 0
    for comp in c.components
        c_input = view(component_inputs, c_index+1:(c_index+comp.inputs))
        compute_dd_theta!(comp, c_input, next_theta, next_d_theta)
        next_theta = view(next_theta, total_neurons(comp)+1:total_neurons(c))
        next_d_theta = view(next_d_theta, total_neurons(comp)+1:total_neurons(c))
        c_index += comp.inputs
    end
    nothing
end

root = Component(5, 10, 3)
sub = Component(2, 11, 4)
sub2 = Component(10, 25, 30)

add_component!(sub, sub2)
add_component!(root, sub)

input = rand(Float64, 5)

test_theta = build_theta(root)::Vector{Float64}
test_d_theta = build_d_theta(root)::Vector{Float64}

output = compute_outputs!(root, test_d_theta)

compute_dd_theta!(root, input, test_theta, test_d_theta)
print(root.dd_theta)
