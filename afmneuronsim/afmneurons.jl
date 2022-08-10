using CUDA
using RecursiveArrayTools

export set_defaults!

const TERA = 1e12 # 10e12
const GIGA = 1e9 # 10e9
const FEMTO = 1e-15 # 10e-15

_fex = 27.5 * TERA

_fe = 1.75 * GIGA


_Φ_init = nothing
_dΦ_init = 0.0
_sigma = 27.1e12
_a = 0.01
_we = _fe * 2pi
_wex = _fex * 2pi
_beta = 0.11e-15
_bias = 0.0002 # 0.000202

default_neuron_params = [_Φ_init, _dΦ_init, _sigma, _a, _we, _wex, _beta, _bias]

function set_default_Φ_init!(Φ_init)
    default_neuron_params[1] = Φ_init
end
function set_default_dΦ_init!(dΦ_init)
    default_neuron_params[2] = dΦ_init
end
function set_default_sigma!(sigma)
    default_neuron_params[3] = sigma
end
function set_default_a!(a)
    default_neuron_params[4] = a
end
function set_default_we!(we)
    default_neuron_params[5] = we
end
function set_default_wex!(wex)
    default_neuron_params[6] = wex
end
function set_default_beta!(beta)
    default_neuron_params[7] = beta
end
function set_default_bias!(bias)
    default_neuron_params[8] = bias
end

get_default_Φ_init() = default_neuron_params[1]
get_default_dΦ_init() = default_neuron_params[2]
get_default_sigma() = default_neuron_params[3]
get_default_a() = default_neuron_params[4]
get_default_we() = default_neuron_params[5]
get_default_wex() = default_neuron_params[6]
get_default_beta() = default_neuron_params[7]
get_default_bias() = default_neuron_params[8]


const ϵ = 1e-12

mutable struct Neurons
    Φ_init::Vector{Float64}
    dΦ_init::Vector{Float64} 
    sigma::Vector{Float64}
    a::Vector{Float64}
    we::Vector{Float64}
    wex::Vector{Float64}
    beta::Vector{Float64}
    bias::Vector{Float64}
end

function Neurons()::Neurons
    Neurons(Vector{Float64}(), Vector{Float64}(), 
    Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), 
    Vector{Float64}(), Vector{Float64}(), Vector{Float64}())
end

function Base.length(n::Neurons)::Int
    # just pick one array, they should all be the same length
    Base.length(n.Φ_init)
end

function set_defaults!(;Φ_init=nothing, dΦ_init=nothing, sigma=nothing, a=nothing, we=nothing, wex=nothing, beta=nothing, bias=nothing)
    if !isnothing(Φ_init)
        set_default_Φ_init!(Φ_init)
    end
    if !isnothing(dΦ_init)
        set_default_dΦ_init!(dΦ_init)
    end
    if !isnothing(sigma)
        set_default_sigma!(sigma)
    end
    if !isnothing(a)
        set_default_a!(a)
    end
    if !isnothing(we)
        set_default_we!(we)
    end
    if !isnothing(wex)
        set_default_wex!(wex)
    end
    if !isnothing(beta)
        set_default_beta!(beta)
    end
    if !isnothing(bias)
        set_default_bias!(bias)
    end
end

# neurons: the neurons struct that will be added to
# n: number of neurons to add with the specified values
# function add_neurons!(neurons::Neurons, n::Int=1; Φ_init::Union{Float64, Nothing}=_Φ_init, dΦ_init=_dΦ_init, sigma=_sigma, a=_a, we=_we, wex=_wex, beta=_beta, bias=_bias)
function add_neurons!(neurons::Neurons, n::Int=1; Φ_init::Union{Float64, Nothing}=get_default_Φ_init(), dΦ_init=get_default_dΦ_init(), sigma=get_default_sigma(), a=get_default_a(), we=get_default_we(), wex=get_default_wex(), beta=get_default_beta(), bias=get_default_bias())
    # if Φ_init is nothing, then initialize it to the resting position calculated from bias - ϵ (or + ϵ)
    # if there is no resting positon due to bias being higher than some threshold, then initialize it to zero i guess
    
    # TODO: is it possible to calulcate the expected dΦ after cyclic behavior stabilizes? then I could
    # pick a value that makes sense for theta_init (i.e the value that would be observed if simulation ran for a long time, for a particular Φ)
    # however, if bias is too high and dampening is too low, then dΦ could explode, making the above pointless
    
    # determine Φ
    Φ_init_calculated = 0.0
    if isnothing(Φ_init)
        temp = 2*sigma * bias / we
        # threshold_curr = we / (2*sigma)
        if -1 <= temp <= 1
        # if abs(bias) <= abs(threshold_curr)
            Φ_init_calculated = (asin(temp)/2) - ϵ
        # else
        #     # TODO: calculate velocity here
        #     Φ_init_calculated = 0
        end
    else
        Φ_init_calculated = Φ_init
    end

    # concatinate new values onto neurons vectors
    neurons.Φ_init = vcat(neurons.Φ_init, fill(Φ_init_calculated, n))
    neurons.dΦ_init = vcat(neurons.dΦ_init, fill(dΦ_init, n))
    neurons.sigma = vcat(neurons.sigma, fill(sigma, n))
    neurons.a = vcat(neurons.a, fill(a, n))
    neurons.we = vcat(neurons.we, fill(we, n))
    neurons.wex = vcat(neurons.wex, fill(wex, n))
    neurons.beta = vcat(neurons.beta, fill(beta, n))
    neurons.bias = vcat(neurons.bias, fill(bias, n))
end

function build_neuron_params(root, gpu=false)
    sigma = map_component_array_depth_first(x->x.neurons.sigma, root)
    a = map_component_array_depth_first(x->x.neurons.a, root)
    we = map_component_array_depth_first(x->x.neurons.we, root)
    wex = map_component_array_depth_first(x->x.neurons.wex, root)
    beta = map_component_array_depth_first(x->x.neurons.beta, root)
    bias = map_component_array_depth_first(x->x.neurons.bias, root)

    if gpu
        sigma = cu(sigma)
        a = cu(a)
        we = cu(we)
        wex = cu(wex)
        beta = cu(beta)
        bias = cu(bias)
    end
    (sigma, a, we, wex, beta, bias)
end

function build_u0(root, gpu=false)
    Φ_init = map_component_array_depth_first(x->x.neurons.Φ_init, root)
    dΦ_init = map_component_array_depth_first(x->x.neurons.dΦ_init, root)

    if gpu
        Φ_init = CuArray{Float32}(Φ_init)
        dΦ_init = CuArray{Float32}(dΦ_init)
    end
    hcat(Φ_init, dΦ_init)
end

# reverse of building. take a state from the solution and apply it to the original neurons within root
function unbuild_u0!(root, u0::Matrix)
    Φ_init = view(u0, :, 1)
    dΦ_init = view(u0, :, 2)
    Φ_view_array = ArrayPartition(map_component_depth_first(x->view(x.neurons.Φ_init, :), root)...)
    dΦ_view_array = ArrayPartition(map_component_depth_first(x->view(x.neurons.dΦ_init, :), root)...)

    Φ_view_array .= Φ_init
    dΦ_view_array .= dΦ_init
    nothing
end