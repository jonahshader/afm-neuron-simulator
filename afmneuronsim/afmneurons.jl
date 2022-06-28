export set_defaults!


const TERA = 10e12
const GIGA = 10e9
const FEMTO = 10e-15

_fex = 27.5 * TERA
_a = 0.01
_fe = 1.75 * GIGA
_sigma = 2.16 * TERA

_wex = _fex * 2pi
_we = _fe * 2pi
_beta = 0.11 * FEMTO

_Φ_init = nothing
_dΦ_init = 0.0
_bias = 0.0023


const ϵ = 1e-8

# TODO: change Neurons to be a type alias on a Matrix with 8 rows and n cols.
# this makes it easier to perform operations on Neurons like concatination
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

function set_defaults!(Φ_init=nothing, dΦ_init=nothing, sigma=nothing, a=nothing, we=nothing, wex=nothing, beta=nothing, bias=nothing)
    if !isnothing(Φ_init)
        _Φ_init = Φ_init
    end
    if !isnothing(dΦ_init)
        _dΦ_init = dΦ_init
    end
    if !isnothing(sigma)
        _sigma = sigma
    end
    if !isnothing(a)
        _a = a
    end
    if !isnothing(we)
        _we = we
    end
    if !isnothing(wex)
        _wex = wex
    end
    if !isnothing(beta)
        _beta = beta
    end
    if !isnothing(bias)
        _bias = bias
    end
end

# neurons: the neurons struct that will be added to
# n: number of neurons to add with the specified values
function add_neurons!(neurons::Neurons, n::Int=1; Φ_init::Union{Float64, Nothing}=_Φ_init, dΦ_init=_dΦ_init, sigma=_sigma, a=_a, we=_we, wex=_wex, beta=_beta, bias=_bias)
    # if Φ_init is nothing, then initialize it to the resting position calculated from bias - ϵ (or + ϵ)
    # if there is no resting positon due to bias being higher than some threshold, then initialize it to zero i guess
    
    # TODO: is it possible to calulcate the expected dΦ after cyclic behavior stabilizes? then I could
    # pick a value that makes sense for theta_init (i.e the value that would be observed if simulation ran for a long time, for a particular Φ)
    # however, if bias is too high and dampening is too low, then dΦ could explode, making the above pointless
    
    # determine Φ
    Φ_init_calculated = 0.0
    if isnothing(Φ_init)
        temp = 2*sigma * bias / we
        if -1 <= temp <= 1
            Φ_init_calculated = (asin(temp)/2) - ϵ
        # else
        #     # TODO: calculate velocity here
        #     Φ_init_calculated = 0
        end
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

function build_neuron_params(root)
    sigma = map_component_array_depth_first(x->x.neurons.sigma, root)
    a = map_component_array_depth_first(x->x.neurons.a, root)
    we = map_component_array_depth_first(x->x.neurons.we, root)
    wex = map_component_array_depth_first(x->x.neurons.wex, root)
    beta = map_component_array_depth_first(x->x.neurons.beta, root)
    bias = map_component_array_depth_first(x->x.neurons.bias, root)
    (sigma, a, we, wex, beta, bias)
end

function build_u0(root)
    Φ_init = map_component_array_depth_first(x->x.neurons.Φ_init, root)
    dΦ_init = map_component_array_depth_first(x->x.neurons.dΦ_init, root)
    hcat(Φ_init, dΦ_init)
end