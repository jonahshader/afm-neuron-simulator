const TERA = 10e12
const GIGA = 10e9
const FEMTO = 10e-15

const _fex = 27.5 * TERA
const _a = 0.01
const _fe = 1.75 * GIGA
const _sigma = 2.16 * TERA

const _wex = _fex * 2pi
const _we = _fe * 2pi
const _beta = 0.11 * FEMTO

const ϵ = 1e-8
# TODO: change Neurons to be a type alias on a Matrix with 8 rows and n cols.
# this makes it easier to perform operations on Neurons like concatination
mutable struct Neurons
    θ_init::Vector{Float64}
    dθ_init::Vector{Float64} 
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
    Base.length(n.θ_init)
end

# neurons: the neurons struct that will be added to
# n: number of neurons to add with the specified values
function add_neurons!(neurons::Neurons, n::Int=1, θ_init::Float64=nothing, dθ_init=0.0, sigma=_sigma, a=_a, we=_we, wex=_wex, beta=_beta, bias=0.0023)
    # if θ_init is nothing, then initialize it to the resting position calculated from bias - ϵ (or + ϵ)
    # if there is no resting positon due to bias being higher than some threshold, then initialize it to zero i guess
    
    # TODO: is it possible to calulcate the expected dθ after cyclic behavior stabilizes? then I could
    # pick a value that makes sense for theta_init (i.e the value that would be observed if simulation ran for a long time, for a particular θ)
    # however, if bias is too high and dampening is too low, then dθ could explode, making the above pointless
    
    # determine θ
    θ_init_calculated = θ_init
    if isnothing(θ_init_calculated)
        temp = 2*sigma * bias / we
        if -1 <= temp <= 1
            θ_init_calculated = (asin(temp)/2) - ϵ
        else
            θ_init_calculated = 0
        end
    end

    # concatinate new values onto neurons vectors
    neurons.θ_init = vcat(neurons.θ_init, fill(θ_init_calculated, n))
    neurons.dθ_init = vcat(neurons.dθ_init, fill(dθ_init, n))
    neurons.sigma = vcat(neurons.sigma, fill(sigma, n))
    neurons.a = vcat(neurons.a, fill(a, n))
    neurons.we = vcat(neurons.we, fill(we, n))
    neurons.wex = vcat(neurons.wex, fill(wex, n))
    neurons.beta = vcat(neurons.beta, fill(beta, n))
    neurons.bias = vcat(neurons.bias, fill(bias, n))
end
