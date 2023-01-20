using Flux

# TODO: determine weight scalar
include("../afmneuronsim/metric_prefixes.jl")
include("../afmneuronsim/afmneurons.jl")

struct AFMLayer
    fc::Dense
    bias::Float32
    sigma::Float32
    a::Float32
    we::Float32
    wex::Float32
    beta::Float32
end

Flux.@functor AFMLayer
Flux.trainable(m::AFMLayer) = (m.fc,)

function AFMLayer((in, out)::Pair{<:Integer, <:Integer}; local_bias = false, a = 0.1f0, we = Float32(_fe * 2pi), 
    wex = Float32(_fex * 2pi), beta = 0.11f-15, sigma = 27.1f12, bias_ratio = 0.97f0, bias = bias_ratio * we / (2 * sigma))
    fc = Dense(in => out, bias = local_bias)
    # TODO: scale fc weights

    return AFMLayer(fc, bias, sigma, a, we, wex, beta)
end


function (m::AFMLayer)(Φ, dΦ)
    voltage = dΦ .* m.beta
    current = m.fc(voltage) .+ m.bias
    dudΦ = (m.sigma .* current .- m.a .* dΦ .- (m.we./2) .* sin.(2 .* Φ)) .* m.wex
    return dΦ, dudΦ
end