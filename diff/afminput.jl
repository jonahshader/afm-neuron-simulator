using Flux

struct AFMInput
    bias::Float32
    sigma::Float32
    a::Float32
    we::Float32
    wex::Float32
    beta::Float32
end

Flux.@functor AFMLayer
Flux.trainable(m::AFMModel) = ()

function AFMLayer(; local_bias = false, a = 0.1f0, we = Float32(_fe * 2pi), 
    wex = Float32(_fex * 2pi), beta = 0.11f-15, sigma = 27.1f12, bias_ratio = 0.97f0, bias = bias_ratio * we / (2 * sigma))
    fc = Dense(in => out, bias = local_bias)
    # TODO: scale fc weights

    return AFMLayer(fc, bias, sigma, a, we, wex, beta)
end