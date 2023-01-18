using Flux

# TODO: determine weight scalar

struct AFMLayer
    fc::Dense
    bias::AbstractFloat
    a::AbstractFloat
end

Flux.@functor AFMLayer




function (m::AFMLayer)(Φ, dΦ)
    voltage = dΦ .* beta
    current = m.fc(voltage) .+ m.bias
    dudΦ = (sigma .* current .- m.a .* dΦ .- (we./2) .* sin.(2 .* Φ)) .* wex
    return dΦ, dudΦ
end