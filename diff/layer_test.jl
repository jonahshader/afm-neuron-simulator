include("afmlayer.jl")
using DifferentialEquations, Flux, DiffEqFlux, Plots


model = AFMLayer(16 => 16)

p, re = Flux.destructure(model)
ps = Flux.params(p)

u0 = hcat(randn(Float32, 16, 1, 1), zeros(Float32, 16, 1, 1))
tspan = (0f0, 300f-12)

function diffeq(u, p, t)
    # TODO: might be more performant to put batches in the second to last dim
    Φ = u[:, 1, :]
    dΦ = u[:, 2, :]

    hcat(re(p)(Φ, dΦ))
end

function forward(x, args...; kwargs...)
    ff = ODEFunction{false}(diffeq, tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob, args...; sensealg=sense, kwargs...)
end

cb = function ()

end

opt = Adam()
data = Iterators.repeated((), 1000)

