include("../afmneuronsim/includes.jl")

function run_model()
    model = make_chain([3, 10, 10, 10, 10, 10, 10, 4]; init_scale=sqrt(6))
    build_and_solve(model, (0.0, 400 * PICO), input_to_spikes(ones(Float64, 3)))
end