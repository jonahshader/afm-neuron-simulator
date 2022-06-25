include("afmneuronsim/includes.jl")

c = Component(3, 3)
add_neurons!(c, 3)

for s in sources(c)
    for d in destinations(c)
        set_weight!(c, s, d, randn() * 0.2)
    end
end

parts = build_model_parts(c, (0.0, 8e-12), input_to_spikes([1.0, 0.5, 0.0]))

solve_parts!(parts)

c_single_output = Component(3, 1)
add_neurons!(c_single_output, 3)



for s in sources(c_single_output)
    for d in destinations(c_single_output)
        set_weight!(c_single_output, s, d, randn() * 0.2)
    end
end

parts_single_output = build_model_parts(c_single_output, (0.0, 8e-12), input_to_spikes([1.0, 0.5, 0.0]))

solve_parts!(parts_single_output)

