
set_defaults!(a=0.01, bias=0.0002)

scale = 0.7
and = Component(["clock", "a", "b"], ["clock", "out"])
add_neuron!(and, "majority")
add_neuron!(and, "clock_prop")

set_weight!(and, "clock", ("clock_prop",), scale)
set_weight!(and, ("clock_prop",), "clock", 1.0)

set_weight!(and, "a", ("majority",), scale/2)
set_weight!(and, "b", ("majority",), scale/2)
set_weight!(and, ("majority",), "out", 1.0)