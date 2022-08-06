# the mem cell contains four neurons, two pairs are connected to each other so that
# they can oscillate independently. these two oscillators can be activated offset
# by 180 degrees so that when their outputs are combined, we get a constant output
# instead of pulses

set_defaults!(a=0.0125, bias=0.00019)

scale = 0.9
off_scale = -1.0
mem_cell = Component(["on", "off"], ["out"])
add_neurons!(mem_cell, 4)
add_neuron!(mem_cell, "smooth", a=0.02, bias=0.0002)

set_weight!(mem_cell, "on", (1,), scale)
set_weight!(mem_cell, "on", (4,), scale)

set_weight!(mem_cell, "off", (1,), off_scale)
set_weight!(mem_cell, "off", (2,), off_scale)
set_weight!(mem_cell, "off", (3,), off_scale)
set_weight!(mem_cell, "off", (4,), off_scale)
set_weight!(mem_cell, "off", ("smooth",), off_scale)

set_weight!(mem_cell, (1,), (2,), scale)
set_weight!(mem_cell, (2,), (1,), scale)

set_weight!(mem_cell, (3,), (4,), scale)
set_weight!(mem_cell, (4,), (3,), scale)


set_weight!(mem_cell, (1,), ("smooth",), 5.0)
set_weight!(mem_cell, (3,), ("smooth",), 5.0)

set_weight!(mem_cell, ("smooth",), "out", 1.0)
