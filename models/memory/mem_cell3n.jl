set_defaults!(a=0.0125, bias=0.0002)

scale = 0.7
off_scale = -1.0
mem_cell = Component(["on", "off"], ["out"])
add_neurons!(mem_cell, 4)

set_weight!(mem_cell, "on", (1,), scale)
set_weight!(mem_cell, "on", (4,), scale)

set_weight!(mem_cell, "off", (1,), off_scale)
set_weight!(mem_cell, "off", (2,), off_scale)
set_weight!(mem_cell, "off", (3,), off_scale)
set_weight!(mem_cell, "off", (4,), off_scale)

set_weight!(mem_cell, (1,), (2,), scale)
set_weight!(mem_cell, (2,), (1,), scale)

set_weight!(mem_cell, (3,), (4,), scale)
set_weight!(mem_cell, (4,), (3,), scale)

set_weight!(mem_cell, (1,), "out", 1.0)
set_weight!(mem_cell, (3,), "out", 1.0)