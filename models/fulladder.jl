include("../afmneuronsim/includes.jl")

# defines and returns the model, which is a component
function make_model()
    # define top level component. this component contains other components along with the top level IO.
    full_adder = Component(["a", "b", "c-in"], ["sum", "c-out"])
    # populate full_adder with the nessesary logic gate components
    xor = Component(2, 1)
    add_component!(full_adder, xor, "xor1")
    add_component!(full_adder, xor, "xor2")
    and = Component(2, 1)
    add_component!(full_adder, and, "and1")
    add_component!(full_adder, and, "and2")
    or = Component(2, 1)
    add_component!(full_adder, or, "or1")
    
    # connect the components within full_adder together
    set_weight!(full_adder, "a", ("xor1", 1), 1.0)
    set_weight!(full_adder, "a", ("and2", 1), 1.0)
    set_weight!(full_adder, "b", ("xor1", 2), 1.0)
    set_weight!(full_adder, "b", ("and2", 2), 1.0)
    set_weight!(full_adder, "c-in", ("xor2", 2), 1.0)
    set_weight!(full_adder, "c-in", ("and1", 1), 1.0)
    set_weight!(full_adder, ("xor1", 1), ("xor2", 1), 1.0)
    set_weight!(full_adder, ("xor1", 1), ("and1", 2), 1.0)
    set_weight!(full_adder, ("xor2", 1), "sum", 1.0)
    set_weight!(full_adder, ("and1", 1), ("or1", 1), 1.0)
    set_weight!(full_adder, ("and2", 1), ("or1", 2), 1.0)
    set_weight!(full_adder, ("or1", 1), "c-out", 1.0)
    
    # define the and gate
    add_neurons!(and, 1)
    set_weight!(and, 1, (1,), .5)
    set_weight!(and, 2, (1,), .5)
    set_weight!(and, (1,), 1, 1.0)
    
    # define the or gate
    add_neurons!(or, 1)
    set_weight!(or, 1, (1,), 1.0)
    set_weight!(or, 2, (1,), 1.0)
    set_weight!(or, (1,), 1, 1.0)
    
    # xor is defined by combining the and and or gates, so add those to the xor component
    add_component!(xor, and, "and1")
    add_component!(xor, or, "or1")

    # the neurons in xor must pass through multiple weights to get to the next neuron or output
    # since we don't want to manipulate the signal twice, we use :pass to ensure the weight is 1.0 after scaling
    # :pass_inv is non-scaling but negating
    set_weight!(xor, 1, ("and1", 1), :pass)
    set_weight!(xor, 2, ("and1", 2), :pass)
    set_weight!(xor, 1, ("or1", 1), :pass)
    set_weight!(xor, 2, ("or1", 2), :pass)
    set_weight!(xor, ("and1", 1), 1, :pass_inv)
    set_weight!(xor, ("or1", 1), 1, :pass)

    return full_adder
end

# run returns solved parts
function run()
    full_adder = make_model()
    input_funs = input_to_spikes([1.0, 1.0, 1.0])
    ts = (0.0, 400 * PICO)
    return build_and_solve(full_adder, ts, input_funs, dense=false)
end

# plot_dΦ(run())
## or using "pipe" or function composition notation:
# run() |> plot_dΦ



