using DifferentialEquations
using Plots
using Zygote
using DiffEqSensitivity
using LinearAlgebra

function nn_test()
    tspan = (0.0, 1.0)

    scalar = 3.0
    io_size = 5
    hidden_size = 640
    A = randn(hidden_size, io_size) * scalar
    B = randn(hidden_size, hidden_size) * scalar
    C = randn(io_size, hidden_size) * scalar
    
    Abias = randn(hidden_size) * scalar
    Bbias = randn(hidden_size) * scalar
    Cbias = randn(io_size) * scalar
    
    # f(u,p,t) = C*tanh.(B*tanh.((A*u)))
    f(u,p,t) = C*tanh.((B*tanh.((A*u) + Abias)) + Bbias ) + Cbias
    u0 = randn(io_size, 1)
    
    # junk = [[1.0, 2.0], [3.0, 4.0, 5.0]]
    
    # f(u,p,t) = junk[1][1] * u
    # u0 = 1.0
    
    
    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob)
    plot(sol)    
end

function nn_train_test()
    tspan = (0.0, 1.0)
    init_weight_scale = 3.0
    io_size = 5
    hidden_size = 64
    A = randn(hidden_size, io_size) * init_weight_scale
    B = randn(hidden_size, hidden_size) * init_weight_scale
    C = randn(io_size, hidden_size) * init_weight_scale

    Abias = randn(hidden_size) * init_weight_scale
    Bbias = randn(hidden_size) * init_weight_scale
    Cbias = randn(io_size) * init_weight_scale

    params = (A, B, C, Abias, Bbias, Cbias)

    f(u,p,t) = C*tanh.((B*tanh.((A*u) + Abias)) + Bbias ) + Cbias
    u0 = randn(io_size, 1)

    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob)
    plot(sol)


    function mean_of_solution_squared(u0, p)
        _prob = remake(prob, u0=u0, p=p)
        _sol = solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.01, sensealg=QuadratureAdjoint())
        sum(((_sol .|> sin) - (_sol.t .|> sin)) .^ 2) / _sol.t.size
    end
    # for i in 1:15
    #     dud01,dp1 = Zygote.gradient(mean_of_solution_squared, u0, )
end

function circle_test(use_inplace::Bool=false)
    Θ = pi/2
    rotation_matrix = 
    [cos(Θ) -sin(Θ)
     sin(Θ) cos(Θ)]
    velocity = pi

    tspan = (0.0, 3.0)
    u0 = [1.0, 0.0]
    f(u,p,t) = p * u * velocity
    # prob = ODEProblem(f,u0,tspan)
    # cache = rotation_matrix * u0 * velocity
    function f_inplace!(du,u,p,t)
        mul!(du, p, u)
        du .*= velocity
        nothing
    end
    prob = ODEProblem(if use_inplace f_inplace! else f end, u0, tspan, rotation_matrix)
    sol = solve(prob, Tsit5(),reltol=1e-6,abstol=1e-6)
    
    # plots = []

    function mean_of_solution_squared(u0, p)
        _prob = remake(prob,u0=u0,p=p)
        _sol = solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.01,sensealg=QuadratureAdjoint())
        # plot!(_sol)
        sum(_sol .^ 2) / length(_sol)
    end
    for i in 1:15
        dud01,dp1 = Zygote.gradient(mean_of_solution_squared, u0, rotation_matrix)

        prob = ODEProblem(f, u0, tspan, rotation_matrix)
        sol = solve(prob, Tsit5(),reltol=1e-6,abstol=1e-6)
        # push!(plots, plot(sol, vars=(1,2)))
        display(plot(sol, vars=(1,2)))

        rotation_matrix .-= dp1 * 0.01 
    end
end
