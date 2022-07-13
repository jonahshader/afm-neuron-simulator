using DifferentialEquations
using Plots
using Zygote
using DiffEqSensitivity

function nn_test()
    tspan = (0.0, 1.0)

    scalar = 3.0
    io_size = 4
    hidden_size = 64
    A = randn(hidden_size, io_size) * scalar
    B = randn(hidden_size, hidden_size) * scalar
    C = randn(io_size, hidden_size) * scalar
    
    Abias = randn(hidden_size) * scalar * 0
    Bbias = randn(hidden_size) * scalar * 0
    Cbias = randn(io_size) * scalar * 0
    
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

function circle_test()
    Θ = pi/3
    rotation_matrix = 
    [cos(Θ) -sin(Θ)
     sin(Θ) cos(Θ)]
    velocity = pi

    tspan = (0.0, 3.0)
    u0 = [1.0, 0.0]
    f(u,p,t) = p * u * velocity
    # prob = ODEProblem(f,u0,tspan)
    prob = ODEProblem(f, u0, tspan, rotation_matrix)
    sol = solve(prob, Tsit5(),reltol=1e-6,abstol=1e-6)
    
    # plots = []

    function sum_of_solution(u0, p)
        _prob = remake(prob,u0=u0,p=p)
        _sol = solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1,sensealg=QuadratureAdjoint())
        # plot!(_sol)
        sum(_sol .^ 2)
    end
    for i in 1:100
        dud01,dp1 = Zygote.gradient(sum_of_solution, u0, rotation_matrix)

        prob = ODEProblem(f, u0, tspan, rotation_matrix)
        sol = solve(prob, Tsit5(),reltol=1e-6,abstol=1e-6)
        # push!(plots, plot(sol, vars=(1,2)))
        display(plot(sol, vars=(1,2)))

        rotation_matrix .-= dp1 * 0.001 
    end
    

    plots
end