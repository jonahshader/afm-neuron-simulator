include("models/fulladder.jl")

function full_adder_trainer()
    function mean_of_solution_squared(u0)
        _prob = remake(ode_problem(parts), u0=u0)
        _sol = solve(ode_problem(parts), Tsit5())

        sum(_sol .^ 2) / length(_sol)
    end

    dud01 = Zygote.gradient(mean_of_solution_squared, u0)
end