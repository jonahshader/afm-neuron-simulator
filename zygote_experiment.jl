# using Zygote
using DifferentialEquations
using Plots

tspan = (0.0,1.0)

# A  = [1. 0  0 -5
#       4 -2  4 -3
#      -4  0  0  1
#       5 -2  2  3]



# A = randn(4, 4)
# u0 = rand(4,1)

# f(u,p,t) = A*u
# prob = ODEProblem(f,u0,tspan)
# sol = solve(prob)
# plot(sol)

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