@doc raw"""
```julia
function immediate_cost(x::AbstractArray{T,2}, u::AbstractArray{T,2})
    sum(u.^2)  # for example
end
```

```julia
function final_cost(xₙ::AbstractArray{T,1})
    sum((some_target_point - xₙ).^2)  # Euclidean distance at end, for example
end
```
"""
function forward_pass(
    mechanism::Mechanism,
    x::AbstractMatrix,
    u::AbstractMatrix,
    K,
    d,
    ΔV,
    J,
    immediate_cost::Function,
    final_cost::Function,
    time::Real,
)
    # TODO: Look into using DifferentialEquations.jl callback inside for loop
    #       rather than regenerating a new ODEProblem each loop
    # x̅ = zeros(size(x)); u̅ = zeros(size(u));
    Δt = time / (size(x)[1])

    x̅[1] = x[1]
    α = 1
    state = MechanismState(mechanism)

    for k = 1:N
        u̅[k, :] = u[k, :] + K * (x̅[k, :] - x[k, :]) + α * d[k]

        # Set constant control signal function
        const_control!(τ::AbstractVector, t, state) = τ .= U̅[k]
        fdynamics = Dynamics(mechanism, const_control!)
        problem = ODEProblem(fdynamics, state, (0.0, Δt))
        sol = solve(problem)
        set_configuration!(state, sol.u[end][1:6])
        set_velocity!(state, sol.u[end][7:end])

        x̅[k+1] = state, sol.u[end][1:6]
    end
    J = cost()


end
