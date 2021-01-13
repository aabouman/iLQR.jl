@doc raw"""
`forward_pass(x, u, 𝛿𝐮ᶠᶠs, 𝐊s, prev_cost, dynamicsf, immediate_cost, final_cost)`

Perform iterativer LQR to compute optimal input and corresponding state
trajectory.

# Arguments
- `x::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `u::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `𝛿𝐮ᶠᶠs::AbstractMatrix{T}`: see output of [`backward_pass(x, u, dynamicsf, immediate_cost, final_cost)`](@ref)
- `𝐊s::AbstractArray{T,3}`: see output of [`backward_pass(x, u, dynamicsf, immediate_cost, final_cost)`](@ref)
- `dynamicsf::Function`: dynamic function, steps the system forward
- `immediate_cost::Function`: Cost after each step
- `final_cost::Function`: Cost after final step

The `dynamicsf` steps the system forward (``x_{i+1} = f(x_i, u_i)``). The
function expects input of the form:
```julia
function dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T}) where T
    ...
    return xᵢ₊₁
end
```

The `immediate_cost` function expect input of the form:
```julia
function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
    return sum(u.^2) + sum(target_state - x.^2)  # for example
end
```
!!! note
    It is important that the function `immediate_cost` be an explict function
    of both `x` and `u` (due to issues using `ForwardDiff` Package). If you want
    to make `immediate_cost` practically only dependent on `u` with the following

    ```julia
    function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
        return sum(u.^2) + sum(x) * 0.0  # Only dependent on u
    end
    ```

The `final_cost` function expect input of the form:
```julia
function final_cost(x::AbstractVector{T})
    return sum(target_state - x.^2)  # for example
end
```

Returns the optimal trajectory ``(x̅, u̅)``
"""
function forward_pass(x::AbstractMatrix{T}, u::AbstractMatrix{T},
                      𝛿𝐮ᶠᶠs::AbstractMatrix{T}, 𝐊s::AbstractArray{T,3},
                      prev_cost::T, dynamicsf::Function,
                      immediate_cost::Function, final_cost::Function
                      ) where {T}
    M, input_size = size(u); N, state_size = size(x)
    @assert(N == M+1)

    x̅ = zeros(T, N, state_size); u̅ = zeros(T, N-1, input_size)
    x̅[1, :] .= x[1, :]
    α = 1.0     # Learning rate
    total_cost = total_cost_generator(immediate_cost, final_cost)
    new_cost = 0.0

    while true
        for k = 1:N-1
            δxᵢ = x̅[k,:] - x[k,:]
            u̅[k,:] .= u[k,:] + α * 𝛿𝐮ᶠᶠs[k,:] + 𝐊s[k,:,:] * δxᵢ
            x̅[k+1,:] .= dynamicsf(x̅[k,:], u̅[k,:])
        end
        new_cost = total_cost(x̅, u̅)
        Δcost = prev_cost - new_cost

        if (Δcost > 0)    # Check if we should persue line search
            break
        else    # Catch overshoot as well as NaNs
            α /= 2
            display("Were line searchin'")
            display(α)
            display(any(isnan, u̅))
        end
    end

    @assert !any(isnan, u̅)
    @assert !any(isnan, x̅)

    return (x̅, u̅, new_cost)
end


@doc raw"""
`fit(x_init, u_init, dynamicsf, immediate_cost, final_cost; max_iter=100, tol=1e-6)`

Perform iterativer LQR to compute optimal input and corresponding state
trajectory.

# Arguments
- `x_init::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `u_init::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `dynamicsf::Function`: dynamic function, steps the system forward
- `immediate_cost::Function`: Cost after each step
- `final_cost::Function`: Cost after final step
- `max_iter::Int64=100`: Maximum number of forward/backward passes to make
- `tol::Float64=1e-6`: Specifies the tolerance at which to consider the input
trajectory has converged

The `dynamicsf` steps the system forward (``x_{i+1} = f(x_i, u_i)``). The
function expects input of the form:
```julia
function dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T}) where T
    ...
    return xᵢ₊₁
end
```

The `immediate_cost` function expect input of the form:
```julia
function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
    return sum(u.^2) + sum(target_state - x.^2)  # for example
end
```
!!! note
    It is important that the function `immediate_cost` be an explict function
    of both `x` and `u` (due to issues using `ForwardDiff` Package). If you want
    to make `immediate_cost` practically only dependent on `u` with the following

    ```julia
    function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
        return sum(u.^2) + sum(x) * 0.0  # Only dependent on u
    end
    ```

The `final_cost` function expect input of the form:
```julia
function final_cost(x::AbstractVector{T})
    return sum(target_state - x.^2)  # for example
end
```

Returns the optimal trajectory ``(x̅, u̅)``
"""
function fit(x_init::AbstractMatrix{T}, u_init::AbstractMatrix{T},
             dynamicsf::Function, immediate_cost::Function,
             final_cost::Function; max_iter::Int64=100, tol::Float64=1e-6,
             ) where {T}
    x̅ⁱ = x_init; u̅ⁱ = u_init
    N, state_size = size(x̅ⁱ); M, input_size = size(u̅ⁱ)
    @assert(N == M + 1, "size(x_init)[2] == size(u_init)[1], (# of states is 1 more than # of inputs in trajectory)")
    total_cost = total_cost_generator(immediate_cost, final_cost)

    prev_cost = Inf; new_cost = NaN
    iter = 0
    for iter = 1:max_iter
        𝛿𝐮ᶠᶠs, 𝐊s = backward_pass(x̅ⁱ, u̅ⁱ, dynamicsf, immediate_cost, final_cost)
        x̅ⁱ⁺¹, u̅ⁱ⁺¹, new_cost = forward_pass(x̅ⁱ, u̅ⁱ, 𝛿𝐮ᶠᶠs, 𝐊s, prev_cost,
                                            dynamicsf, immediate_cost,
                                            final_cost)
        @assert(prev_cost > new_cost); prev_cost = new_cost

        # Check if we have met the tolerance for convergence
        display(size(u̅ⁱ))
        display(size(u̅ⁱ⁺¹))

        convert(Float64, sum((u̅ⁱ⁺¹ - u̅ⁱ).^2)) <= tol && break

        # Update the current trajectory and input estimates
        x̅ⁱ = x̅ⁱ⁺¹
        u̅ⁱ = u̅ⁱ⁺¹
    end

    return (x̅ⁱ, u̅ⁱ)
end

@doc raw"""
`backward_pass(x, u, dynamicsf, immediate_cost, final_cost)`

Computes feedforward and feedback gains (``𝛿𝐮ᵢᶠᶠ`` and ``𝐊ᵢ``).

# Arguments
- `immediate_cost::Function`: Cost after each step
- `final_cost::Function`: Cost after final step

The `immediate_cost` function expect input of the form:
```julia
function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
    return sum(u.^2) + sum(target_state - x.^2)  # for example
end
```
!!! note
    It is important that the function `immediate_cost` be an explict function
    of both `x` and `u` (due to issues using `ForwardDiff` Package). If you want
    to make `immediate_cost` practically only dependent on `u` with the following

    ```julia
    function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
        return sum(u.^2) + sum(x) * 0.0  # Only dependent on u
    end
    ```

The `final_cost` function expect input of the form:
```julia
function final_cost(x::AbstractVector{T})
    return sum(target_state - x.^2)  # for example
end
```

Returns a function `total_cost` which computes the total cost of a state and
input trajectory.
"""
function total_cost_generator(immediate_cost::Function, final_cost::Function)
    function total_cost(x̅ⁱ, u̅ⁱ)
        N = size(u̅ⁱ)[1]
        sum = 0.

        for i in 1:N
            sum += immediate_cost(x̅ⁱ[i,:], u̅ⁱ[i,:])
        end
        sum += final_cost(x̅ⁱ[end,:])
    end

    return total_cost
end
