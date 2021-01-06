using ForwardDiff: gradient, jacobian, hessian

"""
```
function dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T}) where T
    ...
    return xᵢ₊₁
end
```
"""
function linearize_dynamics(
    x̅::AbstractMatrix{T},
    u̅::AbstractMatrix{T},
    dynamicsf::Function,
) where {T}
    N, control_size = size(u̅)
    state_size = size(x̅)[2]

    𝐀s = zeros(T, N, state_size, state_size)
    𝐁s = zeros(T, N, state_size, control_size)

    # Declaring dynamics jacobian functions
    A_func(x̅ᵢ, u̅ᵢ) = jacobian(x̅ᵢ -> dynamicsf(x̅ᵢ, u̅ᵢ), x̅ᵢ)
    B_func(x̅ᵢ, u̅ᵢ) = jacobian(u̅ᵢ -> dynamicsf(x̅ᵢ, u̅ᵢ), u̅ᵢ)

    for k = 1:N
        # Computing jacobian around each point
        𝐀s[k, :, :] .= A_func(x̅[k, :], u̅[k, :])
        𝐁s[k, :, :] .= B_func(x̅[k, :], u̅[k, :])
    end

    return (𝐀s, 𝐁s)
end


@doc raw"""
```julia
function immediate_cost(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T})
    return sum(uᵢ.^2)  # for example
end
```
"""
function cost_quadratization(
    x̅::AbstractMatrix{T},
    u̅::AbstractMatrix{T},
    immediate_cost::Function,
    final_cost::Function,
) where {T}
    N, control_size = size(u̅)
    state_size = size(x̅)[2]

    # Notation copied from ETH lecture notes
    𝑞s = zeros(T, N + 1)  # Cost along path
    𝐪s = zeros(T, N + 1, state_size)  # Cost Jacobian wrt x
    𝐫s = zeros(T, N, control_size)  # Cost Jacobian wrt u
    𝐐s = zeros(T, N + 1, state_size, state_size)  # Cost Hessian wrt x, x
    𝐏s = zeros(T, N, control_size, state_size)  # Cost Hessian wrt u, x
    𝐑s = zeros(T, N, control_size, control_size)  # Cost Hessian wrt u, u

    # Helper jacobain functions
    ∂L∂x(x̅ᵢ, u̅ᵢ) = gradient(x̅ᵢ -> immediate_cost(x̅ᵢ, u̅ᵢ), x̅ᵢ)
    ∂L∂u(x̅ᵢ, u̅ᵢ) = gradient(u̅ᵢ -> immediate_cost(x̅ᵢ, u̅ᵢ), u̅ᵢ)
    ∂²L∂x²(x̅ᵢ, u̅ᵢ) = hessian(x̅ᵢ -> immediate_cost(x̅ᵢ, u̅ᵢ), x̅ᵢ)
    ∂²L∂u∂x(x̅ᵢ, u̅ᵢ) = jacobian(x̅ᵢ -> ∂L∂u(x̅ᵢ, u̅ᵢ), x̅ᵢ)
    ∂²L∂u²(x̅ᵢ, u̅ᵢ) = hessian(u̅ᵢ -> immediate_cost(x̅ᵢ, u̅ᵢ), u̅ᵢ)
    ∂Φ∂x(x̅ᵢ) = gradient(x̅ᵢ -> final_cost(x̅ᵢ), x̅ᵢ)
    ∂²Φ∂x²(x̅ᵢ) = hessian(x̅ᵢ -> final_cost(x̅ᵢ), x̅ᵢ)

    for k = 1:N
        # Cost along path
        𝑞s[k] = immediate_cost(x̅[k, :], u̅[k, :])
        # Cost Jacobian wrt x
        𝐪s[k, :] .= ∂L∂x(x̅[k, :], u̅[k, :])
        # Cost Jacobian wrt u
        𝐫s[k, :] .= ∂L∂u(x̅[k, :], u̅[k, :])
        # Cost Hessian wrt x, x
        𝐐s[k, :, :] .= ∂²L∂x²(x̅[k, :], u̅[k, :])
        # Cost Hessian wrt u, x
        𝐏s[k, :, :] .= ∂²L∂u∂x(x̅[k, :], u̅[k, :])
        # Cost Hessian wrt u, u
        𝐑s[k, :, :] .= ∂²L∂u²(x̅[k, :], u̅[k, :])
    end
    # Final cost
    𝑞s[N+1] = final_cost(x̅[end, :])
    # Final cost Jacobian wrt x
    𝐪s[N+1, :] = ∂Φ∂x(x̅[end, :])
    # Final cost Hessian wrt x
    𝐐s[N+1, :, :] = ∂²Φ∂x²(x̅[end, :])

    return (𝑞s, 𝐪s, 𝐫s, 𝐐s, 𝐏s, 𝐑s)
end


@doc raw"""
"""
function optimal_controller_param(
    𝐀ᵢ::AbstractMatrix{T},
    𝐁ᵢ::AbstractMatrix{T},
    𝐫ᵢ::AbstractMatrix{T},
    𝐏ᵢ::AbstractMatrix{T},
    𝐑ᵢ::AbstractMatrix{T},
    𝐬ᵢ₊₁::AbstractMatrix{T},
    𝐒ᵢ₊₁::AbstractMatrix{T},
) where {T}
    control_size, state_size = size(𝐏)

    𝐠ᵢ = 𝐫ᵢ + transpose(𝐁ᵢ) * 𝐬ᵢ₊₁
    𝐆ᵢ = 𝐏ᵢ + transpose(𝐁ᵢ) * 𝐒ᵢ₊₁ * 𝐀ᵢ
    𝐇ᵢ = 𝐑ᵢ + transpose(𝐁ᵢ) * 𝐒ᵢ₊₁ * 𝐁ᵢ

    return (𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)
end


@doc raw"""
"""
function feedback_parameters(
    𝐠ᵢ::AbstractMatrix{T},
    𝐆ᵢ::AbstractMatrix{T},
    𝐇ᵢ::AbstractMatrix{T},
) where {T}
    𝛿𝐮ᵢᶠᶠ = -inv(𝐇ᵢ) * 𝐠ᵢ
    𝐊ᵢ = -inv(𝐇ᵢ) * 𝐆ᵢ
    return (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ)
end


@doc raw"""
"""
function back_one_step(
    𝐀ᵢ::AbstractMatrix{T},
    𝐁ᵢ::AbstractMatrix{T},
    𝑞ᵢ::AbstractVector{T},
    𝐪ᵢ::AbstractVector{T},
    𝐫ᵢ::AbstractVector{T},
    𝐐ᵢ::AbstractMatrix{T},
    𝐏ᵢ::AbstractMatrix{T},
    𝐑ᵢ::AbstractMatrix{T},
    𝑠ᵢ₊₁::AbstractVector{T},
    𝐬ᵢ₊₁::AbstractVector{T},
    𝐒ᵢ₊₁::AbstractMatrix{T},
) where {T}
    # Compute controller constants
    (𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ) = optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)
    # Compute controller gains
    (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ) = feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)

    𝑠ᵢ = (𝑞ᵢ + 𝑠ᵢ₊₁ + 1 / 2 * transpose(𝛿𝐮ᵢᶠᶠ) * 𝐇ᵢ * 𝛿𝐮ᵢᶠᶠ + transpose(𝛿𝐮ᵢᶠᶠ) * 𝐠ᵢ)
    𝐬ᵢ = (
        𝐪ᵢ +
        transpose(𝐀ᵢ) * 𝐬ᵢ₊₁ +
        transpose(𝐊ᵢ) * 𝐇ᵢ * 𝛿𝐮ᵢᶠᶠ +
        transpose(𝐊ᵢ) * 𝐠ᵢ +
        transpose(𝐆ᵢ) * 𝛿𝐮ᵢᶠᶠ
    )
    𝐒ᵢ = (
        𝐐ᵢ +
        transpose(𝐀ᵢ) * 𝐒ᵢ₊₁ * 𝐀ᵢ +
        transpose(𝐊ᵢ) * 𝐇ᵢ * 𝐊ᵢ +
        transpose(𝐊ᵢ) * 𝐆ᵢ +
        transpose(𝐆ᵢ) * 𝐊ᵢ
    )

    return (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ, 𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ)
end



@doc raw"""

```julia
function dynamics(xᵢ::AbstractArray{T,1}, uᵢ::AbstractArray{T,1})
    ...
    return xᵢ₊₁
end
```

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
function backward_pass(
    x̅::AbstractMatrix{T},
    u̅::AbstractMatrix{T},
    dynamicsf::Function,
    immediate_cost::Function,
    final_cost::Function,
) where {T}
    # Linearize dynamics around each step
    (𝐀s, 𝐁s) = linearize_dynamics(x̅, u̅, dynamicsf)
    # Compute the Quadratization of the cost at each time step
    (𝑞s, 𝐪s, 𝐫s, 𝐐s, 𝐏s, 𝐑s) = cost_quadratization(x̅, u̅, immediate_cost, final_cost)
    # Grab all dimensions
    N, control_size, state_size = size(𝐏s)
    # Initialize matricies
    𝛿𝐮ᶠᶠs = zeros(T, N, control_size)
    𝐊s = zeros(T, N, control_size, state_size)

    (𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁) = (𝑞s[end], 𝐪s[end], 𝐐s[end])
    # Move backward
    for i = N:1
        (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ, 𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ) = back_one_step(
            𝐀s[i],
            𝐁s[i],
            𝑞s[i],
            𝐪s[i],
            𝐫s[i],
            𝐐s[i],
            𝐏s[i],
            𝐑s[i],
            𝑠ᵢ₊₁,
            𝐬ᵢ₊₁,
            𝐒ᵢ₊₁,
        )
        𝛿𝐮ᶠᶠs[i] .= 𝛿𝐮ᵢᶠᶠ
        𝐊s[i] .= 𝐊ᵢ
        (𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁) = (𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ)
    end

    return (𝛿𝐮ᶠᶠs, 𝐊s)
end
