using ForwardDiff: gradient, jacobian, hessian
using LinearAlgebra: svd, Diagonal, inv, I

@doc raw"""
`linearize_dynamics(x, u, dynamicsf)`

linearizes the function `dynamicsf` around the point `x` and `u`.

# Arguments
- `x::AbstractVector{T}`: state at a specific step
- `u::AbstractVector{T}`: input at a specific step
- `dynamicsf::Function`: dynamic function, steps the system forward

The `dynamicsf` steps the system forward (``x_{i+1} = f(x_i, u_i)``). The
function expects input of the form:

```julia
function dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T}) where T
    ...
    return xᵢ₊₁
end
```

Returns ``(A, B)``, which are matricies defined below.

``f(x_k, u_k) ≈ A x_k + B u_k``
"""
function linearize_dynamics(x::AbstractVector{T}, u::AbstractVector{T},
                            dynamicsf::Function) where {T}
    state_size = size(x)[1]; control_size = size(u)[1];

    𝐀 = zeros(T, state_size, state_size)
    𝐁 = zeros(T, state_size, control_size)

    # Declaring dynamics jacobian functions
    A_func(x, u) = jacobian(x -> dynamicsf(x, u), x)
    B_func(x, u) = jacobian(u -> dynamicsf(x, u), u)

    # Computing jacobian around each point
    𝐀 .= A_func(x, u)
    𝐁 .= B_func(x, u)

    return (𝐀, 𝐁)
end


@doc raw"""
`immediate_cost_quadratization(x, u, immediate_cost)`

Turns cost function into a quadratic at time step ``i`` around a point ``(xᵢ, uᵢ)``.
Details given in ETH slides.

# Arguments
- `x::AbstractVector{T}`: state at a specific step
- `u::AbstractVector{T}`: input at a specific step
- `immediate_cost::Function`: Cost after each step

The `immediate_cost` function expect input of the form:
```julia
function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
    return sum(u.^2) + sum(target_state - x.^2)  # for example
end
```

!!! note
    It is important that the function `immediate_cost` be an explict function
    of both `x` and `u` (due to issues using `ForwardDiff` Package). If you want
    to make `immediate_cost` practically only dependent on `u` use the following

    ```julia
    function immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})
        return sum(u.^2) + sum(x) * 0.0  # Only dependent on u
    end
    ```

Returns the matricies `(𝑞ᵢ, 𝐪ᵢ, 𝐫ᵢ, 𝐐ᵢ, 𝐏ᵢ, 𝐑ᵢ)` defined as:

``𝑞ᵢ = L(xᵢ,uᵢ)``, ``𝐪ᵢ = \frac{∂L(xᵢ,uᵢ)}{∂x}``, ``𝐫ᵢ = \frac{∂L(xᵢ,uᵢ)}{∂u}``,
``𝐐ᵢ = \frac{∂^2 L(xᵢ,uᵢ)}{∂x^2}``, ``𝐏ᵢ = \frac{∂^2 L(xᵢ,uᵢ)}{∂x ∂u}``,
``𝐑ᵢ = \frac{∂^2 L(xᵢ,uᵢ)}{∂u^2}``
"""
function immediate_cost_quadratization(x::AbstractVector{T},
                                       u::AbstractVector{T},
                                       immediate_cost::Function) where {T}
    state_size = size(x)[1]; control_size = size(u)[1];

    # Notation copied from ETH lecture notes
    𝑞ᵢ = convert(T, 0.)  # Cost along path
    𝐪ᵢ = zeros(T, state_size)  # Cost Jacobian wrt x
    𝐫ᵢ = zeros(T, control_size)  # Cost Jacobian wrt u
    𝐐ᵢ = zeros(T, state_size, state_size)  # Cost Hessian wrt x, x
    𝐏ᵢ = zeros(T, control_size, state_size)  # Cost Hessian wrt u, x
    𝐑ᵢ = zeros(T, control_size, control_size)  # Cost Hessian wrt u, u

    # Helper jacobain functions
    ∂L∂x(x, u) = gradient(x -> immediate_cost(x, u), x)
    ∂L∂u(x, u) = gradient(u -> immediate_cost(x, u), u)
    ∂²L∂x²(x, u) = hessian(x -> immediate_cost(x, u), x)
    ∂²L∂u∂x(x, u) = jacobian(x -> ∂L∂u(x, u), x)
    ∂²L∂u²(x, u) = hessian(u -> immediate_cost(x, u), u)

    𝑞ᵢ = immediate_cost(x, u)
    𝐪ᵢ = ∂L∂x(x, u)        # Cost gradient wrt x
    𝐫ᵢ = ∂L∂u(x, u)        # Cost gradient wrt u
    𝐐ᵢ = ∂²L∂x²(x, u)      # Cost Hessian wrt x, x
    𝐏ᵢ = ∂²L∂u∂x(x, u)     # Cost Hessian wrt u, x
    𝐑ᵢ = ∂²L∂u²(x, u)      # Cost Hessian wrt u, u

    return (𝑞ᵢ, 𝐪ᵢ, 𝐫ᵢ, 𝐐ᵢ, 𝐏ᵢ, 𝐑ᵢ)
end


@doc raw"""
`final_cost_quadratization(x, final_cost)`

Turns final cost function into a quadratic at last time step, `n`, about point
`(xₙ, uₙ)`. Details given in ETH slides.

# Arguments
- `x::AbstractVector{T}`: state at a specific step
- `final_cost::Function`: Cost after final step

The `final_cost` function expect input of the form:
```julia
function final_cost(x::AbstractVector{T})
    return sum(target_state - x.^2)  # for example
end
```

Returns the matricies `(𝑞ₙ, 𝐪ₙ, 𝐐ₙ)` defined as:

``𝑞ₙ = L(xₙ,uₙ)``, ``𝐪ₙ = \frac{∂L(xₙ,uₙ)}{∂x}``, ``𝐐ₙ = \frac{∂^2 L(xₙ,uₙ)}{∂x^2}``
"""
function final_cost_quadratization(x::AbstractVector{T}, final_cost::Function) where {T}
    state_size = size(x)[1];

    # Notation copied from ETH lecture notes
    𝑞ₙ = convert(T, 0.)  # Cost along path
    𝐪ₙ = zeros(T, state_size)  # Cost Jacobian wrt x
    𝐐ₙ = zeros(T, state_size, state_size)  # Cost Hessian wrt x, x

    # Helper jacobain functions
    ∂Φ∂x(x) = gradient(x -> final_cost(x), x)
    ∂²Φ∂x²(x) = hessian(x -> final_cost(x), x)

    # Final cost
    𝑞ₙ = final_cost(x)
    # Final cost gradient wrt x
    𝐪ₙ = ∂Φ∂x(x)
    # Final cost Hessian wrt x, x
    𝐐ₙ = ∂²Φ∂x²(x)

    return (𝑞ₙ, 𝐪ₙ, 𝐐ₙ)
end


@doc raw"""
`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`

Computes optimal control parameters `(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)`, at time step `i`.
These are used in computing feedforward and feedback gains.

# Arguments
- `𝐀ᵢ::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `𝐁ᵢ::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `𝐫ᵢ::AbstractVector{T}`: see output of [`immediate_cost_quadratization(x, u, immediate_cost)`](@ref)
- `𝐏ᵢ::AbstractMatrix{T}`: see output of [`immediate_cost_quadratization(x, u, immediate_cost)`](@ref)
- `𝐑ᵢ::AbstractMatrix{T}`: see output of [`immediate_cost_quadratization(x, u, immediate_cost)`](@ref)
- `𝐬ᵢ₊₁::AbstractVector{T}`: Rollback parameter
- `𝐒ᵢ₊₁::AbstractMatrix{T}`: Rollback parameter

Returns the matricies `(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)` defined as:

``𝐠ᵢ = 𝐫ᵢ + 𝐁ᵢ^T 𝐬ᵢ₊₁``, ``𝐆ᵢ = 𝐏ᵢ + 𝐁ᵢ^T 𝐒ᵢ₊₁ 𝐀ᵢ``,
``𝐇ᵢ = 𝐑ᵢ + 𝐁ᵢ^T 𝐒ᵢ₊₁ 𝐁ᵢ``
"""
function optimal_controller_param(𝐀ᵢ::AbstractMatrix{T}, 𝐁ᵢ::AbstractMatrix{T},
                                  𝐫ᵢ::AbstractVector{T}, 𝐏ᵢ::AbstractMatrix{T},
                                  𝐑ᵢ::AbstractMatrix{T}, 𝐬ᵢ₊₁::AbstractVector{T},
                                  𝐒ᵢ₊₁::AbstractMatrix{T}) where {T}
    𝐠ᵢ = 𝐫ᵢ + 𝐁ᵢ' * 𝐬ᵢ₊₁
    𝐆ᵢ = 𝐏ᵢ + 𝐁ᵢ' * 𝐒ᵢ₊₁ * 𝐀ᵢ
    𝐇ᵢ = 𝐑ᵢ + 𝐁ᵢ' * 𝐒ᵢ₊₁ * 𝐁ᵢ

    return (𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)
end


@doc raw"""
`feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)`

Computes feedforward and feedback gains, ``(𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ)``.

# Arguments
- `𝐠ᵢ::AbstractVector{T}`: see output of [`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`](@ref)
- `𝐆ᵢ::AbstractMatrix{T}`: see output of [`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`](@ref)
- `𝐇ᵢ::AbstractMatrix{T}`: see output of [`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`](@ref)

Returns the matricies `(𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ)` defined as:

``𝛿𝐮ᵢᶠᶠ = - 𝐇ᵢ^{-1} 𝐠ᵢ``, ``𝐊ᵢ = - 𝐇ᵢ^{-1} 𝐆ᵢ``

Because ``𝐇ᵢ`` can be poorly conditioned, the regularized inverse of the matrix
is computed instead of the true inverse.
"""
function feedback_parameters(𝐠ᵢ::AbstractVector{T}, 𝐆ᵢ::AbstractMatrix{T},
                             𝐇ᵢ::AbstractMatrix{T}) where {T}
    # 𝛿𝐮ᵢᶠᶠ = - 𝐇ᵢ \ 𝐠ᵢ
    # 𝐊ᵢ = - 𝐇ᵢ \ 𝐆ᵢ
    # H_inv = regularized_persudo_inverse(𝐇ᵢ)

    n = size(𝐇ᵢ)[1]
    H_inv = inv(𝐇ᵢ + 0.01 * I(n))
    𝛿𝐮ᵢᶠᶠ = - H_inv * 𝐠ᵢ
    𝐊ᵢ = - H_inv * 𝐆ᵢ
    return (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ)
end


function regularized_persudo_inverse(matrix::AbstractMatrix{T}; reg=1e-5) where {T}
    @assert !any(isnan, matrix)

    SVD = svd(matrix)

    SVD.S[ SVD.S .< 0 ] .= 0.0        #truncate negative values...
    diag_s_inv = zeros(T, (size(SVD.V)[1], size(SVD.U)[2]))
    diag_s_inv[1:length(SVD.S), 1:length(SVD.S)] .= Diagonal(1.0 / (SVD.S .+ reg))

    regularized_matrix = SVD.V * diag_s_inv * transpose(SVD.U)
    # println(regularized_matrix)
    return regularized_matrix
end


@doc raw"""
`step_back(𝐀ᵢ, 𝑞ᵢ, 𝐪ᵢ, 𝐐ᵢ, 𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ, 𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ, 𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`

Computes the rollback parameters ``𝑠ᵢ``, ``𝐬ᵢ``, and ``𝐒ᵢ`` for the next step
backward.

# Arguments
- `𝐀ᵢ::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `𝐁ᵢ::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `𝑞ᵢ::T`: see output of [`immediate_cost_quadratization(x, u, immediate_cost)`](@ref)
- `𝐪ᵢ::AbstractVector{T}`: see output of [`immediate_cost_quadratization(x, u, immediate_cost)`](@ref)
- `𝐐ᵢ::AbstractMatrix{T}`: see output of [`immediate_cost_quadratization(x, u, immediate_cost)`](@ref)
- `𝐠ᵢ::AbstractVector{T}`: see output of [`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`](@ref)
- `𝐆ᵢ::AbstractMatrix{T}`: see output of [`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`](@ref)
- `𝐇ᵢ::AbstractMatrix{T}`: see output of [`optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)`](@ref)
- `𝛿𝐮ᵢᶠᶠ::AbstractVector{T}`: see output of [`feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)`](@ref)
- `𝐊ᵢ::AbstractMatrix{T}`: see output of [`feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)`](@ref)
- `𝑠ᵢ₊₁::T`: Rollback parameter
- `𝐬ᵢ₊₁::AbstractVector{T}`: Rollback parameter
- `𝐒ᵢ₊₁::AbstractMatrix{T}`: Rollback parameter

Returns the next-step-back's rollback parameters, ``(𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ)``

Because ``𝐇ᵢ`` can be poorly conditioned, the regularized inverse of the matrix
is computed instead of the true inverse.
"""
function step_back(𝐀ᵢ::AbstractMatrix{T}, 𝑞ᵢ::T, 𝐪ᵢ::AbstractVector{T},
                   𝐐ᵢ::AbstractMatrix{T}, 𝐠ᵢ::AbstractVector{T},
                   𝐆ᵢ::AbstractMatrix{T}, 𝐇ᵢ::AbstractMatrix{T},
                   𝛿𝐮ᵢᶠᶠ::AbstractVector{T}, 𝐊ᵢ::AbstractMatrix{T},
                   𝑠ᵢ₊₁::T, 𝐬ᵢ₊₁::AbstractVector{T}, 𝐒ᵢ₊₁::AbstractMatrix{T}
                   ) where {T}
    𝑠ᵢ = (𝑞ᵢ + 𝑠ᵢ₊₁ + .5 * 𝛿𝐮ᵢᶠᶠ' * 𝐇ᵢ * 𝛿𝐮ᵢᶠᶠ + 𝛿𝐮ᵢᶠᶠ' * 𝐠ᵢ)
    𝐬ᵢ = (𝐪ᵢ + 𝐀ᵢ' * 𝐬ᵢ₊₁ + 𝐊ᵢ' * 𝐇ᵢ * 𝛿𝐮ᵢᶠᶠ + 𝐊ᵢ' * 𝐠ᵢ + 𝐆ᵢ' * 𝛿𝐮ᵢᶠᶠ)
    𝐒ᵢ = (𝐐ᵢ + 𝐀ᵢ' * 𝐒ᵢ₊₁ * 𝐀ᵢ + 𝐊ᵢ' * 𝐇ᵢ * 𝐊ᵢ + 𝐊ᵢ' * 𝐆ᵢ + 𝐆ᵢ' * 𝐊ᵢ)

    return (𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ)
end


@doc raw"""
`backward_pass(x, u, dynamicsf, immediate_cost, final_cost)`

Computes feedforward and feedback gains (``𝛿𝐮ᵢᶠᶠ`` and ``𝐊ᵢ``).

# Arguments
- `x::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
- `u::AbstractMatrix{T}`: see output of [`linearize_dynamics(x, u, dynamicsf)`](@ref)
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

Returns the feedback parameters ``𝛿𝐮ᵢᶠᶠᵢ``, and ``𝐊ᵢ`` for each time step ``i``
"""
function backward_pass(x::AbstractMatrix{T}, u::AbstractMatrix{T},
                       dynamicsf::Function, immediate_cost::Function,
                       final_cost::Function) where {T}
    # Grab all dimensions
    N, state_size = size(x); M, input_size = size(u);
    @assert(N == M+1)

    # Initialize matricies
    𝛿𝐮ᶠᶠs = zeros(T, N-1, input_size)
    𝐊s = zeros(T, N-1, input_size, state_size)

    (𝑞ₙ, 𝐪ₙ, 𝐐ₙ) = final_cost_quadratization(x[N,:], final_cost)
    (𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁) = (𝑞ₙ, 𝐪ₙ, 𝐐ₙ)

    # Move backward
    for i = (N-1):-1:1
        (𝐀ᵢ, 𝐁ᵢ) = linearize_dynamics(x[i,:], u[i,:], dynamicsf)
        (𝑞ᵢ, 𝐪ᵢ, 𝐫ᵢ, 𝐐ᵢ, 𝐏ᵢ, 𝐑ᵢ) = immediate_cost_quadratization(x[i,:], u[i,:], immediate_cost)
        (𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ) = optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)
        (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ) = feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)

        𝛿𝐮ᶠᶠs[i,:] .= 𝛿𝐮ᵢᶠᶠ
        𝐊s[i,:,:] .= 𝐊ᵢ

        (𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ) = step_back(𝐀ᵢ, 𝑞ᵢ, 𝐪ᵢ, 𝐐ᵢ, 𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ, 𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ,
                                𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)
        (𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁) = (𝑠ᵢ, 𝐬ᵢ, 𝐒ᵢ)
    end

    @assert !any(isnan, 𝛿𝐮ᶠᶠs)
    @assert !any(isnan, 𝐊s)

    return (𝛿𝐮ᶠᶠs, 𝐊s)
end
