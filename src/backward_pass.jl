using ForwardDiff: gradient, jacobian, hessian
using LinearAlgebra: svd, Diagonal, inv, I

"""
Propegates the system dynamics
```
function dynamicsf(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T}) where T
    ...
    return xᵢ₊₁
end
```
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


"""
```julia
function immediate_cost(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T})
    return sum(uᵢ.^2)  # for example
end
```
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


"""
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


"""
Computes the feedback_parameters
"""
function feedback_parameters(𝐠ᵢ::AbstractVector{T}, 𝐆ᵢ::AbstractMatrix{T},
                             𝐇ᵢ::AbstractMatrix{T}) where {T}
    # 𝛿𝐮ᵢᶠᶠ = - 𝐇ᵢ \ 𝐠ᵢ
    # 𝐊ᵢ = - 𝐇ᵢ \ 𝐆ᵢ
    # TODO: Test for when this becomes unstable
    H_inv = regularized_persudo_inverse(𝐇ᵢ)

    # n = size(𝐇ᵢ)[1]
    # H_inv = inv(𝐇ᵢ + 0.01 * I(n))
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
    return regularized_matrix
end


"""
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


"""

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

    return (𝛿𝐮ᶠᶠs, 𝐊s)
end
