using LinearAlgebra: norm, I
using ForwardDiff: jacobian

n_links = 2
l₁, l₂ = sqrt(2.)/2., sqrt(2.)/2.         # Link lengths
r₁, r₂ = 0.5 * l₁, 0.5 * l₂   # Length from joint to link's COM
m₁, m₂ = 1.0, 1.0         # Link masses
Iz1, Iz2 = 1.0/12.0*m₁*l₁^2, 1.0/12.0*m₂*l₂^2   # Link inertias


α = Iz1 + Iz2 + m₁ * r₁^2 + m₂ * (l₁^2 + r₂^2)
β = m₂ * l₁ * r₂
δ = Iz2 + m₂ * r₂^2
Δt = 0.01

target_tool_loc = [0.6, 0.0]


function InertiaMatrix(θ::AbstractVector{T}) where {T}
    retmat = [α+2*β*cos(θ[2]) δ+β*cos(θ[2])
              δ+β*cos(θ[2]) δ]
    return retmat
end


function CoriolisMatrix(θ::AbstractVector{T}, θ̇::AbstractVector{T}) where {T}
    ∇M = jacobian(InertiaMatrix, θ)
    ∇M = reshape(∇M, (length(θ), length(θ), length(θ)))
    C = zeros(T, length(θ), length(θ))

    for i = 1:length(θ), j = 1:length(θ)
        C[i, j] = sum([
            1 / 2 * (∇M[k, i, j] + ∇M[j, i, k] - ∇M[i, k, j]) * θ̇[k] for k in length(θ)
        ])
    end
    return C
end


function dynamicsf(state::AbstractVector, ext_wrench::AbstractVector)
    n = length(state) ÷ 2
    θ = state[1:n]
    θ̇ = state[n+1:end]

    M_mat = InertiaMatrix(θ)
    C_mat = CoriolisMatrix(θ, θ̇)

    mat1 = [
        zeros(n, n) I(2)
        zeros(n, n) -M_mat\C_mat
    ]
    mat2 = [zeros(n, n); inv(M_mat)]

    state_dot = mat1 * state + mat2 * ext_wrench
    new_state = state + Δt * state_dot

    return new_state
end


function immediate_cost(x̅ᵢ::AbstractVector, u̅ᵢ::AbstractVector)
    # return sum(u̅ᵢ.^2) + sum(x̅ᵢ) * 0.0
    # return sum(u̅ᵢ.^2) * 0.0 + sum((target_tool_loc .- x̅ᵢ[1:2]).^2)
    state_size = length(x̅ᵢ)
    θ₁, θ₂ = x̅ᵢ[1:(state_size÷2)]
    x = l₁ * cos(θ₁) + l₂ * cos(θ₁ + θ₂)
    y = l₁ * sin(θ₁) + l₂ * sin(θ₁ + θ₂)
    euclidean_penalty = sum((target_tool_loc .- [x, y]).^2)

    Q = [0. 0. 0. 0.; 0. 0. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]
    velocity_penalty = x̅ᵢ' * Q * x̅ᵢ

    torque_penalty = sum(u̅ᵢ.^2)

    return euclidean_penalty * 1.0 + torque_penalty * 1.0
end


function final_cost(x̅ₙ::AbstractVector)
    state_size = length(x̅ₙ)
    θ₁, θ₂ = x̅ₙ[1:(state_size÷2)]
    x = l₁ * cos(θ₁) + l₂ * cos(θ₁ + θ₂)
    y = l₁ * sin(θ₁) + l₂ * sin(θ₁ + θ₂)
    euclidean_penalty = sum((target_tool_loc .- [x, y]).^2)

    return euclidean_penalty * 1.0
end
