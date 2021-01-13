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

target_tool_loc = [0.6, -0.5]


function InverseKinematics(workspace_target::AbstractVector{T}) where {T}
    x, y = workspace_target

    q₂ = acos((x^2 + y^2 - l₁^2 - l₂^2) / (2 * l₁ * l₂))
    q₁ = atan(y, x) - atan(l₂ * sin(q₂), l₁ + l₂ * cos(q₂))

    return [q₁, q₂]
end


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

function dynamicsf(x_n::AbstractVector, u::AbstractVector)

    function continuous_dynamics(state::AbstractVector, ext_wrench::AbstractVector)
        n = length(state) ÷ 2
        θ = state[1:n]
        θ̇ = state[n+1:end]

        M_mat = InertiaMatrix(θ)
        C_mat = CoriolisMatrix(θ, θ̇ )

        mat1 = [
            zeros(n, n) I(2)
            zeros(n, n) -M_mat\C_mat
        ]
        mat2 = [zeros(n, n); inv(M_mat)]

        state_dot = mat1 * state + mat2 * ext_wrench
        # new_state = state + Δt * state_dot

        return state_dot
    end

    # RK 4 integration
    k1 = Δt * continuous_dynamics(x_n, u)
    k2 = Δt * continuous_dynamics(x_n+k1/2, u)
    k3 = Δt * continuous_dynamics(x_n+k2/2, u)
    k4 = Δt * continuous_dynamics(x_n+k3, u)

    new_state = (x_n + (1/6)*(k1+2*k2+2*k3 + k4))
    return new_state
end


function immediate_cost(x̅ᵢ::AbstractVector, u̅ᵢ::AbstractVector)
    # return sum(u̅ᵢ.^2) + sum(x̅ᵢ) * 0.0
    # return sum(u̅ᵢ.^2) * 0.0 + sum((target_tool_loc .- x̅ᵢ[1:2]).^2)
    state_size = length(x̅ᵢ)
    θ₁, θ₂ = x̅ᵢ[1:(state_size÷2)]
    target_joint_loc = InverseKinematics(target_tool_loc)

    euclidean_penalty = sum((target_joint_loc .- [θ₁, θ₂]).^2)

    Q = [0. 0. 0. 0.; 0. 0. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]
    velocity_penalty = x̅ᵢ' * Q * x̅ᵢ

    torque_penalty = sum(u̅ᵢ.^2)

    return euclidean_penalty * 1.0 + torque_penalty * 1.0
end


function final_cost(x̅ₙ::AbstractVector)
    state_size = length(x̅ₙ)
    θ₁, θ₂ = x̅ₙ[1:(state_size÷2)]
    target_joint_loc = InverseKinematics(target_tool_loc)

    euclidean_penalty = sum((target_joint_loc .- [θ₁, θ₂]).^2)

    return euclidean_penalty * 1.0
end
