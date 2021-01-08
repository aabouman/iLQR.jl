using LinearAlgebra: I
using ForwardDiff: jacobian
using RigidBodyDynamics, StaticArrays,


urdf = "../urdf/6Dof_arm.urdf"
# Load in arm and remove fixed joint from tree
mechanism = parse_urdf(
    urdf;
    scalar_type = Float64,
    floating = false,
    gravity = [0, 0, 0],
    remove_fixed_tree_joints = false,
)
state = MechanismState(mechanism)



function dynamicsf(xᵢ::AbstractVector, uᵢ::AbstractVector)
    n = 13
    q, q̇ = xᵢ[1:n], xᵢ[n:end],

    set_configuration!(state, q)
    set_velocity!(state, q̇)

    result = SVector{size(velocity(state)), Float64}
    dynamics!(result, state, uᵢ)

    # M_mat = mass_matrix(state)
    # C_mat = dynamics_bias(state) # C_mat * q̇

    mat1 = [
        zeros(n, n) I(2)
        zeros(n, n) -M_mat\C_mat
    ]
    mat2 = [zeros(n, n); inv(M_mat)]

    #state_ddot = inv(M_mat) *(ext_wrench - C_mat)
    state_dot = mat1 + mat2 * uᵢ
    new_state = state + Δt * state_dot

    return new_state
end



function dynamicsf(state::AbstractVector, input::AbstractVector)

    dynamics!()
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
