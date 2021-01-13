using RigidBodyDynamics, Attitude


# Load up model URDF
urdf = joinpath(dirname(@__FILE__) ,"../urdf/2Dof_arm.urdf")
mechanism = parse_urdf(urdf, gravity = [0.; 0.; 0.], floating = true)
state = MechanismState(mechanism)
set_configuration!(state, [0.,0.,0.,1,.5,.75,1.,0.,0.])


"""
Converts RBD.jl's MechanisimState to the corresponding [q;v] state vector of
poitions and velocities.
"""
function mech_state_to_vec(state::MechanismState{T}) where T
    return [configuration(state);velocity(state)]
end


"""
Converts quaternion orienation state to an MRP orientaiton state
"""
function RBD_to_iLQR_state(x::AbstractVector{T}) where T
    q = x[1:4]
    return [p_from_q(q); x[5:end]]
end


"""
Converts MRP orienation state to an quaternion orientaiton state
"""
function iLQR_to_RBD_state(x::AbstractVector{T}) where T
    p = x[1:3]
    return [q_from_p(p); x[4:end]]
end


"""
Wraps RBD.jl dynamics function with a rk4 integrator descrete dynamics function
for use with the iLQR.jl package.
"""
function dynamicsf(x::AbstractVector, u::AbstractVector)
    # Continous function wrapped around RBD library's forward dynamics function
    function continuous_dynamics(x̅ᵢ::AbstractVector{T}, u̅ᵢ::AbstractVector) where T
        p = x̅ᵢ[1:3];  r = x̅ᵢ[4:6];   θ = x̅ᵢ[7:8]    # pose
        ω = x̅ᵢ[9:11]; v = x̅ᵢ[12:14]; θ̇ = x̅ᵢ[15:16]  # velocity

        # now we convert it to a state for RBD
        𝑞 = [q_from_p(p); r; θ]; 𝑣 = [ω; v; θ̇];

        state = MechanismState{T}(mechanism)
        set_configuration!(state, 𝑞); set_velocity!(state, 𝑣);

        # get the dynamics for v (this state is the same for both)
        M = Array(mass_matrix(state))
        if hasnan(M)
            return NaN * x̅ᵢ
        else
            v̇ = M \ (-dynamics_bias(state) + u̅ᵢ) # dynamics
            q̇ = [pdot_from_w(p,ω); v; θ̇]        # kinematics

            return [q̇;v̇]
        end
    end

    # RK 4 integration
    k1 = Δt * continuous_dynamics(x, u)
    k2 = Δt * continuous_dynamics(x + k1/2, u)
    k3 = Δt * continuous_dynamics(x + k2/2, u)
    k4 = Δt * continuous_dynamics(x + k3, u)

    new_state = x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state
end


"""
Immediate cost function, evaluated after each time step in the trajectory
"""
function immediate_cost(x̅ᵢ::AbstractVector, u̅ᵢ::AbstractVector)
    pos = x̅ᵢ[1:8]

    euclidean_penalty = sum((target_pose .- pos).^2)

    # Q = [0. 0. 0. 0.; 0. 0. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]
    # velocity_penalty = x̅ᵢ' * Q * x̅ᵢ

    torque_penalty = sum(u̅ᵢ.^2)

    return euclidean_penalty * 10.0 + torque_penalty * 1.0
end


"""
Immediate cost function, evaluated after each time step in the trajectory
"""
function final_cost(x̅ₙ::AbstractVector)
    pos = x̅ₙ[1:8]

    euclidean_penalty = sum((target_pose .- pos).^2)

    return euclidean_penalty * 10000.0
end
