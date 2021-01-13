import RigidBodyDynamics


function dynamics(x::AbstractVector{T}, u::AbstractVector{T}) where T
    """Solver state and RBD state differ, note below

         # solver state
         [configuration]
         p = x[1:3] <: q
         r = x[4:6] <: q
         θ = x[7:9] <: q

         [velocity]
         ω = x[10:12]
         vel = x[13:15]
         θ̇ = x[16:18]

         # RBD state
         [configuration]
         quat = x[1:4] <: q
         r = x[5:7] <: q
         θ = x[8:10] <: q

         [velocity]
         ω = x[11:13]
         vel = x[14:16]
         θ̇ = x[17:19]

    """

    # solver state
    # configuration stuff
    p = x[1:3]; r = x[4:6]; θ = x[7:9]
    # velocity stuff
    ω = x[10:12]; vel = x[13:15]; θ̇ = x[16:18]

    # now we convert it to a state for RBD
    state = statecache[T]
    copyto!(state, [q_from_p(p); x[4:end]])

    # get the dynamics for v (this state is the same for both)
    M = Array(mass_matrix(state))
    if hasnan(M)
        return NaN*x
    else
        # dynamics
        v̇ = (M)\(-dynamics_bias(state) + u)

        # kinematics
        q̇ = [pdot_from_w(p,ω);vel;θ̇]

        return [q̇;v̇]
    end
end
