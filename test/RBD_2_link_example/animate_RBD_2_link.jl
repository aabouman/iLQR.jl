using iLQR
using MeshCatMechanisms

# set simulation parameters- length of time step, # of steps
Δt = 0.01; num_steps = 1000;
# target pose is size 8 (3 orr, 3 pos, 2 joint)
target_pose = [0., 0., 0., 5., 1., 2., 1., .3]

# add in helper functions
include("RBD_helper_functions.jl")

# set solver parameters, # of iLQR iterations and tolerance til convergence
maximum_iterations = convert(Int64, 1e6); tolerance = 1e-6;

# initalize trajectories
input_traj = zeros(num_steps, length(target_pose))
state_traj = zeros(num_steps+1, length(target_pose)*2)
# Set inital configuration
state_traj[1,:] = RBD_to_iLQR_state(mech_state_to_vec(state))
for i = 1:num_steps
      state_traj[i+1,:] .= dynamicsf(state_traj[i,:], input_traj[i, :])
end


# Build vector of time steps
ts = [0.:Δt:(Δt*num_steps);]
# Fit using iLQR
(x̅ᶠ, u̅ᶠ) = iLQR.fit(state_traj, input_traj, dynamicsf, immediate_cost,
                    final_cost; max_iter = maximum_iterations, tol = tolerance)

x_RBD = zeros(size(x̅ᶠ)[1], size(x̅ᶠ)[2]+1)
for i in 1:size(x̅ᶠ)[1]
    x_RBD[i,:] .= iLQR_to_RBD_state(x̅ᶠ[i,:])
end

q_RBD = x_RBD[:,1:9]
v_RBD = x_RBD[:,10:17]

q_RBD_lol = [q_RBD[i,:] for i in 1:size(q_RBD)[1]]
v_RBD_lol = [x_RBD[i,:] for i in 1:size(x_RBD)[1]]

println("Animating...")
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf));
open(mvis)
MeshCatMechanisms.animate(mvis, ts, q_RBD_lol; realtimerate = 1.);
