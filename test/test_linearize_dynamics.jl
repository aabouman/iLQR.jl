using iLQR
# using LinearAlgebra
# using ForwardDiff: jacobian
# using Plots


state_traj = rand(100, 4)
last_state = states[1, :]
input_traj = rand(100, 2)
(𝐀s, 𝐁s) = iLQR.linearize_dynamics(state_traj, input_traj, dynamicsf)

for i = 1:100
    states[i, :] .= dynamicsf(last_state, input_traj[i, :])
    linearized_states[i, :] .= 𝐀s[i, :, :] * last_state + 𝐁s[i, :, :] * input_traj[i, :]

    global last_state = states[i, :]
end

# p1 = plot(states[:,1], label="θ₁")
# plot!(p1, states[:,2], label="θ₂")
# plot!(p1, linearized_states[:,1], label="Linearized θ₁")
# plot!(p1, linearized_states[:,2], label="Linearized θ₂")

# Check that the linearized dynamics are ≈ to the true dynamics
@test abs(mean((states-linearized_states)[:, 1:2])) < 1e-10
