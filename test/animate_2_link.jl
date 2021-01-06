using iLQR
using Plots
using Plots.PlotMeasures

include("2_link_helper_functions.jl")

maximum_iterations = convert(Int64, 1e5)
tolerance = 1e-6
final_distance = .01
num_steps = 100
state_traj = rand(num_steps+1, n_links * 2)
last_state = state_traj[1, :]
input_traj = rand(num_steps, n_links)

# # Build vector of time steps
# t = [0.:convert(Float64, num_steps);]
# # Fit using iLQR
# (x̅ᶠ, u̅ᶠ) = iLQR.fit(state_traj, input_traj, dynamicsf, immediate_cost,
#                     final_cost; max_iter = maximum_iterations, tol = tolerance,
#                     )
#

state_traj = rand(num_steps, 4)
last_state = state_traj[1, :]
linearized_states = zeros(num_steps, 4)
input_traj = rand(num_steps, 2)
(𝐀s, 𝐁s) = iLQR.linearize_dynamics(state_traj, input_traj, dynamicsf)
t = [1.:convert(Float64, num_steps);]
for i = 1:num_steps
    state_traj[i, :] .= dynamicsf(last_state, input_traj[i, :])
    linearized_states[i, :] .= 𝐀s[i, :, :] * last_state + 𝐁s[i, :, :] * input_traj[i, :]

    global last_state = state_traj[i, :]
end
x̅ᶠ = state_traj

# p1 = plot(t[1:10:end], θ̂s[1:10:end, 1], ribbon=Ps[1:10:end,1,1], fillalpha=.25,
#           label="Estimated θ₁")
# plot!(p1, t, θs[:, 1], label="Ground Truth θ₁")
# p2 = plot(t[1:10:end], θ̂s[1:10:end, 2], ribbon=Ps[1:10:end,2,2], fillalpha=.25,
#           label="Estimated θ₂")
# plot!(p2, t, θs[:, 2], label="Ground Truth θ₂")
# p3 = plot(p1, p2, layout = (1, 2), size=(800, 400), dpi=1000)
# savefig(p3, "../figures/EstimatedState_1Link_Spring.png")

df = 1
anim = @animate for t = 1:df:length(t)
    p4 = plot([0, l₁ * cos(x̅ᶠ[t, 1]), l₁ * cos(x̅ᶠ[t, 1]) + l₂ * cos(x̅ᶠ[t, 1] + x̅ᶠ[t, 2])],
              [0, l₁ * sin(x̅ᶠ[t, 1]), l₁ * sin(x̅ᶠ[t, 1]) + l₂ * sin(x̅ᶠ[t, 1] + x̅ᶠ[t, 2])],
              alpha=0.5, linewidth=5, color=:red, markershape=:circle, label="",
              aspect_ratio=:equal, size=(400,400));
    plot!(p4, [0, l₁ * cos(x̅ᶠ[t, 1]), l₁ * cos(x̅ᶠ[t, 1]) + l₂ * cos(x̅ᶠ[t, 1] + x̅ᶠ[t, 2])],
          [0, l₁ * sin(x̅ᶠ[t, 1]), l₁ * sin(x̅ᶠ[t, 1]) + l₂ * sin(x̅ᶠ[t, 1] + x̅ᶠ[t, 2])],
          alpha=0.2, linewidth=5, color=:blue, markershape=:circle, label="",)
    xlims!((-2, 2))
    ylims!((-2, 2))
    # axis("tight")
end
gif(anim, "Driven_1Link_Spring.gif", fps = 20)
