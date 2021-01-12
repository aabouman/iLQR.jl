using iLQR
using Plots
using Plots.PlotMeasures

include("2_link_helper_functions.jl")

maximum_iterations = convert(Int64, 1e6)
tolerance = 1e-6
num_steps = 900

input_traj = zeros(num_steps, n_links)
state_traj = zeros(num_steps+1, n_links*2)
state_traj[1,:] = [.1, -.1, 0., 0.] #rand(4)
for i = 1:num_steps
      state_traj[i+1,:] .= dynamicsf(state_traj[i,:], input_traj[i, :])
end


# Build vector of time steps
t = [0.:convert(Float64, num_steps);]
# Fit using iLQR
(x̅ᶠ, u̅ᶠ) = iLQR.fit(state_traj, input_traj, dynamicsf, immediate_cost,
                    final_cost; max_iter = maximum_iterations, tol = tolerance,
                    )

println("Animating...")
df = 10
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
gif(anim, "iLQR_2_link_quad_4.gif", fps = 20)
