using iLQR
using Plots
using Plots.PlotMeasures

# %%
include("2_link_helper_functions.jl")

maximum_iterations = convert(Int64, 1e5)
tolerance = 1e-6
final_distance = .01
num_steps = 500


state_traj = rand(num_steps, 4)
last_state = state_traj[1, :]
input_traj = rand(num_steps, 2)
t = [1.:convert(Float64, num_steps);]
for i = 1:num_steps
    state_traj[i, :] .= dynamicsf(last_state, input_traj[i, :])
    global last_state = state_traj[i, :]
end
x̅ᶠ = state_traj

# %%
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

save_loc = joinpath(@__DIR__, "figures/openloop_2_link.gif")
gif(anim, save_loc, fps = 20)
