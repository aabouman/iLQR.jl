using iLQR
using Test

maximum_iterations = 1e5
tolerance = 1e-6
final_distance = .01

inital_state = rand(4)
# state_traj = rand(101, 4)
state_traj = repeat(inital_state, 101, 1)'
last_state = state_traj[1, :]
# input_traj = rand(100, 2)
input_traj = zeros(100, 2)

(x̅ᶠ, u̅ᶠ) = iLQR.fit(state_traj, input_traj, dynamicsf, immediate_cost,
                    final_cost; max_iter = maximum_iterations, tol = tolerance,
                    )
# Check that the euclidean distance from tool to target is within tolerance
@test final_cost(x̅ᶠ[end,:]) < final_distance

# convert(Float64, sum((target_tool_loc - x̅ᶠ[end,:]).^2)) < final_distance
