var documenterSearchIndex = {"docs":
[{"location":"#iLQR.jl-Documentation","page":"Home","title":"iLQR.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package performs iterative Linear Quadratic Regulation.","category":"page"},{"location":"documentation/#iLQR.jl-Documentation","page":"Documentation","title":"iLQR.jl Documentation","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"documentation/#Index","page":"Documentation","title":"Index","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"documentation/#Trajectory-Optimization","page":"Documentation","title":"Trajectory Optimization","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.fit","category":"page"},{"location":"documentation/#iLQR.fit","page":"Documentation","title":"iLQR.fit","text":"fit(x_init, u_init, dynamicsf, immediate_cost, final_cost; max_iter=100, tol=1e-6)\n\nPerform iterativer LQR to compute optimal input and corresponding state trajectory.\n\nArguments\n\nx_init::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\nu_init::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\ndynamicsf::Function: dynamic function, steps the system forward\nimmediate_cost::Function: Cost after each step\nfinal_cost::Function: Cost after final step\nx_traj::AbstractMatrix{T}=zero(x_init): reference trajectory using in the cost function\nmax_iter::Int64=100: Maximum number of forward/backward passes to make\ntol::Float64=1e-6: Specifies the tolerance at which to consider the input\n\ntrajectory has converged\n\nThe dynamicsf steps the system forward, x_i+1 = f(x_i u_i). The function expects input of the form:\n\nfunction dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{S}) where {T, S}\n    ...\n    return xᵢ₊₁\nend\n\nThe immediate_cost function expect input of the form:\n\nfunction immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})\n    return sum(u.^2) + sum(target_state - x.^2)  # for example\nend\n\nnote: Note\nIt is important that the function immediate_cost be an explict function of both x and u (due to issues using ForwardDiff Package). If you want to make immediate_cost practically only dependent on u with the followingfunction immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})\n    return sum(u.^2) + sum(x) * 0.0  # Only dependent on u\nend\n\nThe final_cost function expect input of the form:\n\nfunction final_cost(x::AbstractVector{T})\n    return sum(target_state - x.^2)  # for example\nend\n\nReturns the optimal trajectory (barx baru)\n\n\n\n\n\n","category":"function"},{"location":"documentation/#Forward-Rollout","page":"Documentation","title":"Forward Rollout","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.forward_pass","category":"page"},{"location":"documentation/#iLQR.forward_pass","page":"Documentation","title":"iLQR.forward_pass","text":"forward_pass(x, u, 𝛿𝐮ᶠᶠs, 𝐊s, prev_cost, dynamicsf, immediate_cost, final_cost)\n\nPerform iterativer LQR to compute optimal input and corresponding state trajectory.\n\nArguments\n\nx::AbstractMatrix: see output of linearize_dynamics(x, u, dynamicsf)\nu::AbstractMatrix: see output of linearize_dynamics(x, u, dynamicsf)\nx_traj::AbstractMatrix: reference trajectory using in the cost function\n𝛿𝐮ᶠᶠs::AbstractMatrix: see output of backward_pass(x, u, dynamicsf, immediate_cost, final_cost)\n𝐊s::AbstractArray{3}: see output of backward_pass(x, u, dynamicsf, immediate_cost, final_cost)\ndynamicsf::Function: dynamic function, steps the system forward\nimmediate_cost::Function: Cost after each step\nfinal_cost::Function: Cost after final step\n\nThe dynamicsf steps the system forward (x_i+1 = f(x_i u_i)). The function expects input of the form:\n\nfunction dynamics(xᵢ::AbstractVector, uᵢ::AbstractVector)\n    ...\n    return xᵢ₊₁\nend\n\nThe immediate_cost function expect input of the form:\n\nfunction immediate_cost(x::AbstractVector, u::AbstractVector)\n    return sum(u.^2) + sum(target_state - x.^2)  # for example\nend\n\nnote: Note\nIt is important that the function immediate_cost be an explict function of both x and u (due to issues using ForwardDiff Package). If you want to make immediate_cost practically only dependent on u with the followingfunction immediate_cost(x::AbstractVector, u::AbstractVector)\n    return sum(u.^2) + sum(x) * 0.0  # Only dependent on u\nend\n\nThe final_cost function expect input of the form:\n\nfunction final_cost(x::AbstractVector)\n    return sum(target_state - x.^2)  # for example\nend\n\nReturns the optimal trajectory (barx baru).\n\n\n\n\n\n","category":"function"},{"location":"documentation/#Backward-Pass","page":"Documentation","title":"Backward Pass","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.backward_pass","category":"page"},{"location":"documentation/#iLQR.backward_pass","page":"Documentation","title":"iLQR.backward_pass","text":"backward_pass(x, u, dynamicsf, immediate_cost, final_cost)\n\nComputes feedforward and feedback gains (delta bf u_i^ff, and bf K_i).\n\nArguments\n\nx::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\nu::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\ndynamicsf::Function: dynamic function, steps the system forward\nimmediate_cost::Function: Cost after each step\nfinal_cost::Function: Cost after final step\n\nThe dynamicsf steps the system forward (x_i+1 = f(x_i u_i)). The function expects input of the form:\n\nfunction dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{T}) where T\n    ...\n    return xᵢ₊₁\nend\n\nThe immediate_cost function expect input of the form:\n\nfunction immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})\n    return sum(u.^2) + sum(target_state - x.^2)  # for example\nend\n\nnote: Note\nIt is important that the function immediate_cost be an explict function of both x and u (due to issues using ForwardDiff Package). If you want to make immediate_cost practically only dependent on u with the followingfunction immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})\n    return sum(u.^2) + sum(x) * 0.0  # Only dependent on u\nend\n\nThe final_cost function expect input of the form:\n\nfunction final_cost(x::AbstractVector{T})\n    return sum(target_state - x.^2)  # for example\nend\n\nReturns the feedback parameters delta bf u_i^ff, and bf K_i for each time step i\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.step_back","category":"page"},{"location":"documentation/#iLQR.step_back","page":"Documentation","title":"iLQR.step_back","text":"step_back(𝐀ᵢ, 𝑞ᵢ, 𝐪ᵢ, 𝐐ᵢ, 𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ, 𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ, 𝑠ᵢ₊₁, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n\nComputes the rollback parameters it s_i, bf s_i, and bf S_i for the next step backward.\n\nArguments\n\n𝐀ᵢ::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\n𝐁ᵢ::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\n𝑞ᵢ::T: see output of immediate_cost_quadratization(x, u, immediate_cost)\n𝐪ᵢ::AbstractVector{T}: see output of immediate_cost_quadratization(x, u, immediate_cost)\n𝐐ᵢ::AbstractMatrix{T}: see output of immediate_cost_quadratization(x, u, immediate_cost)\n𝐠ᵢ::AbstractVector{T}: see output of optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n𝐆ᵢ::AbstractMatrix{T}: see output of optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n𝐇ᵢ::AbstractMatrix{T}: see output of optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n𝛿𝐮ᵢᶠᶠ::AbstractVector{T}: see output of feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)\n𝐊ᵢ::AbstractMatrix{T}: see output of feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)\n𝑠ᵢ₊₁::T: Rollback parameter\n𝐬ᵢ₊₁::AbstractVector{T}: Rollback parameter\n𝐒ᵢ₊₁::AbstractMatrix{T}: Rollback parameter\n\nReturns the next-step-back's rollback parameters, (it s_i bf s_i bf S_i)\n\nBecause 𝐇ᵢ can be poorly conditioned, the regularized inverse of the matrix is computed instead of the true inverse.\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.linearize_dynamics","category":"page"},{"location":"documentation/#iLQR.linearize_dynamics","page":"Documentation","title":"iLQR.linearize_dynamics","text":"linearize_dynamics(x, u, dynamicsf)\n\nlinearizes the function dynamicsf around the point x and u.\n\nArguments\n\nx::AbstractVector{T}: state at a specific step\nu::AbstractVector{S}: input at a specific step\ndynamicsf::Function: dynamic function, steps the system forward\n\nThe dynamicsf steps the system forward (x_i+1 = f(x_i u_i)). The function expects input of the form:\n\nfunction dynamics(xᵢ::AbstractVector{T}, uᵢ::AbstractVector{S}) where {T, S}\n    ...\n    return xᵢ₊₁\nend\n\nReturns (A B), which are matricies defined below.\n\nf(x_k u_k) approx A x_k + B u_k\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.immediate_cost_quadratization","category":"page"},{"location":"documentation/#iLQR.immediate_cost_quadratization","page":"Documentation","title":"iLQR.immediate_cost_quadratization","text":"immediate_cost_quadratization(x, u, immediate_cost)\n\nTurns cost function into a quadratic at time step i around a point (xᵢ uᵢ). Details given in ETH slides.\n\nArguments\n\nx::AbstractVector{T}: state at a specific step\nu::AbstractVector{T}: input at a specific step\nimmediate_cost::Function: Cost after each step\n\nThe immediate_cost function expect input of the form:\n\nfunction immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})\n    return sum(u.^2) + sum(target_state - x.^2)  # for example\nend\n\nnote: Note\nIt is important that the function immediate_cost be an explict function of both x and u (due to issues using ForwardDiff Package). If you want to make immediate_cost practically only dependent on u use the followingfunction immediate_cost(x::AbstractVector{T}, u::AbstractVector{T})\n    return sum(u.^2) + sum(x) * 0.0  # Only dependent on u\nend\n\nReturns the matricies (𝑞ᵢ, 𝐪ᵢ, 𝐫ᵢ, 𝐐ᵢ, 𝐏ᵢ, 𝐑ᵢ) defined as:\n\nit q_i = L(x_i u_i), bf q_i = fracpartial L(x_i u_i)partial x, bf r_i = fracpartial L(x_i u_i)partial u, bf Q_i = fracpartial^2 L(x_i u_i)partial x^2, bf P_i = fracpartial^2 L(x_i u_i)partial x partial u, bf R_i = fracpartial^2 L(x_i u_i)partial u^2\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.final_cost_quadratization","category":"page"},{"location":"documentation/#iLQR.final_cost_quadratization","page":"Documentation","title":"iLQR.final_cost_quadratization","text":"final_cost_quadratization(x, final_cost)\n\nTurns final cost function into a quadratic at last time step, n, about point (xₙ, uₙ). Details given in ETH slides.\n\nArguments\n\nx::AbstractVector{T}: state at a specific step\nfinal_cost::Function: Cost after final step\n\nThe final_cost function expect input of the form:\n\nfunction final_cost(x::AbstractVector{T})\n    return sum(target_state - x.^2)  # for example\nend\n\nReturns the matricies ({\\it q}_n, {\\bf q}_n, {\\it Q}_n) defined as:\n\nit q_n = L(x_n u_n), bf q_n = fracpartial L(x_n u_n)partial x, bf Q_n = fracpartial^2 L(x_n u_n)partial x^2\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.optimal_controller_param","category":"page"},{"location":"documentation/#iLQR.optimal_controller_param","page":"Documentation","title":"iLQR.optimal_controller_param","text":"optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n\nComputes optimal control parameters (𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ), at time step i. These are used in computing feedforward and feedback gains.\n\nArguments\n\n𝐀ᵢ::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\n𝐁ᵢ::AbstractMatrix{T}: see output of linearize_dynamics(x, u, dynamicsf)\n𝐫ᵢ::AbstractVector{T}: see output of immediate_cost_quadratization(x, u, immediate_cost)\n𝐏ᵢ::AbstractMatrix{T}: see output of immediate_cost_quadratization(x, u, immediate_cost)\n𝐑ᵢ::AbstractMatrix{T}: see output of immediate_cost_quadratization(x, u, immediate_cost)\n𝐬ᵢ₊₁::AbstractVector{T}: Rollback parameter\n𝐒ᵢ₊₁::AbstractMatrix{T}: Rollback parameter\n\nReturns the matricies (𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ) defined as:\n\nbf g_i = bf r_i + bf B_i^T bf s_i+1, bf G_i = bf P_i + bf B_i^T bf S_i+1 bf A_i, bf H_i = bf R_i + bf B_i^T bf S_i+1 bf B_i\n\n\n\n\n\n","category":"function"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.feedback_parameters","category":"page"},{"location":"documentation/#iLQR.feedback_parameters","page":"Documentation","title":"iLQR.feedback_parameters","text":"feedback_parameters(𝐠ᵢ, 𝐆ᵢ, 𝐇ᵢ)\n\nComputes feedforward and feedback gains, (delta bf u_i^ff bf K_i).\n\nArguments\n\n𝐠ᵢ::AbstractVector{T}: see output of optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n𝐆ᵢ::AbstractMatrix{T}: see output of optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n𝐇ᵢ::AbstractMatrix{T}: see output of optimal_controller_param(𝐀ᵢ, 𝐁ᵢ, 𝐫ᵢ, 𝐏ᵢ, 𝐑ᵢ, 𝐬ᵢ₊₁, 𝐒ᵢ₊₁)\n\nReturns the matricies (𝛿𝐮ᵢᶠᶠ, 𝐊ᵢ) defined as:\n\ndelta bf u_i^ff = - bf H_i^-1 bf g_i, bf K_i = - bf H_i^-1 bf G_i\n\nBecause bf H_i can be poorly conditioned, the regularized inverse of the matrix is computed instead of the true inverse.\n\n\n\n\n\n","category":"function"}]
}