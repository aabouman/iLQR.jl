@doc raw"""

fit trajectory using Iterative Linear Quadratic Regulator (iLQR).

# Arguments
- `x_init::AbstractMatrix{T}`: state trajectory, `length(x_init) == N + 1`
- `u_init::AbstractMatrix{T}`: control input trajectory, `length(u_init) == N`
- `dynamicsf::Function`: forward descrete dynamics ``f(x_k, u_k)``
- `immediate_cost::Function`: forward descrete dynamics ``f(x_k, u_k)``
- `final_cost::Function`: forward descrete dynamics ``f(x_k, u_k)``
- `max_iter::Int64 = 100`: forward descrete dynamics ``f(x_k, u_k)``
- `tol::Float64 = 1e-6`: tolerance to test for input trajectory convergence. Converged when ``\lVert u_{k+1} - u_k \rVert``

"""
function fit(x_init::AbstractMatrix{T}, u_init::AbstractMatrix{T},
             dynamicsf::Function, immediate_cost::Function,
             final_cost::Function; max_iter::Int64 = 100, tol::Float64 = 1e-6,
             ) where {T}
    x̅ⁱ = x_init
    u̅ⁱ = u_init
    N, input_size = size(u̅ⁱ)
    M, state_size = size(x̅ⁱ)
    @assert(N + 1 == M,
            "size(x_init)[2] == size(u_init)[1], (# of states is 1 more than # of inputs in trajectory)"
            )
    total_cost = total_cost_generator(immediate_cost, final_cost)

    iter = 0
    for iter = 1:max_iter
        # println(x̅ⁱ[end,:])

        𝛿𝐮ᶠᶠs, 𝐊s = backward_pass(x̅ⁱ::AbstractMatrix{T}, u̅ⁱ::AbstractMatrix{T},
                                  dynamicsf::Function, immediate_cost::Function,
                                  final_cost::Function)
        x̅ⁱ⁺¹, u̅ⁱ⁺¹ = forward_pass(x̅ⁱ, u̅ⁱ, 𝛿𝐮ᶠᶠs, 𝐊s, dynamicsf)

        # Check if we have met the tolerance for convergence
        convert(Float64, sum((u̅ⁱ⁺¹ - u̅ⁱ).^2)) <= tol && break
        # Update the current trajectory and input estimates
        x̅ⁱ = x̅ⁱ⁺¹
        u̅ⁱ = u̅ⁱ⁺¹
    end

    return (x̅ⁱ, u̅ⁱ)
end


function total_cost_generator(immediate_cost::Function, final_cost::Function)
    function total_cost(x̅ⁱ, u̅ⁱ)
        N = size(u̅ⁱ)[1]
        sum = 0.

        for i in 1:N
            sum += immediate_cost(x̅ⁱ[i,:], u̅ⁱ[i,:])
        end
        sum += final_cost(x̅ⁱ[end,:])
    end

    return total_cost
end
