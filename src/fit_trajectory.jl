@doc raw"""
fit usig iLQR
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
                                  final_cost::Function,
                                  )
        # println("Backward Pass Outputs")
        # println(any(isnan, 𝛿𝐮ᶠᶠs))
        # println(any(isnan, 𝐊s))

        x̅ⁱ⁺¹, u̅ⁱ⁺¹ = forward_pass(x̅ⁱ, u̅ⁱ, 𝛿𝐮ᶠᶠs, 𝐊s, dynamicsf)
        # println("Forward Pass Outputs")
        # println(any(isnan, x̅ⁱ⁺¹))
        # println(any(isnan, u̅ⁱ⁺¹))
        # println(x̅ⁱ⁺¹)
        # println(u̅ⁱ⁺¹)

        # Check if we have met the tolerance for convergence
        convert(Float64, sum((u̅ⁱ⁺¹ - u̅ⁱ).^2)) <= tol && break
        # Update the current trajectory and input estimates
        x̅ⁱ = x̅ⁱ⁺¹
        u̅ⁱ = u̅ⁱ⁺¹

        # println(total_cost(x̅ⁱ, u̅ⁱ))
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
