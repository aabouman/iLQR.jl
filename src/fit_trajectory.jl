using LinearAlgebra: norm

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
    @assert(
        N + 1 == M,
        "size(x_init)[2] == size(u_init)[1], (# of states is 1 more than # of inputs in trajectory)"
    )

    iter = 0
    for iter = 1:max_iter
        𝛿𝐮ᶠᶠs, 𝐊s = backward_pass(x̅ⁱ::AbstractMatrix{T}, u̅ⁱ::AbstractMatrix{T},
                                  dynamicsf::Function, immediate_cost::Function,
                                  final_cost::Function,
                                  )
        x̅ⁱ⁺¹, u̅ⁱ⁺¹ = forward_pass(x̅ⁱ, u̅ⁱ, 𝛿𝐮ᶠᶠs, 𝐊s, dynamicsf)
        # Check if we have met the tolerance for convergence
        convert(Float64, norm(u̅ⁱ⁺¹ - u̅ⁱ)) <= tol && break
        # Update the current trajectory and input estimates
        x̅ⁱ = x̅ⁱ⁺¹
        u̅ⁱ = u̅ⁱ⁺¹
    end

    return (x̅ⁱ, u̅ⁱ)
end
