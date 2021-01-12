# prev_cost = Inf

@doc raw"""

```julia
function dynamics(xᵢ::AbstractArray{T,1}, uᵢ::AbstractArray{T,1})
    ...
    return xᵢ₊₁
end
```

```julia
function immediate_cost(x::AbstractArray{T,2}, u::AbstractArray{T,2})
    sum(u.^2)  # for example
end
```

```julia
function final_cost(xₙ::AbstractArray{T,1})
    sum((some_target_point - xₙ).^2)  # Euclidean distance at end, for example
end
```
"""
function forward_pass(x::AbstractMatrix{T}, u::AbstractMatrix{T},
                      𝛿𝐮ᶠᶠs::AbstractMatrix{T}, 𝐊s::AbstractArray{T,3},
                      prev_cost::T, dynamicsf::Function,
                      immediate_cost::Function, final_cost::Function
                      ) where {T}
    M, input_size = size(u); N, state_size = size(x)
    @assert(N == M+1)

    x̅ = zeros(T, N, state_size); u̅ = zeros(T, N-1, input_size)
    x̅[1, :] .= x[1, :]
    α = 1.0     # Learning rate
    total_cost = total_cost_generator(immediate_cost, final_cost)
    new_cost = 0.0

    while true
        for k = 1:N-1
            δxᵢ = x̅[k,:] - x[k,:]
            u̅[k,:] .= u[k,:] + α * 𝛿𝐮ᶠᶠs[k,:] + 𝐊s[k,:,:] * δxᵢ
            x̅[k+1,:] .= dynamicsf(x̅[k,:], u̅[k,:])
        end
        new_cost = total_cost(x̅, u̅)
        Δcost = prev_cost - new_cost

        if (Δcost > 0)    # Check if we should persue line search
            break
        else    # Catch overshoot as well as NaNs
            α /= 2
            display("Were line searchin'")
            display(α)
            display(any(isnan, u̅))
        end
    end

    @assert !any(isnan, u̅)
    @assert !any(isnan, x̅)

    return (x̅, u̅, new_cost)
end


@doc raw"""
fit usig iLQR
"""
function fit(x_init::AbstractMatrix{T}, u_init::AbstractMatrix{T},
             dynamicsf::Function, immediate_cost::Function,
             final_cost::Function; max_iter::Int64=100, tol::Float64=1e-6,
             ) where {T}
    x̅ⁱ = x_init; u̅ⁱ = u_init
    N, state_size = size(x̅ⁱ); M, input_size = size(u̅ⁱ)
    @assert(N == M + 1,
            "size(x_init)[2] == size(u_init)[1], (# of states is 1 more than # of inputs in trajectory)"
            )
    total_cost = total_cost_generator(immediate_cost, final_cost)

    prev_cost = Inf; new_cost = NaN
    iter = 0
    for iter = 1:max_iter
        𝛿𝐮ᶠᶠs, 𝐊s = backward_pass(x̅ⁱ, u̅ⁱ, dynamicsf, immediate_cost, final_cost)
        x̅ⁱ⁺¹, u̅ⁱ⁺¹, new_cost = forward_pass(x̅ⁱ, u̅ⁱ, 𝛿𝐮ᶠᶠs, 𝐊s, prev_cost,
                                            dynamicsf, immediate_cost,
                                            final_cost)
        @assert(prev_cost > new_cost); prev_cost = new_cost

        # Check if we have met the tolerance for convergence
        display(size(u̅ⁱ))
        display(size(u̅ⁱ⁺¹))

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
