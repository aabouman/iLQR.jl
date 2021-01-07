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
                      dynamicsf::Function) where {T}
    N, input_size = size(u)
    state_size = size(x)[2]
    x̅ = zeros(T, N+1, state_size)
    u̅ = zeros(T, N, input_size)
    x̅[1, :] .= x[1, :]

    display(size(𝐊s))

    for k = 1:N
        δxᵢ = x̅[k, :] - x[k, :]
        u̅[k, :] .= u[k, :] + 𝛿𝐮ᶠᶠs[k, :] + 𝐊s[k,:,:] * δxᵢ
        x̅[k+1, :] .= dynamicsf(x̅[k, :], u̅[k, :])

        @assert(!any(isnan, x̅[k, :]), [k, x̅[k, :]])
        @assert(!any(isnan, x̅[k+1, :]), [k, display(u̅[1:20, :])])
    end

    @assert !any(isnan, u̅)
    @assert !any(isnan, x̅)

    return (x̅, u̅)
end
