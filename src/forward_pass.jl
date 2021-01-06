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
function forward_pass(x̅ⁱ::AbstractMatrix{T}, u̅ⁱ::AbstractMatrix{T},
                      𝛿𝐮ᶠᶠs::AbstractMatrix{T}, 𝐊s::AbstractArray{T,3},
                      dynamicsf::Function) where {T}
    N, input_size = size(u̅ⁱ)
    state_size = size(x̅ⁱ)[2]
    x̅ⁱ⁺¹ = zeros(T, N + 1, state_size)
    u̅ⁱ⁺¹ = zeros(T, N, input_size)
    x̅ⁱ⁺¹[1, :] .= x̅ⁱ[1, :]

    for n = 1:N
        u̅ⁱ⁺¹[n, :] .= u̅ⁱ[n, :] + 𝛿𝐮ᶠᶠs[n, :] + 𝐊s[n, :, :] * (x̅ⁱ⁺¹[n, :] - x̅ⁱ[n, :])
        x̅ⁱ⁺¹[n+1, :] .= dynamicsf(x̅ⁱ⁺¹[n, :], u̅ⁱ⁺¹[n, :])
    end

    return (x̅ⁱ⁺¹, u̅ⁱ⁺¹)
end
