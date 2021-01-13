@doc raw"""
Returns a simple `final_cost` function that is just the weighted euclidean
distance from the specified tool location to a target location.
"""
function simple_final_cost(
    mechanism::Mechanism{T},
    body::RigidBody{T},
    point::Point3D{SVector{3,T}},
    final_target::AbstractArray{T,1},
    weight::T,
) where {T}
    @assert 3 == length(final_target)

    state = MechanismState(mechanism)

    function final_cost(xₙ::AbstractArray{T,1})
        work_space_traj = zeros(length(final_target))

        set_configuration!(state, xₙ)
        work_space_traj .= (transform_to_root(state, body) * point).v
        final_dist = sum((work_space_traj[end] .- transpose(final_target)) .^ 2)

        return weight * final_dist
    end

    return final_cost
end


@doc raw"""
Returns a simple `immediate_cost` function that is just the sum squared applied
torques.
"""
function simple_immediate_cost(
    mechanism::Mechanism{T},
    body::RigidBody{T},
    point::Point3D{SVector{3,T}},
    final_target::AbstractArray{T,1},
    weight::T,
) where {T}
    @assert 3 == length(final_target)

    state = MechanismState(mechanism)

    function immediate_cost(xᵢ::AbstractArray{T,1}, uᵢ::AbstractArray{T,1})
        control_size = size(uᵢ)
        state_size = length(xᵢ)
        work_space_traj = (transform_to_root(state, body) * point).v

        return sum(uᵢ .^ 2)
    end

    return immediate_cost
end
