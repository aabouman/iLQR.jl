module iLQR

using ForwardDiff: gradient, jacobian, hessian
using LinearAlgebra: svd, Diagonal, inv, I

include("forward_pass.jl")
include("backward_pass.jl")

# include("cost_functions.jl")

end
