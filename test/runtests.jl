using iLQR

# Include cost functions and dyanmics function
include("helper_functions.jl")

# Test iLQR package on 2-link robot
include("test_linearize_dynamics.jl")
include("test_iLQR.jl")
