var documenterSearchIndex = {"docs":
[{"location":"#iLQR.jl","page":"iLQR.jl","title":"iLQR.jl","text":"","category":"section"},{"location":"","page":"iLQR.jl","title":"iLQR.jl","text":"A simple package for Iterative Linear Quadratic Regulator (iLQR) trajectory optimization.","category":"page"},{"location":"documentation/#iLQR.jl-Documentation","page":"Documentation","title":"iLQR.jl Documentation","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"documentation/#Functions","page":"Documentation","title":"Functions","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"iLQR.fit","category":"page"},{"location":"documentation/#iLQR.fit","page":"Documentation","title":"iLQR.fit","text":"fit trajectory using Iterative Linear Quadratic Regulator (iLQR).\n\nArguments\n\nx_init::AbstractMatrix{T}: state trajectory, length(x_init) == N + 1\nu_init::AbstractMatrix{T}: control input trajectory, length(u_init) == N\ndynamicsf::Function: forward descrete dynamics f(x_k u_k)\nimmediate_cost::Function: forward descrete dynamics f(x_k u_k)\nfinal_cost::Function: forward descrete dynamics f(x_k u_k)\nmax_iter::Int64 = 100: forward descrete dynamics f(x_k u_k)\ntol::Float64 = 1e-6: tolerance to test for input trajectory convergence. Converged when lVert u_k+1 - u_k rVert\n\n\n\n\n\n","category":"function"},{"location":"documentation/#Index","page":"Documentation","title":"Index","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"}]
}
