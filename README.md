# iLQR.jl
Environment for working on robot dynamics in Julia. To setup:

    $ cd path/to/JuliaDynamics
    $ julia --project=@.
    julia> ]
    (JuliaDynamics) pkg>

We see that we are now in the JuliaDynamics environment.
If you're working on developing new modules, and are likely to make changes
to them after importing. be sure to first load in the `Revise` package.

    julia> using Revise

## Adding Local Modules
For example, lets add the module `iLQR` to the environment.

    julia> ]
    (JuliaDynamics) pkg> dev path/to/JuliaDynamics/iLQR

We are now free to import the module `iLQR` as you normally would:

    julia> using iLQR

## Removing Local Modules
If you decide to delete a custom module you can remove the module
(in this example `iLQR`) from the environment by running:

    julia> ]
    (JuliaDynamics) pkg> rm iLQR

## Test Modules
Test a local module (assuming their is a `test/runtests.jl` script) (in this
example `iLQR`).

    julia> ]
    (JuliaDynamics) pkg> test iLQR
