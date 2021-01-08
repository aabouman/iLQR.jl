using Documenter
# push!(LOAD_PATH,"../src/")
using iLQR

# Documenter.makedocs(
#     root = "./",
#     source = "src",
#     build = "build",
#     clean = true,
#     doctest = false,
#     modules = Module[iLQR],
#     repo = "",
#     highlightsig = true,
#     sitename = "iLQR_Documentation",
#     expandfirst = [],
#     pages = ["Index" => "index.md"],
# )

Documenter.makedocs(
    sitename = "iLQR.jl",
    repo = "https://github.com/aabouman/iLQR.jl",
    pages = [
        "Home" => "index.md",
        "Documentation" => "documentation.md",
        "Tutorial" => "tutorial.md"
    ]
)

deploydocs(;
    repo="https://github.com/aabouman/iLQR.jl",
)
